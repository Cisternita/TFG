#!/usr/bin/env python3
"""Entrena los modelos del notebook (problemas A, B y C) con un representante de
cada familia (lineal, bagging, boosting), guarda los pipelines en .joblib y
exporta un JSON consumible por el frontend Svelte.

Mantiene la lógica de feature engineering del notebook
``main_arreglado_markdown_final.ipynb``: panel país-semana, lags 1/2/4/12,
medias móviles y sumas acumuladas calculadas a partir de información rezagada,
ratios de persistencia y escalada, macro contemporánea y rezagada dos años, e
interacciones macro-conflicto. El target principal son los eventos disruptivos
en t+1; los problemas multietiqueta cubren tipos principales (B) y subtipos
seleccionados (C)."""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
WEEK_FREQ = "W-SAT"

# Conjunto de tipos de evento "violentos" según el notebook.
VIOLENT_EVENT_TYPES = {
    "Riots",
    "Battles",
    "Violence against civilians",
    "Explosions/Remote violence",
}

EVENT_TYPE_ORDER = [
    "Protests",
    "Riots",
    "Battles",
    "Violence against civilians",
    "Explosions/Remote violence",
    "Strategic developments",
]

STRATEGIC_SUBTYPES = {
    "Change to group/activity",
    "Agreement",
    "Non-state actor overtakes territory",
    "Government regains territory",
    "Non-violent transfer of territory",
    "Headquarters or base established",
}

COUNTRY_NAME_MAP = {"Czechia": "Czech Republic", "Kosovo*": "Kosovo"}

TERRITORY_EXCLUSIONS = {
    "Akrotiri and Dhekelia",
    "Bailiwick of Guernsey",
    "Bailiwick of Jersey",
    "Faroe Islands",
    "Gibraltar",
    "Greenland",
    "Isle of Man",
}
CENTRAL_ASIA_EXCLUSIONS = {
    "Kazakhstan",
    "Kyrgyzstan",
    "Tajikistan",
    "Turkmenistan",
    "Uzbekistan",
}
AMBIGUOUS_EXCLUSIONS = {"Armenia", "Azerbaijan", "Georgia", "Russia"}

EXCLUDED_COUNTRIES = (
    TERRITORY_EXCLUSIONS | CENTRAL_ASIA_EXCLUSIONS | AMBIGUOUS_EXCLUSIONS
)


# ---------------------------------------------------------------------------
# Utilidades numéricas
# ---------------------------------------------------------------------------

def to_float(value: Any, decimals: int = 6) -> float:
    if value is None:
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(result) or math.isinf(result):
        return result
    return float(np.round(result, decimals))


def safe_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return float(np.round(result, 4))


# ---------------------------------------------------------------------------
# Carga de fuentes
# ---------------------------------------------------------------------------

def load_conflict(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df["WEEK"] = pd.to_datetime(df["WEEK"])
    return df.sort_values(["WEEK", "COUNTRY"]).reset_index(drop=True)


def parse_eurostat(
    path: Path,
    sheet_name: str,
    header_row_idx: int,
    start_row_idx: int,
    value_name: str,
) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    stop_words = {"Special value", ":", "Observation flags:", "b", "d", "e", "p", "[0]"}
    year_row = raw.iloc[header_row_idx]
    year_columns: list[tuple[int, int]] = []
    for col_idx, value in year_row.items():
        if pd.notna(value):
            try:
                year_columns.append((col_idx, int(float(value))))
            except (TypeError, ValueError):
                continue

    records: list[dict[str, Any]] = []
    for row_idx in range(start_row_idx, len(raw)):
        geo = raw.iat[row_idx, 0]
        if pd.isna(geo):
            continue
        geo = str(geo).strip()
        if geo in stop_words:
            break
        for col_idx, year in year_columns:
            cell = raw.iat[row_idx, col_idx]
            if pd.isna(cell) or cell == ":":
                continue
            try:
                cell_value = float(cell)
            except (TypeError, ValueError):
                continue
            records.append(
                {
                    "COUNTRY": COUNTRY_NAME_MAP.get(geo, geo),
                    "year": year,
                    value_name: cell_value,
                }
            )
    return pd.DataFrame(records)


def load_macro(gdp: Path, unemp: Path, infl: Path) -> dict[str, pd.DataFrame]:
    return {
        "gdp": parse_eurostat(gdp, "Sheet 1", 8, 10, "gdp_pc"),
        "unemployment": parse_eurostat(unemp, "Sheet 1", 9, 11, "unemployment_rate"),
        "inflation": parse_eurostat(infl, "Sheet 6", 8, 10, "inflation_rate"),
    }


# ---------------------------------------------------------------------------
# Preparación de la muestra europea
# ---------------------------------------------------------------------------

def filter_europe(conflict_df: pd.DataFrame) -> pd.DataFrame:
    keep = ~conflict_df["COUNTRY"].isin(EXCLUDED_COUNTRIES)
    europe = conflict_df.loc[keep].copy()
    europe["COUNTRY"] = europe["COUNTRY"].replace(COUNTRY_NAME_MAP)
    europe["is_violent_type"] = europe["EVENT_TYPE"].isin(VIOLENT_EVENT_TYPES).astype(int)
    europe["is_disruptive_target"] = (
        (europe["EVENT_TYPE"] != "Strategic developments")
        & (europe["SUB_EVENT_TYPE"] != "Peaceful protest")
    ).astype(int)
    europe["year"] = europe["WEEK"].dt.year
    europe["quarter"] = europe["WEEK"].dt.quarter
    return europe.sort_values(["COUNTRY", "WEEK"]).reset_index(drop=True)


def select_modelable_subtypes(europe_df: pd.DataFrame) -> list[str]:
    stats = (
        europe_df.loc[
            europe_df["EVENT_TYPE"] != "Strategic developments",
            ["SUB_EVENT_TYPE", "EVENTS", "WEEK"],
        ]
        .groupby("SUB_EVENT_TYPE")
        .agg(total=("EVENTS", "sum"), weeks=("WEEK", "nunique"))
        .reset_index()
        .sort_values(["total", "weeks"], ascending=False)
    )
    keep = stats[
        (stats["total"] >= 150)
        & (stats["weeks"] >= 40)
        & (~stats["SUB_EVENT_TYPE"].isin(STRATEGIC_SUBTYPES))
    ]
    return keep.head(11)["SUB_EVENT_TYPE"].tolist()


def build_macro_master(macro: dict[str, pd.DataFrame]):
    base = (
        macro["gdp"][["COUNTRY", "year", "gdp_pc"]]
        .merge(
            macro["unemployment"][["COUNTRY", "year", "unemployment_rate"]],
            on=["COUNTRY", "year"],
            how="outer",
        )
        .merge(
            macro["inflation"][["COUNTRY", "year", "inflation_rate"]],
            on=["COUNTRY", "year"],
            how="outer",
        )
        .sort_values(["COUNTRY", "year"])
        .reset_index(drop=True)
    )

    base["gdp_pc_growth"] = (
        base.groupby("COUNTRY")["gdp_pc"].pct_change().replace([np.inf, -np.inf], np.nan)
    )
    base["unemployment_change"] = base.groupby("COUNTRY")["unemployment_rate"].diff()
    base["inflation_change"] = base.groupby("COUNTRY")["inflation_rate"].diff()

    for year, idx in base.groupby("year").groups.items():
        chunk = base.loc[idx]
        infl_q3 = chunk["inflation_rate"].quantile(0.75)
        unemp_q3 = chunk["unemployment_rate"].quantile(0.75)
        gdp_q1 = chunk["gdp_pc"].quantile(0.25)
        base.loc[idx, "high_inflation"] = (chunk["inflation_rate"] >= infl_q3).astype(float)
        base.loc[idx, "high_unemployment"] = (
            chunk["unemployment_rate"] >= unemp_q3
        ).astype(float)
        base.loc[idx, "low_income"] = (chunk["gdp_pc"] <= gdp_q1).astype(float)

        def zscore(series: pd.Series) -> pd.Series:
            std = series.std(ddof=0)
            if pd.isna(std) or std == 0:
                return pd.Series(np.zeros(len(series)), index=series.index)
            return (series - series.mean()) / std

        base.loc[idx, "macro_stress_index"] = (
            zscore(chunk["inflation_rate"]).fillna(0)
            + zscore(chunk["unemployment_rate"]).fillna(0)
            - zscore(np.log1p(chunk["gdp_pc"])).fillna(0)
        )

    current_cols = [
        "gdp_pc",
        "unemployment_rate",
        "inflation_rate",
        "gdp_pc_growth",
        "unemployment_change",
        "inflation_change",
        "high_inflation",
        "high_unemployment",
        "low_income",
        "macro_stress_index",
    ]

    macro_current = base.copy()
    for col in current_cols:
        macro_current[f"{col}_missing"] = macro_current[col].isna().astype(int)

    macro_lag = base[["COUNTRY", "year"] + current_cols].copy()
    macro_lag["year"] = macro_lag["year"] + 2
    rename = {col: f"{col}_lag2" if col != "year" else col for col in macro_lag.columns}
    rename["COUNTRY"] = "COUNTRY"
    rename["year"] = "year"
    rename["gdp_pc"] = "gdp_pc_lag2"
    rename["unemployment_rate"] = "unemployment_lag2"
    rename["inflation_rate"] = "inflation_lag2"
    rename["gdp_pc_growth"] = "gdp_pc_growth_lag2"
    rename["unemployment_change"] = "unemployment_change_lag2"
    rename["inflation_change"] = "inflation_change_lag2"
    rename["high_inflation"] = "high_inflation_lag2"
    rename["high_unemployment"] = "high_unemployment_lag2"
    rename["low_income"] = "low_income_lag2"
    rename["macro_stress_index"] = "macro_stress_index_lag2"
    macro_lag = macro_lag.rename(columns=rename)
    for col in [
        "gdp_pc_lag2",
        "unemployment_lag2",
        "inflation_lag2",
        "gdp_pc_growth_lag2",
        "unemployment_change_lag2",
        "inflation_change_lag2",
        "high_inflation_lag2",
        "high_unemployment_lag2",
        "low_income_lag2",
        "macro_stress_index_lag2",
    ]:
        macro_lag[f"{col}_missing"] = macro_lag[col].isna().astype(int)

    return macro_current, macro_lag


# ---------------------------------------------------------------------------
# Panel país-semana
# ---------------------------------------------------------------------------

def build_panel(
    europe_df: pd.DataFrame,
    subtype_labels: list[str],
    macro_current: pd.DataFrame,
    macro_lag: pd.DataFrame,
):
    weeks = pd.date_range(europe_df["WEEK"].min(), europe_df["WEEK"].max(), freq=WEEK_FREQ)

    weekly_base = (
        europe_df.groupby(["COUNTRY", "WEEK"])
        .agg(
            total_events=("EVENTS", "sum"),
            total_fatalities=("FATALITIES", "sum"),
            total_violent_events=(
                "EVENTS",
                lambda s: s[
                    europe_df.loc[s.index, "EVENT_TYPE"].isin(VIOLENT_EVENT_TYPES)
                ].sum(),
            ),
            disruptive_events=(
                "EVENTS",
                lambda s: s[europe_df.loc[s.index, "is_disruptive_target"] == 1].sum(),
            ),
        )
        .reset_index()
    )

    type_pivot = (
        europe_df.pivot_table(
            index=["COUNTRY", "WEEK"],
            columns="EVENT_TYPE",
            values="EVENTS",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    subtype_pivot = (
        europe_df[europe_df["SUB_EVENT_TYPE"].isin(subtype_labels)]
        .pivot_table(
            index=["COUNTRY", "WEEK"],
            columns="SUB_EVENT_TYPE",
            values="EVENTS",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    panel = (
        weekly_base.merge(type_pivot, on=["COUNTRY", "WEEK"], how="left")
        .merge(subtype_pivot, on=["COUNTRY", "WEEK"], how="left")
        .fillna(0)
    )

    frames: list[pd.DataFrame] = []
    value_cols = [c for c in panel.columns if c not in {"COUNTRY", "WEEK"}]
    for country, group in panel.groupby("COUNTRY"):
        merged = pd.DataFrame({"WEEK": weeks}).merge(group, on="WEEK", how="left")
        merged["COUNTRY"] = country
        merged[value_cols] = merged[value_cols].fillna(0)
        frames.append(merged)

    panel = pd.concat(frames, ignore_index=True).sort_values(["COUNTRY", "WEEK"]).reset_index(drop=True)
    panel["year"] = panel["WEEK"].dt.year
    panel["quarter"] = panel["WEEK"].dt.quarter
    panel["iso_week"] = panel["WEEK"].dt.isocalendar().week.astype(int)
    panel["week_sin"] = np.sin(2 * np.pi * panel["iso_week"] / 52)
    panel["week_cos"] = np.cos(2 * np.pi * panel["iso_week"] / 52)

    base_signal_cols: list[str] = [
        "total_events",
        "total_fatalities",
        "total_violent_events",
        "disruptive_events",
    ]
    base_signal_cols += [c for c in EVENT_TYPE_ORDER if c in panel.columns]
    base_signal_cols += [c for c in subtype_labels if c in panel.columns]

    for col in base_signal_cols:
        group = panel.groupby("COUNTRY")[col]
        for lag in (1, 2, 4, 12):
            panel[f"{col}_lag{lag}"] = group.shift(lag)
        panel[f"{col}_ma4"] = group.transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean()
        )
        panel[f"{col}_ma12"] = group.transform(
            lambda s: s.shift(1).rolling(12, min_periods=1).mean()
        )
        panel[f"{col}_sum4"] = group.transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).sum()
        )
        panel[f"{col}_sum12"] = group.transform(
            lambda s: s.shift(1).rolling(12, min_periods=1).sum()
        )
        panel[f"{col}_diff1"] = group.transform(lambda s: s.shift(1).diff(1))
        panel[f"{col}_growth1"] = group.transform(
            lambda s: (s.shift(1) / (s.shift(2) + 1)).replace([np.inf, -np.inf], np.nan)
        )

    panel["violent_share_lag1"] = panel["total_violent_events_lag1"] / (
        panel["total_events_lag1"] + 1
    )
    panel["fatality_rate_lag1"] = panel["total_fatalities_lag1"] / (
        panel["total_events_lag1"] + 1
    )
    panel["recent_active_weeks_4"] = panel.groupby("COUNTRY")["total_events"].transform(
        lambda s: (s.shift(1) > 0).rolling(4, min_periods=1).sum()
    )
    panel["recent_active_weeks_12"] = panel.groupby("COUNTRY")["total_events"].transform(
        lambda s: (s.shift(1) > 0).rolling(12, min_periods=1).sum()
    )
    panel["recent_violent_weeks_4"] = panel.groupby("COUNTRY")["total_violent_events"].transform(
        lambda s: (s.shift(1) > 0).rolling(4, min_periods=1).sum()
    )
    panel["recent_violent_weeks_12"] = panel.groupby("COUNTRY")["total_violent_events"].transform(
        lambda s: (s.shift(1) > 0).rolling(12, min_periods=1).sum()
    )
    panel["escalation_ratio_4_vs_12"] = panel["disruptive_events_sum4"] / (
        panel["disruptive_events_sum12"] + 1
    )
    panel["persistence_ratio_violent"] = panel["recent_violent_weeks_4"] / (
        panel["recent_active_weeks_4"] + 1
    )
    panel["recurrence_index"] = panel["recent_active_weeks_12"] / 12
    panel["recent_violence_growth"] = panel["total_violent_events_sum4"] / (
        panel["total_violent_events_sum12"] + 1
    )
    panel["escalation_flag"] = (
        panel["total_violent_events_sum4"] > panel["total_violent_events_ma12"] * 4
    ).astype(int)

    panel = panel.merge(macro_current, on=["COUNTRY", "year"], how="left")
    panel = panel.merge(macro_lag, on=["COUNTRY", "year"], how="left")

    panel["inflation_x_protest_lag"] = panel["inflation_lag2"] * panel.get(
        "Protests_lag1", pd.Series(0, index=panel.index)
    )
    panel["unemployment_x_recent_violence"] = (
        panel["unemployment_lag2"] * panel["total_violent_events_sum4"]
    )
    panel["low_gdp_x_high_inflation"] = (
        panel["low_income_lag2"] * panel["high_inflation_lag2"]
    )

    def shift_target(s: pd.Series) -> pd.Series:
        nxt = s.shift(-1)
        return (nxt > 0).astype(float).where(nxt.notna(), np.nan)

    panel["y_next_disruptive_any"] = panel.groupby("COUNTRY")["disruptive_events"].transform(shift_target)
    panel["y_next_any_violent"] = panel.groupby("COUNTRY")["total_violent_events"].transform(shift_target)
    panel["y_next_any_event"] = panel.groupby("COUNTRY")["total_events"].transform(shift_target)

    type_target_cols: list[str] = []
    for event_type in EVENT_TYPE_ORDER:
        if event_type in panel.columns:
            col = f"y_next_type__{event_type}"
            panel[col] = panel.groupby("COUNTRY")[event_type].transform(shift_target)
            type_target_cols.append(col)

    subtype_target_cols: list[str] = []
    for subtype in subtype_labels:
        if subtype in panel.columns:
            col = f"y_next_subtype__{subtype}"
            panel[col] = panel.groupby("COUNTRY")[subtype].transform(shift_target)
            subtype_target_cols.append(col)

    panel = panel.replace([np.inf, -np.inf], np.nan)

    required_features = [
        c for c in panel.columns if c.endswith("lag12") or c.endswith("ma12") or c.endswith("sum12")
    ]
    required_targets = (
        ["y_next_disruptive_any", "y_next_any_violent", "y_next_any_event"]
        + type_target_cols
        + subtype_target_cols
    )
    model_panel = panel.dropna(subset=required_features).dropna(subset=required_targets).copy()

    return panel, model_panel, type_target_cols, subtype_target_cols


# ---------------------------------------------------------------------------
# Construcción de los conjuntos de features
# ---------------------------------------------------------------------------

def build_feature_sets(panel: pd.DataFrame, subtype_labels: list[str]) -> dict[str, list[str]]:
    feature_candidates = [
        c for c in panel.columns
        if c not in {
            "WEEK",
            "iso_week",
            "y_next_any_event",
            "y_next_any_violent",
            "y_next_disruptive_any",
        }
        and not c.startswith("y_next_type__")
        and not c.startswith("y_next_subtype__")
    ]

    fixed = {
        "COUNTRY",
        "year",
        "quarter",
        "week_sin",
        "week_cos",
        "recent_active_weeks_4",
        "recent_active_weeks_12",
        "recent_violent_weeks_4",
        "recent_violent_weeks_12",
        "violent_share_lag1",
        "fatality_rate_lag1",
        "escalation_ratio_4_vs_12",
        "persistence_ratio_violent",
        "recurrence_index",
        "recent_violence_growth",
        "escalation_flag",
    }
    prefixes = (
        "total_",
        "disruptive_",
        "Protests",
        "Riots",
        "Battles",
        "Violence against civilians",
        "Explosions/Remote violence",
    )

    conflict_only = [
        c
        for c in feature_candidates
        if c.startswith(prefixes)
        or c in fixed
        or c in subtype_labels
        or any(c.startswith(f"{s}_") for s in subtype_labels)
    ]

    macro_lagged = conflict_only + [
        c
        for c in panel.columns
        if c.startswith(
            (
                "gdp_pc_lag2",
                "unemployment_lag2",
                "inflation_lag2",
                "gdp_pc_growth_lag2",
                "unemployment_change_lag2",
                "inflation_change_lag2",
                "high_inflation_lag2",
                "high_unemployment_lag2",
                "low_income_lag2",
                "macro_stress_index_lag2",
            )
        )
    ]

    macro_interactions = macro_lagged + [
        "inflation_x_protest_lag",
        "unemployment_x_recent_violence",
        "low_gdp_x_high_inflation",
    ]

    return {
        "conflict_only": conflict_only,
        "macro_lagged": macro_lagged,
        "macro_interactions": macro_interactions,
    }


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def build_preprocessor(
    feature_df: pd.DataFrame, categorical_cols: list[str], dense: bool
) -> ColumnTransformer:
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
            ),
        ]
    )
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if not dense:
        numeric_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(numeric_steps)
    return ColumnTransformer(
        [("categorical", cat_pipe, categorical_cols), ("numeric", num_pipe, numeric_cols)],
        sparse_threshold=0 if dense else 0.3,
    )


def make_binary_pipeline(
    train_df: pd.DataFrame, feature_cols: list[str], spec: dict[str, Any]
) -> Pipeline:
    cat_cols = [c for c in ("COUNTRY", "quarter") if c in feature_cols]
    preprocessor = build_preprocessor(train_df[feature_cols], cat_cols, dense=spec["dense"])
    return Pipeline([("prep", preprocessor), ("model", clone(spec["estimator"]))])


def make_multioutput_pipeline(
    train_df: pd.DataFrame, feature_cols: list[str], spec: dict[str, Any]
) -> Pipeline:
    cat_cols = [c for c in ("COUNTRY", "quarter") if c in feature_cols]
    preprocessor = build_preprocessor(train_df[feature_cols], cat_cols, dense=spec["dense"])
    estimator = MultiOutputClassifier(clone(spec["estimator"]), n_jobs=1)
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    out = {
        "rocAuc": None,
        "prAuc": None,
        "f1": safe_metric(f1_score(y_true, pred, zero_division=0)),
        "precision": safe_metric(precision_score(y_true, pred, zero_division=0)),
        "recall": safe_metric(recall_score(y_true, pred, zero_division=0)),
        "brier": safe_metric(brier_score_loss(y_true, prob)),
        "logLoss": None,
        "positiveRate": safe_metric(float(np.mean(y_true))),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) > 1:
        out["rocAuc"] = safe_metric(roc_auc_score(y_true, prob))
        out["prAuc"] = safe_metric(average_precision_score(y_true, prob))
        out["logLoss"] = safe_metric(log_loss(y_true, np.clip(prob, 1e-9, 1 - 1e-9)))
    return out


# ---------------------------------------------------------------------------
# Catálogos de modelos (mejor representante de cada familia)
# ---------------------------------------------------------------------------

PROBLEM_A_MODELS: list[dict[str, Any]] = [
    {
        "id": "linear",
        "name": "LogisticRegression (L2)",
        "family": "Lineal",
        "description": "Regresión logística L2 con balanceo de clases.",
        "feature_set": "macro_lagged",
        "dense": False,
        "estimator": LogisticRegression(
            max_iter=2500, class_weight="balanced", C=1.0, random_state=RANDOM_STATE
        ),
    },
    {
        "id": "bagging",
        "name": "ExtraTrees",
        "family": "Bagging",
        "description": "Bosque aleatorio extra con balanceo y 260 árboles.",
        "feature_set": "conflict_only",
        "dense": True,
        "estimator": ExtraTreesClassifier(
            n_estimators=260,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=1,
        ),
    },
    {
        "id": "boosting",
        "name": "HistGradientBoosting",
        "family": "Boosting",
        "description": "Boosting tabular con histograma; ganador del problema A.",
        "feature_set": "conflict_only",
        "dense": True,
        "estimator": HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=260,
            random_state=RANDOM_STATE,
        ),
    },
]

PROBLEM_B_MODELS: list[dict[str, Any]] = [
    {
        "id": "linear",
        "name": "LogisticRegression",
        "family": "Lineal",
        "description": "One-vs-rest sobre los seis tipos principales.",
        "feature_set": "macro_lagged",
        "dense": False,
        "estimator": LogisticRegression(
            max_iter=1800, class_weight="balanced", random_state=RANDOM_STATE
        ),
    },
    {
        "id": "bagging",
        "name": "RandomForest",
        "family": "Bagging",
        "description": "Random forest multi-output con balanceo subsample.",
        "feature_set": "conflict_only",
        "dense": True,
        "estimator": RandomForestClassifier(
            n_estimators=140,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    },
    {
        "id": "boosting",
        "name": "HistGradientBoosting",
        "family": "Boosting",
        "description": "Boosting tabular; ganador del problema B.",
        "feature_set": "macro_lagged",
        "dense": True,
        "estimator": HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.08,
            max_iter=140,
            random_state=RANDOM_STATE,
        ),
    },
]

PROBLEM_C_MODELS: list[dict[str, Any]] = [
    {
        "id": "linear",
        "name": "LogisticRegression",
        "family": "Lineal",
        "description": "One-vs-rest sobre los subtipos modelables.",
        "feature_set": "macro_lagged",
        "dense": False,
        "estimator": LogisticRegression(
            max_iter=1800, class_weight="balanced", random_state=RANDOM_STATE
        ),
    },
    {
        "id": "bagging",
        "name": "RandomForest",
        "family": "Bagging",
        "description": "Random forest multi-output sobre subtipos.",
        "feature_set": "conflict_only",
        "dense": True,
        "estimator": RandomForestClassifier(
            n_estimators=140,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    },
    {
        "id": "boosting",
        "name": "HistGradientBoosting",
        "family": "Boosting",
        "description": "Boosting tabular con interacciones macro; ganador del problema C.",
        "feature_set": "macro_interactions",
        "dense": True,
        "estimator": HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.08,
            max_iter=120,
            random_state=RANDOM_STATE,
        ),
    },
]


# ---------------------------------------------------------------------------
# Centroides
# ---------------------------------------------------------------------------

def compute_centroids(europe_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    cutoff = europe_df["WEEK"].max() - pd.Timedelta(weeks=52)
    source = europe_df[europe_df["WEEK"] >= cutoff]
    if source.empty:
        source = europe_df
    centroids: dict[str, dict[str, float]] = {}
    for country, group in source.groupby("COUNTRY"):
        weights = group["EVENTS"].fillna(0).clip(lower=0) + 1
        if "CENTROID_LATITUDE" not in group.columns or "CENTROID_LONGITUDE" not in group.columns:
            continue
        lat = float(np.average(group["CENTROID_LATITUDE"], weights=weights))
        lon = float(np.average(group["CENTROID_LONGITUDE"], weights=weights))
        centroids[country] = {"lat": to_float(lat, decimals=4), "lon": to_float(lon, decimals=4)}
    return centroids


# ---------------------------------------------------------------------------
# Entrenamiento + predicción por problema
# ---------------------------------------------------------------------------

def train_problem_a(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    models_dir: Path,
):
    target = "y_next_disruptive_any"
    results: list[dict[str, Any]] = []
    for spec in PROBLEM_A_MODELS:
        feature_cols = feature_sets[spec["feature_set"]]
        pipeline = make_binary_pipeline(train_df, feature_cols, spec)
        pipeline.fit(train_df[feature_cols], train_df[target].astype(int))

        test_prob = pipeline.predict_proba(test_df[feature_cols])[:, 1]
        metrics = binary_metrics(test_df[target].astype(int).values, test_prob)

        infer_prob = pipeline.predict_proba(inference_df[feature_cols])[:, 1]
        country_preds = {
            row["COUNTRY"]: to_float(prob)
            for row, prob in zip(inference_df.to_dict("records"), infer_prob)
        }

        joblib.dump(pipeline, models_dir / f"problemA_{spec['id']}.joblib")

        results.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "family": spec["family"],
                "description": spec["description"],
                "featureSet": spec["feature_set"],
                "metrics": metrics,
                "countryRisk": country_preds,
            }
        )
    return results


def train_problem_multi(
    problem_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target_cols: list[str],
    target_label_prefix: str,
    model_specs: list[dict[str, Any]],
    models_dir: Path,
):
    results: list[dict[str, Any]] = []
    labels = [c.replace(target_label_prefix, "") for c in target_cols]

    for spec in model_specs:
        feature_cols = feature_sets[spec["feature_set"]]
        pipeline = make_multioutput_pipeline(train_df, feature_cols, spec)
        pipeline.fit(train_df[feature_cols], train_df[target_cols].astype(int))

        test_prob_list = pipeline.predict_proba(test_df[feature_cols])
        infer_prob_list = pipeline.predict_proba(inference_df[feature_cols])

        per_label_metrics: list[dict[str, Any]] = []
        macro_pr, macro_roc, macro_f1, macro_brier = [], [], [], []
        for idx, target in enumerate(target_cols):
            label = labels[idx]
            y_true = test_df[target].astype(int).values
            probs = test_prob_list[idx][:, 1]
            metrics = binary_metrics(y_true, probs)
            metrics["label"] = label
            per_label_metrics.append(metrics)
            if metrics["prAuc"] is not None:
                macro_pr.append(metrics["prAuc"])
            if metrics["rocAuc"] is not None:
                macro_roc.append(metrics["rocAuc"])
            if metrics["f1"] is not None:
                macro_f1.append(metrics["f1"])
            if metrics["brier"] is not None:
                macro_brier.append(metrics["brier"])

        country_preds: dict[str, dict[str, float]] = {}
        records = inference_df.to_dict("records")
        for row_idx, row in enumerate(records):
            per_label = {}
            for label_idx, label in enumerate(labels):
                per_label[label] = to_float(infer_prob_list[label_idx][row_idx, 1])
            country_preds[row["COUNTRY"]] = per_label

        joblib.dump(pipeline, models_dir / f"problem{problem_id}_{spec['id']}.joblib")

        results.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "family": spec["family"],
                "description": spec["description"],
                "featureSet": spec["feature_set"],
                "metrics": {
                    "macroRocAuc": safe_metric(np.mean(macro_roc)) if macro_roc else None,
                    "macroPrAuc": safe_metric(np.mean(macro_pr)) if macro_pr else None,
                    "macroF1": safe_metric(np.mean(macro_f1)) if macro_f1 else None,
                    "macroBrier": safe_metric(np.mean(macro_brier)) if macro_brier else None,
                },
                "perLabelMetrics": per_label_metrics,
                "countryByLabel": country_preds,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Construcción del payload final
# ---------------------------------------------------------------------------

def build_payload(
    europe_df: pd.DataFrame,
    inference_week: pd.Timestamp,
    cutoff: pd.Timestamp,
    horizon_days: int,
    centroids: dict[str, dict[str, float]],
    inference_df: pd.DataFrame,
    problem_a: list[dict[str, Any]],
    problem_b: list[dict[str, Any]],
    type_labels: list[str],
    problem_c: list[dict[str, Any]],
    subtype_labels: list[str],
    forecast_start_override: pd.Timestamp | None = None,
    forecast_end_override: pd.Timestamp | None = None,
) -> dict[str, Any]:
    if forecast_start_override is not None:
        forecast_start = forecast_start_override.date()
    else:
        forecast_start = (inference_week + pd.Timedelta(days=1)).date()
    if forecast_end_override is not None:
        forecast_end = forecast_end_override.date()
    else:
        forecast_end = (inference_week + pd.Timedelta(days=horizon_days)).date()

    countries: dict[str, dict[str, Any]] = {}
    for row in inference_df.to_dict("records"):
        country = row["COUNTRY"]
        centroid = centroids.get(country)
        if centroid is None:
            continue
        countries[country] = {
            "centroid": centroid,
            "stats": {
                "events": to_float(row["total_events"]),
                "violentEvents": to_float(row["total_violent_events"]),
                "eventsMA4": to_float(row["total_events_ma4"]),
                "violentMA4": to_float(row["total_violent_events_ma4"]),
                "disruptiveSum4": to_float(row["disruptive_events_sum4"]),
                "fatalitiesLag1": to_float(row["total_fatalities_lag1"]),
            },
        }

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "region": "Europe",
            "dateRange": {
                "start": europe_df["WEEK"].min().date().isoformat(),
                "end": europe_df["WEEK"].max().date().isoformat(),
            },
            "forecastWindow": {
                "start": forecast_start.isoformat(),
                "end": forecast_end.isoformat(),
            },
            "cutoff": cutoff.date().isoformat(),
            "violentEventTypes": sorted(VIOLENT_EVENT_TYPES),
        },
        "countries": countries,
        "problems": [
            {
                "id": "A",
                "label": "Riesgo agregado de evento disruptivo",
                "shortLabel": "Riesgo país",
                "type": "binary",
                "description": "Probabilidad de que la próxima semana ocurra al menos un evento disruptivo (violento o protesta no pacífica).",
                "models": problem_a,
            },
            {
                "id": "B",
                "label": "Probabilidad por tipo principal",
                "shortLabel": "Por tipo",
                "type": "multi-type",
                "description": "Una probabilidad independiente por cada tipo principal en t+1.",
                "labels": type_labels,
                "models": problem_b,
            },
            {
                "id": "C",
                "label": "Probabilidad por subtipo",
                "shortLabel": "Por subtipo",
                "type": "multi-subtype",
                "description": "Probabilidades por subtipo modelable seleccionado.",
                "labels": subtype_labels,
                "models": problem_c,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


_ACLED_DATE_RE = re.compile(
    r"Europe-Central-Asia_aggregated_data_up_to(?:_week_of)?-(\d{4}-\d{2}-\d{2})\.xlsx$"
)


def find_latest_acled(root: Path) -> Path:
    """Localiza el XLSX agregado de ACLED con la fecha más reciente en su nombre.

    Acepta ambas variantes del patrón:
      Europe-Central-Asia_aggregated_data_up_to-YYYY-MM-DD.xlsx
      Europe-Central-Asia_aggregated_data_up_to_week_of-YYYY-MM-DD.xlsx
    """
    candidates: list[tuple[date, Path]] = []
    for path in root.glob("Europe-Central-Asia_aggregated_data_up_to*.xlsx"):
        match = _ACLED_DATE_RE.search(path.name)
        if not match:
            continue
        try:
            file_date = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        candidates.append((file_date, path))

    if not candidates:
        raise SystemExit(
            f"No encuentro Europe-Central-Asia_aggregated_data_up_to*.xlsx en {root}. "
            "Coloca el XLSX descargado de ACLED ahí o pasa --conflict explícitamente."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def next_iso_week_window(today: date) -> tuple[date, date]:
    """(lunes próximo, domingo próximo) desde hoy. Si hoy es lunes, salta al siguiente."""
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    monday = today + timedelta(days=days_until_monday)
    sunday = monday + timedelta(days=6)
    return monday, sunday


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--conflict",
        type=Path,
        default=None,
        help="XLSX agregado de ACLED. Si se omite, autodetecta el más reciente en la raíz.",
    )
    parser.add_argument(
        "--gdp",
        type=Path,
        default=REPO_ROOT.parent / "sdg_08_10_page_spreadsheet.xlsx",
        help="XLSX Eurostat PIB per cápita",
    )
    parser.add_argument(
        "--unemployment",
        type=Path,
        default=REPO_ROOT.parent / "tps00203_page_spreadsheet.xlsx",
        help="XLSX Eurostat desempleo",
    )
    parser.add_argument(
        "--inflation",
        type=Path,
        default=REPO_ROOT.parent / "prc_hicp_aind$defaultview_spreadsheet.xlsx",
        help="XLSX Eurostat inflación HICP",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "src/lib/data/predictions.json",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "output/models",
    )
    parser.add_argument("--cutoff", type=str, default="2025-01-04", help="Corte cronológico train/test")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument(
        "--forecast-window",
        choices=["next-iso", "after-data"],
        default="next-iso",
        help=(
            "next-iso: próxima semana ISO (lunes-domingo) desde hoy. "
            "after-data: semana siguiente al último dato disponible (modo antiguo)."
        ),
    )
    parser.add_argument(
        "--forecast-start",
        type=str,
        default=None,
        help="Fuerza el inicio de la ventana (YYYY-MM-DD); sobreescribe --forecast-window",
    )
    parser.add_argument(
        "--forecast-end",
        type=str,
        default=None,
        help="Fuerza el fin de la ventana (YYYY-MM-DD); sobreescribe --forecast-window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoff = pd.Timestamp(args.cutoff)

    conflict_path = args.conflict or find_latest_acled(REPO_ROOT)
    print(f"[1/6] Cargando dataset de conflictos · {conflict_path.name}")
    conflict_raw = load_conflict(conflict_path)
    europe_df = filter_europe(conflict_raw)

    print("[2/6] Cargando macroeconomía…")
    macro = load_macro(args.gdp, args.unemployment, args.inflation)
    macro_current, macro_lag = build_macro_master(macro)

    print("[3/6] Construyendo panel país-semana y targets…")
    subtype_labels = select_modelable_subtypes(europe_df)
    full_panel, model_panel, type_target_cols, subtype_target_cols = build_panel(
        europe_df, subtype_labels, macro_current, macro_lag
    )
    feature_sets = build_feature_sets(model_panel, subtype_labels)

    train_df = model_panel[model_panel["WEEK"] < cutoff].copy()
    test_df = model_panel[model_panel["WEEK"] >= cutoff].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("El cutoff deja vacío train o test; ajusta --cutoff")

    inference_week = full_panel["WEEK"].max()
    inference_df = full_panel[full_panel["WEEK"] == inference_week].copy()

    args.models_dir.mkdir(parents=True, exist_ok=True)

    print("[4/6] Entrenando problema A (binario)…")
    problem_a = train_problem_a(train_df, test_df, inference_df, feature_sets, args.models_dir)

    print("[5/6] Entrenando problemas B y C (multietiqueta)…")
    type_labels = [c.replace("y_next_type__", "") for c in type_target_cols]
    problem_b = train_problem_multi(
        "B",
        train_df,
        test_df,
        inference_df,
        feature_sets,
        type_target_cols,
        "y_next_type__",
        PROBLEM_B_MODELS,
        args.models_dir,
    )
    subtype_label_clean = [c.replace("y_next_subtype__", "") for c in subtype_target_cols]
    problem_c = train_problem_multi(
        "C",
        train_df,
        test_df,
        inference_df,
        feature_sets,
        subtype_target_cols,
        "y_next_subtype__",
        PROBLEM_C_MODELS,
        args.models_dir,
    )

    print("[6/6] Calculando centroides y exportando JSON…")
    centroids = compute_centroids(europe_df)

    if args.forecast_start:
        forecast_start_override = pd.Timestamp(args.forecast_start)
    elif args.forecast_window == "next-iso":
        ws, _ = next_iso_week_window(date.today())
        forecast_start_override = pd.Timestamp(ws)
    else:
        forecast_start_override = None

    if args.forecast_end:
        forecast_end_override = pd.Timestamp(args.forecast_end)
    elif args.forecast_window == "next-iso":
        _, we = next_iso_week_window(date.today())
        forecast_end_override = pd.Timestamp(we)
    else:
        forecast_end_override = None

    payload = build_payload(
        europe_df,
        inference_week,
        cutoff,
        args.horizon_days,
        centroids,
        inference_df,
        problem_a,
        problem_b,
        type_labels,
        problem_c,
        subtype_label_clean,
        forecast_start_override=forecast_start_override,
        forecast_end_override=forecast_end_override,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"OK. JSON: {args.output}")
    print(f"OK. Modelos: {args.models_dir}")
    print(
        "Ventana de pronóstico:",
        payload["dataset"]["forecastWindow"]["start"],
        "→",
        payload["dataset"]["forecastWindow"]["end"],
    )
    print(f"Países: {len(payload['countries'])} · Problemas: {len(payload['problems'])}")


if __name__ == "__main__":
    main()
