export type ProblemId = 'A' | 'B' | 'C';
export type ProblemType = 'binary' | 'multi-type' | 'multi-subtype';
export type ModelId = 'linear' | 'bagging' | 'boosting';
export type ModelFamily = 'Lineal' | 'Bagging' | 'Boosting';

export interface BinaryMetrics {
  rocAuc: number | null;
  prAuc: number | null;
  f1: number | null;
  precision: number | null;
  recall: number | null;
  brier: number | null;
  logLoss: number | null;
  positiveRate: number | null;
  threshold: number;
}

export interface PerLabelMetrics extends BinaryMetrics {
  label: string;
}

export interface MacroMetrics {
  macroRocAuc: number | null;
  macroPrAuc: number | null;
  macroF1: number | null;
  macroBrier: number | null;
}

export interface BinaryModel {
  id: ModelId;
  name: string;
  family: ModelFamily;
  description: string;
  featureSet: string;
  metrics: BinaryMetrics;
  countryRisk: Record<string, number>;
}

export interface MultiOutputModel {
  id: ModelId;
  name: string;
  family: ModelFamily;
  description: string;
  featureSet: string;
  metrics: MacroMetrics;
  perLabelMetrics: PerLabelMetrics[];
  countryByLabel: Record<string, Record<string, number>>;
}

export interface BinaryProblem {
  id: 'A';
  label: string;
  shortLabel: string;
  type: 'binary';
  description: string;
  models: BinaryModel[];
}

export interface MultiOutputProblem {
  id: 'B' | 'C';
  label: string;
  shortLabel: string;
  type: 'multi-type' | 'multi-subtype';
  description: string;
  labels: string[];
  models: MultiOutputModel[];
}

export type Problem = BinaryProblem | MultiOutputProblem;

export interface CountryMeta {
  centroid: { lat: number; lon: number };
  stats: {
    events: number;
    violentEvents: number;
    eventsMA4: number;
    violentMA4: number;
    disruptiveSum4: number;
    fatalitiesLag1: number;
  };
}

export interface PredictionData {
  generatedAt: string;
  dataset: {
    region: string;
    dateRange: { start: string; end: string };
    forecastWindow: { start: string; end: string };
    cutoff: string;
    violentEventTypes: string[];
  };
  countries: Record<string, CountryMeta>;
  problems: Problem[];
}
