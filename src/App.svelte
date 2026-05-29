<script lang="ts">
  import { onMount, tick } from 'svelte';
  import L, { type LayerGroup, type Map as LeafletMap } from 'leaflet';
  import 'leaflet/dist/leaflet.css';
  import { geoGraticule10 } from 'd3-geo';
  import { feature as topoFeature } from 'topojson-client';
  import countriesTopology from 'world-atlas/countries-110m.json';
  import landTopology from 'world-atlas/land-110m.json';

  import GlobeIntro from './lib/components/GlobeIntro.svelte';
  import rawData from './lib/data/predictions.json';
  import type {
    BinaryModel,
    BinaryProblem,
    MultiOutputModel,
    MultiOutputProblem,
    PredictionData,
    Problem,
    ModelId,
  } from './lib/types';

  const data = rawData as unknown as PredictionData;

  type MapStage = 'hidden' | 'flying' | 'ready';

  const ENTRY_VIEW = { center: [22, 0] as [number, number], zoom: 2.1 };
  const EUROPE_VIEW = { center: [52, 14] as [number, number], zoom: 4.4 };

  let selectedProblemId: 'A' | 'B' | 'C' = 'A';
  let selectedModelId: ModelId = 'boosting';
  let selectedLabel: string = '';
  let countrySearch = '';
  let minProbability = 0.0;

  $: selectedProblem = data.problems.find((p) => p.id === selectedProblemId) as Problem;
  $: selectedModel = selectedProblem.models.find((m) => m.id === selectedModelId) ?? selectedProblem.models[0];

  $: isBinary = selectedProblem.type === 'binary';
  $: availableLabels = isBinary ? [] : (selectedProblem as MultiOutputProblem).labels;

  // Mantener selectedLabel coherente con el problema activo.
  $: if (!isBinary && (!selectedLabel || !availableLabels.includes(selectedLabel))) {
    selectedLabel = availableLabels[0] ?? '';
  }

  $: probabilityByCountry = computeCountryProbabilities(selectedProblem, selectedModel, selectedLabel);

  function computeCountryProbabilities(
    problem: Problem,
    model: BinaryModel | MultiOutputModel,
    label: string,
  ): Record<string, number> {
    if (problem.type === 'binary') {
      return (model as BinaryModel).countryRisk ?? {};
    }
    const byLabel = (model as MultiOutputModel).countryByLabel ?? {};
    const out: Record<string, number> = {};
    for (const [country, probs] of Object.entries(byLabel)) {
      out[country] = probs[label] ?? 0;
    }
    return out;
  }

  $: rankedCountries = Object.entries(probabilityByCountry)
    .filter(([country]) => Boolean(data.countries[country]))
    .map(([country, probability]) => ({ country, probability }))
    .filter(({ probability }) => probability >= minProbability)
    .filter(({ country }) =>
      countrySearch.trim().length === 0
        ? true
        : country.toLowerCase().includes(countrySearch.trim().toLowerCase()),
    )
    .sort((a, b) => b.probability - a.probability);

  function asPercent(value: number): string {
    return new Intl.NumberFormat('es-ES', {
      style: 'percent',
      maximumFractionDigits: 1,
    }).format(value);
  }

  function asDate(value: string): string {
    return new Date(value).toLocaleDateString('es-ES', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
    });
  }

  function asMetric(value: number | null): string {
    return value == null ? '—' : value.toFixed(3);
  }

  // ---- mapa ----
  let mapContainer: HTMLDivElement;
  let map: LeafletMap | null = null;
  let countryLayer: LayerGroup | null = null;
  let pointLayer: LayerGroup | null = null;
  const mapRenderer = L.canvas({ padding: 1.4 });

  let showIntro = true;
  let mapReady = false;
  let mapStage: MapStage = 'hidden';
  let entryRequested = false;
  let entryStarted = false;
  let entryFallbackTimer: ReturnType<typeof setTimeout> | null = null;

  const countriesGeoJson = (() => {
    const topology = countriesTopology as any;
    return topoFeature(topology, topology.objects.countries) as any;
  })();
  const landGeoJson = (() => {
    const topology = landTopology as any;
    return topoFeature(topology, topology.objects.land) as any;
  })();
  const graticuleGeoJson = geoGraticule10() as any;

  function setMapInteractions(enabled: boolean): void {
    if (!map) return;
    const action = enabled ? 'enable' : 'disable';
    map.dragging[action]();
    map.scrollWheelZoom[action]();
    map.doubleClickZoom[action]();
    map.boxZoom[action]();
    map.keyboard[action]();
    map.touchZoom[action]();
  }

  function finalizeEntry(): void {
    if (!map || mapStage === 'ready') return;
    if (entryFallbackTimer) {
      clearTimeout(entryFallbackTimer);
      entryFallbackTimer = null;
    }
    map.stop();
    map.setView(EUROPE_VIEW.center, EUROPE_VIEW.zoom, { animate: false });
    mapStage = 'ready';
    setMapInteractions(true);
  }

  function addBaseLayers(): void {
    if (!map) return;

    map.createPane('land-pane');
    map.createPane('graticule-pane');
    map.createPane('country-pane');
    map.createPane('event-pane');

    const setZ = (name: string, z: number) => {
      const pane = map?.getPane(name);
      if (pane) pane.style.zIndex = String(z);
    };
    setZ('land-pane', 330);
    setZ('graticule-pane', 340);
    setZ('country-pane', 360);
    setZ('event-pane', 420);

    L.geoJSON(landGeoJson, {
      pane: 'land-pane',
      interactive: false,
      style: { fillColor: '#e2e2e2', fillOpacity: 0.95, color: '#cdcdcd', opacity: 0.9, weight: 0.8 },
      renderer: mapRenderer,
    } as L.GeoJSONOptions).addTo(map);

    L.geoJSON(graticuleGeoJson, {
      pane: 'graticule-pane',
      interactive: false,
      style: { color: '#b8b8b8', opacity: 0.32, weight: 0.6 },
      renderer: mapRenderer,
    } as L.GeoJSONOptions).addTo(map);

    countryLayer = L.layerGroup().addTo(map);
    pointLayer = L.layerGroup().addTo(map);
  }

  function probabilityToColor(probability: number): string {
    const p = Math.max(0, Math.min(1, probability));
    if (p < 0.001) return '#e2e2e2';
    // Gradiente perceptual: amarillo cálido -> naranja -> rojo profundo.
    const stops: Array<[number, [number, number, number]]> = [
      [0.0, [232, 232, 232]],
      [0.2, [255, 224, 178]],
      [0.4, [255, 183, 110]],
      [0.6, [241, 124, 73]],
      [0.8, [200, 64, 56]],
      [1.0, [128, 24, 38]],
    ];
    let from = stops[0];
    let to = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i += 1) {
      if (p >= stops[i][0] && p <= stops[i + 1][0]) {
        from = stops[i];
        to = stops[i + 1];
        break;
      }
    }
    const t = (p - from[0]) / Math.max(to[0] - from[0], 1e-6);
    const lerp = (a: number, b: number) => Math.round(a + (b - a) * t);
    const [r, g, b] = [lerp(from[1][0], to[1][0]), lerp(from[1][1], to[1][1]), lerp(from[1][2], to[1][2])];
    return `rgb(${r}, ${g}, ${b})`;
  }

  // Mapeo nombre país (ACLED) -> ISO numeric (world-atlas).
  // Convertimos el geojson para usar la propiedad "name", que ya está alineada casi en su totalidad.
  function syncCountryLayer(probabilities: Record<string, number>): void {
    if (!map || !countryLayer) return;
    countryLayer.clearLayers();

    const options: any = {
      pane: 'country-pane',
      renderer: mapRenderer,
      style: (feature: any) => {
        const name = String(feature?.properties?.name ?? '');
        const probability = probabilities[name] ?? probabilities[normalizeName(name)] ?? null;
        if (probability == null) {
          return {
            fillColor: '#ededed',
            fillOpacity: 0.65,
            color: '#bdbdbd',
            opacity: 0.6,
            weight: 0.6,
          };
        }
        return {
          fillColor: probabilityToColor(probability),
          fillOpacity: 0.78,
          color: '#3a3a3a',
          opacity: 0.85,
          weight: 0.7,
        };
      },
      onEachFeature: (feature: any, layer: L.Layer) => {
        const name = String(feature?.properties?.name ?? '');
        const probability = probabilities[name] ?? probabilities[normalizeName(name)] ?? null;
        if (probability == null) return;
        const detail = isBinary
          ? `<p>Riesgo (modelo ${selectedModel.name}): <strong>${asPercent(probability)}</strong></p>`
          : `<p>${selectedLabel} → <strong>${asPercent(probability)}</strong></p>`;
        (layer as L.Path).bindPopup(
          `<p><strong>${name}</strong></p>${detail}<p class="muted">Familia ${selectedModel.family}</p>`,
          { autoPan: false },
        );
        (layer as L.Path).on('mouseover', () =>
          (layer as L.Path).setStyle({ weight: 1.6, color: '#111' }),
        );
        (layer as L.Path).on('mouseout', () =>
          (layer as L.Path).setStyle({ weight: 0.7, color: '#3a3a3a' }),
        );
      },
    };

    L.geoJSON(countriesGeoJson, options).addTo(countryLayer);
  }

  function normalizeName(name: string): string {
    const alias: Record<string, string> = {
      'Czech Republic': 'Czechia',
      'Czechia': 'Czech Republic',
      'North Macedonia': 'North Macedonia',
      'Republic of Serbia': 'Serbia',
    };
    return alias[name] ?? name;
  }

  function syncPointLayer(probabilities: Record<string, number>): void {
    if (!map || !pointLayer) return;
    pointLayer.clearLayers();

    for (const [country, probability] of Object.entries(probabilities)) {
      if (probability < Math.max(minProbability, 0.05)) continue;
      const meta = data.countries[country];
      if (!meta) continue;
      const marker = L.circleMarker([meta.centroid.lat, meta.centroid.lon], {
        pane: 'event-pane',
        renderer: mapRenderer,
        radius: 4 + probability * 14,
        color: '#1a1a1a',
        weight: 0.9,
        fillColor: probabilityToColor(probability),
        fillOpacity: 0.92,
      });
      const subtitle = isBinary
        ? `<p>Riesgo evento disruptivo</p>`
        : `<p>${selectedLabel}</p>`;
      marker.bindPopup(
        `<p><strong>${country}</strong></p>${subtitle}<p><strong>${asPercent(probability)}</strong></p>
         <p class="muted">Eventos rec. (MA4): ${meta.stats.eventsMA4.toFixed(1)}</p>`,
        { autoPan: false },
      );
      marker.addTo(pointLayer);
    }
  }

  $: if (mapReady && mapStage !== 'hidden') {
    syncCountryLayer(probabilityByCountry);
    syncPointLayer(probabilityByCountry);
  }

  function runEntryFlight(): void {
    if (!map || entryStarted || !entryRequested) return;
    entryRequested = false;
    entryStarted = true;
    map.stop();
    setMapInteractions(false);
    mapStage = 'flying';
    map.invalidateSize({ animate: false });

    requestAnimationFrame(() => {
      if (!map) return;
      map.once('moveend', finalizeEntry);
      map.flyTo(EUROPE_VIEW.center, EUROPE_VIEW.zoom, {
        animate: true,
        duration: 2.4,
        easeLinearity: 0.22,
      });
      entryFallbackTimer = setTimeout(() => finalizeEntry(), 3200);
    });
  }

  function maybeStartEntry(): void {
    if (!mapReady || !entryRequested) return;
    runEntryFlight();
  }

  function onIntroEnter(): void {
    mapStage = 'hidden';
  }

  function onIntroComplete(): void {
    showIntro = false;
    mapStage = 'flying';
    entryRequested = true;
    maybeStartEntry();
  }

  async function focusCountry(country: string): Promise<void> {
    if (!map || mapStage !== 'ready') return;
    const meta = data.countries[country];
    if (!meta) return;
    await tick();
    map.flyTo([meta.centroid.lat, meta.centroid.lon], 5.6, {
      animate: true,
      duration: 1.4,
      easeLinearity: 0.25,
    });
  }

  function resetView(): void {
    if (!map || mapStage !== 'ready') return;
    map.flyTo(EUROPE_VIEW.center, EUROPE_VIEW.zoom, {
      animate: true,
      duration: 1.3,
      easeLinearity: 0.25,
    });
  }

  onMount(() => {
    map = L.map(mapContainer, {
      zoomControl: true,
      minZoom: 2,
      maxZoom: 7,
      zoomSnap: 0.1,
      zoomDelta: 0.25,
      preferCanvas: true,
      worldCopyJump: false,
      attributionControl: false,
      renderer: mapRenderer,
    });

    map.setView(ENTRY_VIEW.center, ENTRY_VIEW.zoom, { animate: false });
    setMapInteractions(false);
    addBaseLayers();
    mapReady = true;
    maybeStartEntry();

    return () => {
      if (entryFallbackTimer) clearTimeout(entryFallbackTimer);
      map?.remove();
      map = null;
    };
  });
</script>

<main class="app-root">
  <section class="page">
    <header class="hero" class:visible={mapStage !== 'hidden'} class:hidden={mapStage === 'hidden'}>
      <div class="hero-title">
        <p class="eyebrow">Forecast desk · {data.dataset.region}</p>
        <h1>Predicción semanal de conflictividad</h1>
        <p class="hero-sub">
          Ventana: <strong>{asDate(data.dataset.forecastWindow.start)}</strong> →
          <strong>{asDate(data.dataset.forecastWindow.end)}</strong> · cutoff
          {asDate(data.dataset.cutoff)}
        </p>
      </div>

      <div class="metrics">
        {#if isBinary}
          {@const m = (selectedModel as BinaryModel).metrics}
          <article>
            <span>ROC-AUC</span><strong>{asMetric(m.rocAuc)}</strong>
          </article>
          <article>
            <span>PR-AUC</span><strong>{asMetric(m.prAuc)}</strong>
          </article>
          <article>
            <span>F1</span><strong>{asMetric(m.f1)}</strong>
          </article>
          <article>
            <span>Brier</span><strong>{asMetric(m.brier)}</strong>
          </article>
        {:else}
          {@const m = (selectedModel as MultiOutputModel).metrics}
          <article>
            <span>macro ROC</span><strong>{asMetric(m.macroRocAuc)}</strong>
          </article>
          <article>
            <span>macro PR</span><strong>{asMetric(m.macroPrAuc)}</strong>
          </article>
          <article>
            <span>macro F1</span><strong>{asMetric(m.macroF1)}</strong>
          </article>
          <article>
            <span>macro Brier</span><strong>{asMetric(m.macroBrier)}</strong>
          </article>
        {/if}
      </div>
    </header>

    <section class="layout">
      <aside class="sidebar" class:visible={mapStage !== 'hidden'} class:hidden={mapStage === 'hidden'}>
        <div class="card">
          <div class="card-head">
            <h2>Problema</h2>
            <span class="hint">{selectedProblem.shortLabel}</span>
          </div>
          <div class="segmented">
            {#each data.problems as problem}
              <button
                type="button"
                class:active={problem.id === selectedProblemId}
                on:click={() => (selectedProblemId = problem.id)}
              >
                <span class="seg-tag">{problem.id}</span>
                <span class="seg-label">{problem.shortLabel}</span>
              </button>
            {/each}
          </div>
          <p class="card-desc">{selectedProblem.description}</p>
        </div>

        <div class="card">
          <div class="card-head">
            <h2>Modelo</h2>
            <span class="hint">{selectedModel.family}</span>
          </div>
          <div class="segmented vertical">
            {#each selectedProblem.models as model}
              <button
                type="button"
                class:active={model.id === selectedModelId}
                on:click={() => (selectedModelId = model.id)}
              >
                <span class="seg-bullet" data-family={model.family.toLowerCase()}></span>
                <span class="seg-content">
                  <span class="seg-name">{model.name}</span>
                  <span class="seg-meta">{model.family} · {model.featureSet.replace('_', ' ')}</span>
                </span>
              </button>
            {/each}
          </div>
          <p class="card-desc">{selectedModel.description}</p>
        </div>

        {#if !isBinary}
          <div class="card">
            <div class="card-head">
              <h2>{selectedProblem.type === 'multi-type' ? 'Tipo de evento' : 'Subtipo'}</h2>
              <span class="hint">{selectedLabel}</span>
            </div>
            <div class="chips">
              {#each availableLabels as label}
                <button
                  type="button"
                  class:selected={label === selectedLabel}
                  on:click={() => (selectedLabel = label)}
                >
                  {label}
                </button>
              {/each}
            </div>
          </div>
        {/if}

        <div class="card">
          <h2>Filtros</h2>
          <label class="field">
            <span>Buscar país</span>
            <input bind:value={countrySearch} placeholder="p. ej. Germany" type="text" />
          </label>
          <label class="field">
            <span>Probabilidad mínima: {asPercent(minProbability)}</span>
            <input bind:value={minProbability} max="0.95" min="0" step="0.01" type="range" />
          </label>
          <button class="ghost-btn" on:click={resetView} type="button">Reencuadrar Europa</button>
        </div>

        <div class="card">
          <h2>Top {Math.min(12, rankedCountries.length)} países</h2>
          <ul class="ranked">
            {#each rankedCountries.slice(0, 12) as item, index (item.country)}
              <li>
                <button on:click={() => focusCountry(item.country)} type="button">
                  <span class="rank">{String(index + 1).padStart(2, '0')}</span>
                  <span class="country-name">{item.country}</span>
                  <span class="country-prob">{asPercent(item.probability)}</span>
                </button>
              </li>
            {:else}
              <li class="empty">Sin países con probabilidad ≥ {asPercent(minProbability)}.</li>
            {/each}
          </ul>
        </div>

        <div class="legend card">
          <h2>Escala</h2>
          <div class="legend-bar">
            <div class="legend-gradient"></div>
            <div class="legend-ticks">
              <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
            </div>
          </div>
        </div>
      </aside>

      <div class="map-shell" class:visible={mapStage !== 'hidden'} class:hidden={mapStage === 'hidden'}>
        <div bind:this={mapContainer} class="map-canvas"></div>
        <div class="map-fade"></div>
        <div class="map-badge">
          <span class="badge-dot"></span>
          <div>
            <strong>{selectedProblem.label}</strong>
            <p>{selectedModel.family} · {selectedModel.name}{isBinary ? '' : ` · ${selectedLabel}`}</p>
          </div>
        </div>
      </div>
    </section>

    <footer class="status" class:visible={mapStage !== 'hidden'} class:hidden={mapStage === 'hidden'}>
      <p>
        <strong>{rankedCountries.length}</strong> países visibles ·
        Cobertura del dataset: {asDate(data.dataset.dateRange.start)} → {asDate(data.dataset.dateRange.end)}
      </p>
      <p>
        Generado: {new Date(data.generatedAt).toLocaleString('es-ES')}
      </p>
    </footer>
  </section>

  {#if showIntro}
    <GlobeIntro
      forecastWindow={data.dataset.forecastWindow}
      datasetEnd={data.dataset.dateRange.end}
      on:enter={onIntroEnter}
      on:complete={onIntroComplete}
    />
  {/if}
</main>
