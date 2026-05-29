<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { gsap } from 'gsap';
  import * as THREE from 'three';
  import { geoContains } from 'd3-geo';
  import { feature as topoFeature } from 'topojson-client';
  import landTopology from 'world-atlas/land-110m.json';

  const dispatch = createEventDispatcher<{ enter: void; complete: void }>();

  // Props
  export let forecastWindow: { start: string; end: string } = {
    start: '',
    end: '',
  };
  export let datasetEnd: string = '';

  // Rotación final que orienta Europa hacia la cámara.
  const EUROPE_ROTATION = new THREE.Euler(0.18, -2.18, 0.0, 'XYZ');

  const SHORT_MONTH = {
    '01': 'ene', '02': 'feb', '03': 'mar', '04': 'abr',
    '05': 'may', '06': 'jun', '07': 'jul', '08': 'ago',
    '09': 'sep', '10': 'oct', '11': 'nov', '12': 'dic',
  } as Record<string, string>;

  function shortDate(iso: string): string {
    // espera YYYY-MM-DD
    if (!iso) return '—';
    const [, month, day] = iso.split('-');
    return `${parseInt(day, 10)} ${SHORT_MONTH[month] ?? month}`;
  }

  function shortDateWithYear(iso: string): string {
    if (!iso) return '—';
    const [year, month, day] = iso.split('-');
    return `${parseInt(day, 10)} ${SHORT_MONTH[month] ?? month} ${year}`;
  }

  $: forecastLabel = forecastWindow.start && forecastWindow.end
    ? `${shortDate(forecastWindow.start)} → ${shortDateWithYear(forecastWindow.end)}`
    : '—';
  $: datasetLabel = datasetEnd ? `Datos hasta ${shortDateWithYear(datasetEnd)}` : 'ACLED · Eurostat';

  let canvasWrap: HTMLDivElement;
  let titleEl: HTMLHeadingElement;
  let subtitleEl: HTMLParagraphElement;
  let eyebrowEl: HTMLParagraphElement;
  let metricsEl: HTMLDivElement;
  let ctaEl: HTMLButtonElement;
  let footerEl: HTMLDivElement;

  let shellOpacity = 1;
  let entering = false;
  let initialised = false;

  let renderer: THREE.WebGLRenderer | null = null;
  let scene: THREE.Scene | null = null;
  let camera: THREE.PerspectiveCamera | null = null;
  let globe: THREE.Group | null = null;
  let atmosphere: THREE.Mesh | null = null;
  let frameId = 0;
  let resizeObserver: ResizeObserver | null = null;
  let idleSpinTween: gsap.core.Tween | null = null;
  let idleBobTween: gsap.core.Tween | null = null;
  let revealTimeline: gsap.core.Timeline | null = null;

  // Reloj para que las animaciones sean independientes del frame rate.
  const clock = new THREE.Clock();

  function latLonToVector(lat: number, lon: number, radius: number): THREE.Vector3 {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    return new THREE.Vector3(
      -(radius * Math.sin(phi) * Math.cos(theta)),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.sin(theta),
    );
  }

  function buildLandDots(radius: number): THREE.InstancedMesh {
    const topology = landTopology as any;
    const landShape = topoFeature(topology, topology.objects.land) as any;

    const points: Array<{ lat: number; lon: number }> = [];
    const step = 1.7;
    for (let lat = -58; lat <= 84; lat += step) {
      for (let lon = -180; lon <= 180; lon += step) {
        if (geoContains(landShape, [lon, lat])) {
          points.push({ lat, lon });
        }
      }
    }

    const geometry = new THREE.SphereGeometry(0.0085, 6, 6);
    const material = new THREE.MeshStandardMaterial({
      color: 0x0c0c0c,
      roughness: 0.5,
      metalness: 0.08,
    });
    const mesh = new THREE.InstancedMesh(geometry, material, points.length);
    const helper = new THREE.Object3D();

    points.forEach((point, idx) => {
      const position = latLonToVector(point.lat, point.lon, radius);
      helper.position.copy(position);
      helper.lookAt(position.clone().multiplyScalar(2));
      helper.updateMatrix();
      mesh.setMatrixAt(idx, helper.matrix);
    });

    mesh.instanceMatrix.needsUpdate = true;
    return mesh;
  }

  function buildAtmosphere(radius: number): THREE.Mesh {
    // Halo sutil monocromo emulando atmósfera mediante shader Fresnel.
    const vertexShader = /* glsl */ `
      varying vec3 vNormal;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;
    const fragmentShader = /* glsl */ `
      varying vec3 vNormal;
      uniform vec3 uColor;
      uniform float uIntensity;
      void main() {
        float fresnel = pow(1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.6);
        gl_FragColor = vec4(uColor, fresnel * uIntensity);
      }
    `;
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      transparent: true,
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: {
        uColor: { value: new THREE.Color('#111111') },
        uIntensity: { value: 0.35 },
      },
    });
    return new THREE.Mesh(new THREE.SphereGeometry(radius, 64, 64), material);
  }

  function animate(): void {
    if (!renderer || !scene || !camera) return;
    frameId = requestAnimationFrame(animate);
    clock.getDelta();
    renderer.render(scene, camera);
  }

  function isCompact(width: number, height: number): boolean {
    // En pantallas estrechas centramos el globo bajo el texto.
    return width / height < 1.05 || width < 880;
  }

  function updateLayout(): void {
    if (!renderer || !camera || !canvasWrap || !globe) return;
    const width = canvasWrap.clientWidth;
    const height = canvasWrap.clientHeight;
    if (width === 0 || height === 0) return;

    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    if (isCompact(width, height)) {
      globe.position.set(0, -0.05, 0);
      globe.scale.setScalar(0.7);
    } else {
      // Desplazamos el globo a la derecha para liberar el espacio del texto.
      const aspect = width / height;
      const offsetX = aspect > 2.0 ? 0.85 : aspect > 1.6 ? 0.7 : 0.55;
      globe.position.set(offsetX, 0, 0);
      globe.scale.setScalar(aspect > 1.6 ? 0.95 : 0.85);
    }
  }

  function normalizeAngle(value: number): number {
    return THREE.MathUtils.euclideanModulo(value + Math.PI, Math.PI * 2) - Math.PI;
  }

  function closestAngle(from: number, target: number): number {
    let delta = normalizeAngle(target) - normalizeAngle(from);
    if (delta > Math.PI) delta -= Math.PI * 2;
    if (delta < -Math.PI) delta += Math.PI * 2;
    return from + delta;
  }

  function startEntry(): void {
    if (!camera || !globe || entering) return;
    entering = true;
    dispatch('enter');

    idleSpinTween?.kill();
    idleBobTween?.kill();

    globe.rotation.set(
      normalizeAngle(globe.rotation.x),
      normalizeAngle(globe.rotation.y),
      normalizeAngle(globe.rotation.z),
    );

    const targetRotation = {
      x: closestAngle(globe.rotation.x, EUROPE_ROTATION.x),
      y: closestAngle(globe.rotation.y, EUROPE_ROTATION.y),
      z: closestAngle(globe.rotation.z, EUROPE_ROTATION.z),
    };

    const fade = { shell: 1, atmo: 0.55 };

    // Fade del UI con stagger (más cinematográfico).
    gsap.to([eyebrowEl, titleEl, subtitleEl, metricsEl, ctaEl, footerEl], {
      opacity: 0,
      y: -14,
      duration: 0.55,
      ease: 'power3.in',
      stagger: 0.04,
      overwrite: true,
    });

    const tl = gsap.timeline({
      defaults: { overwrite: true },
      onComplete: () => {
        dispatch('complete');
      },
    });

    // Centramos el globo mientras lo orientamos a Europa.
    tl.to(globe.position, { x: 0, y: 0, duration: 1.5, ease: 'power3.inOut' }, 0);
    tl.to(globe.rotation, {
      x: targetRotation.x,
      y: targetRotation.y,
      z: targetRotation.z,
      duration: 2.0,
      ease: 'power3.inOut',
    }, 0);

    // Acercamiento de cámara.
    tl.to(camera.position, { z: 1.6, duration: 2.2, ease: 'power2.in' }, 0.05);

    // Boost momentáneo del halo justo antes del fade.
    tl.to(
      fade,
      {
        atmo: 1.3,
        duration: 0.45,
        ease: 'power2.out',
        onUpdate: () => {
          if (atmosphere) {
            (atmosphere.material as THREE.ShaderMaterial).uniforms.uIntensity.value = fade.atmo;
          }
        },
      },
      1.4,
    );

    // Crossfade del shell.
    tl.to(
      fade,
      {
        shell: 0,
        duration: 0.55,
        ease: 'power2.out',
        onUpdate: () => {
          shellOpacity = fade.shell;
        },
      },
      1.65,
    );
  }

  function playRevealAnimation(): void {
    // Reveal escalonado de las piezas de texto.
    revealTimeline?.kill();

    const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });
    tl.from(eyebrowEl, { opacity: 0, y: 14, duration: 0.6 }, 0.15)
      .from(titleEl, { opacity: 0, y: 28, duration: 0.85 }, '-=0.42')
      .from(subtitleEl, { opacity: 0, y: 16, duration: 0.65 }, '-=0.55')
      .from(
        metricsEl.children,
        { opacity: 0, y: 14, duration: 0.5, stagger: 0.07 },
        '-=0.4',
      )
      .from(ctaEl, { opacity: 0, y: 14, scale: 0.97, duration: 0.55 }, '-=0.35')
      .from(footerEl, { opacity: 0, y: 12, duration: 0.55 }, '-=0.35');

    revealTimeline = tl;
  }

  onMount(() => {
    if (!canvasWrap || initialised) return;
    initialised = true;

    scene = new THREE.Scene();
    scene.background = new THREE.Color('#f4f4f4');

    camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
    camera.position.set(0, 0, 3.5);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    canvasWrap.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 0.85);
    const key = new THREE.DirectionalLight(0xffffff, 1.05);
    key.position.set(3.6, 2.1, 3.4);
    const rim = new THREE.DirectionalLight(0xffc4b6, 0.55);
    rim.position.set(-3.4, -0.8, -2.4);
    const fill = new THREE.DirectionalLight(0xeaf0ff, 0.35);
    fill.position.set(-1.2, 2.2, 2.5);
    scene.add(ambient, key, rim, fill);

    globe = new THREE.Group();
    globe.rotation.set(0.22, -1.05, 0.02);

    const planet = new THREE.Mesh(
      new THREE.SphereGeometry(0.78, 96, 96),
      new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.88,
        metalness: 0.05,
      }),
    );

    const dots = buildLandDots(0.792);
    atmosphere = buildAtmosphere(0.92);

    globe.add(atmosphere, planet, dots);
    scene.add(globe);

    idleSpinTween = gsap.to(globe.rotation, {
      y: globe.rotation.y + Math.PI * 2,
      duration: 32,
      ease: 'none',
      repeat: -1,
    });
    idleBobTween = gsap.to(globe.position, {
      y: 0.025,
      duration: 3.4,
      ease: 'sine.inOut',
      yoyo: true,
      repeat: -1,
    });

    requestAnimationFrame(() => {
      updateLayout();
      playRevealAnimation();
    });
    animate();

    resizeObserver = new ResizeObserver(() => updateLayout());
    resizeObserver.observe(canvasWrap);

    return () => {
      if (frameId) cancelAnimationFrame(frameId);
      resizeObserver?.disconnect();
      idleSpinTween?.kill();
      idleBobTween?.kill();
      revealTimeline?.kill();
      renderer?.dispose();
      renderer?.domElement.remove();
      scene?.traverse((obj: THREE.Object3D) => {
        if (!(obj instanceof THREE.Mesh)) return;
        obj.geometry.dispose();
        if (Array.isArray(obj.material)) {
          obj.material.forEach((m: THREE.Material) => m.dispose());
        } else {
          obj.material.dispose();
        }
      });
    };
  });
</script>

<div class="intro-shell" style:opacity={shellOpacity}>
  <div class="grid-bg" aria-hidden="true">
    <svg width="100%" height="100%" preserveAspectRatio="none">
      <defs>
        <pattern id="grid" width="64" height="64" patternUnits="userSpaceOnUse">
          <path d="M 64 0 L 0 0 0 64" fill="none" stroke="rgba(0,0,0,0.04)" stroke-width="1" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#grid)" />
    </svg>
  </div>

  <div class="canvas-wrap" bind:this={canvasWrap}></div>

  <div class="vignette" aria-hidden="true"></div>
  <div class="grain" aria-hidden="true"></div>

  <div class="intro-content" class:hide={entering}>
    <div class="content-grid">
      <div class="text-block">
        <p class="eyebrow" bind:this={eyebrowEl}>
          <span class="dot"></span>
          Early Signal System · v1.0
        </p>

        <h1 class="display" bind:this={titleEl}>
          Forecast Europa<br />
          Conflictividad semanal
        </h1>

        <p class="lead" bind:this={subtitleEl}>
          Sistema de alerta temprana basado en datos ACLED y Eurostat. Tres familias de modelos
          —lineal, bagging y boosting— estiman el riesgo de eventos disruptivos a una semana vista.
        </p>

        <div class="metrics-strip" bind:this={metricsEl}>
          <div class="metric">
            <span class="metric-value">43</span>
            <span class="metric-label">países</span>
          </div>
          <span class="metric-sep"></span>
          <div class="metric">
            <span class="metric-value">9</span>
            <span class="metric-label">modelos</span>
          </div>
          <span class="metric-sep"></span>
          <div class="metric">
            <span class="metric-value">0.93</span>
            <span class="metric-label">macro ROC</span>
          </div>
        </div>

        <button class="cta" on:click={startEntry} type="button" bind:this={ctaEl}>
          <span>Entrar al mapa</span>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path
              d="M2.5 8h11M9 3.5l4.5 4.5L9 12.5"
              stroke="currentColor"
              stroke-width="1.5"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
        </button>
      </div>
    </div>
  </div>

  <div class="footer-ticker" bind:this={footerEl}>
    <div class="tick">
      <span class="tick-label">DATA</span>
      <span class="tick-value">{datasetLabel}</span>
    </div>
    <div class="tick">
      <span class="tick-label">REGIÓN</span>
      <span class="tick-value">Europa</span>
    </div>
    <div class="tick">
      <span class="tick-label">VENTANA</span>
      <span class="tick-value">{forecastLabel}</span>
    </div>
    <div class="tick coord">
      <span class="tick-label">52° 14°</span>
      <span class="tick-value">N · E</span>
    </div>
  </div>
</div>

<style>
  .intro-shell {
    position: fixed;
    inset: 0;
    z-index: 40;
    background:
      radial-gradient(ellipse 80% 60% at 70% 35%, rgba(0, 0, 0, 0.06), transparent 70%),
      radial-gradient(ellipse 60% 80% at 15% 70%, rgba(0, 0, 0, 0.04), transparent 70%),
      #f4f4f4;
    transition: opacity 0.55s ease;
    overflow: hidden;
  }

  .grid-bg {
    position: absolute;
    inset: 0;
    opacity: 0.5;
    pointer-events: none;
  }

  .canvas-wrap {
    position: absolute;
    inset: 0;
  }

  .canvas-wrap :global(canvas) {
    width: 100% !important;
    height: 100% !important;
    display: block;
  }

  .vignette {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
      radial-gradient(circle at 70% 50%, transparent 35%, rgba(0, 0, 0, 0.08) 90%),
      linear-gradient(180deg, rgba(255, 255, 255, 0) 60%, rgba(0, 0, 0, 0.06));
  }

  .grain {
    position: absolute;
    inset: 0;
    pointer-events: none;
    opacity: 0.35;
    mix-blend-mode: multiply;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180' viewBox='0 0 180 180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.06 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
  }

  .intro-content {
    position: absolute;
    inset: 0;
    display: grid;
    align-items: center;
    padding: clamp(1.5rem, 4vw, 3.5rem);
    transition: opacity 0.5s ease, transform 0.55s cubic-bezier(0.2, 0.8, 0.2, 1);
  }

  .intro-content.hide {
    opacity: 0;
    transform: translateY(16px);
    pointer-events: none;
  }

  .content-grid {
    width: min(1280px, 100%);
    margin: 0 auto;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    align-items: center;
    gap: 2rem;
  }

  .text-block {
    display: grid;
    gap: 1.1rem;
    color: #111;
    max-width: 32rem;
  }

  .eyebrow {
    margin: 0;
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    color: #4a4a4a;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }

  .eyebrow .dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: #111;
    box-shadow: 0 0 0 4px rgba(17, 17, 17, 0.14);
    animation: pulse 2.4s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 4px rgba(17, 17, 17, 0.14); }
    50%      { box-shadow: 0 0 0 8px rgba(17, 17, 17, 0.0); }
  }

  .display {
    margin: 0;
    font-family: 'Inter Tight', 'Inter', system-ui, sans-serif;
    font-weight: 700;
    font-size: clamp(2.6rem, 6.4vw, 5.2rem);
    line-height: 0.98;
    letter-spacing: -0.03em;
    color: #0c0c0c;
  }

  .lead {
    margin: 0;
    font-size: clamp(0.94rem, 1.5vw, 1.05rem);
    color: #2a2a2a;
    line-height: 1.55;
    max-width: 36rem;
  }

  .metrics-strip {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem 1.1rem;
    padding-top: 0.4rem;
  }

  .metric {
    display: grid;
    gap: 0.15rem;
    line-height: 1;
  }

  .metric-value {
    font-family: 'Inter Tight', 'Inter', system-ui, sans-serif;
    font-weight: 600;
    font-size: 1.55rem;
    color: #111;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
  }

  .metric-label {
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #6c6c6c;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }

  .metric-sep {
    width: 1px;
    height: 26px;
    background: rgba(0, 0, 0, 0.16);
  }

  .cta {
    margin-top: 0.4rem;
    border: 1px solid #111;
    background: #111;
    color: #f5f5f5;
    font-family: inherit;
    padding: 0.78rem 1.4rem 0.78rem 1.55rem;
    border-radius: 999px;
    font-weight: 600;
    letter-spacing: 0.02em;
    font-size: 0.92rem;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    width: fit-content;
    position: relative;
    overflow: hidden;
    transition:
      transform 0.32s cubic-bezier(0.2, 0.8, 0.2, 1),
      box-shadow 0.32s ease,
      background 0.32s ease;
  }

  .cta::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent 0%, rgba(255, 255, 255, 0.12) 50%, transparent 100%);
    transform: translateX(-100%);
    transition: transform 0.7s ease;
  }

  .cta:hover {
    transform: translateY(-2px);
    box-shadow: 0 22px 40px rgba(0, 0, 0, 0.22);
    background: #1f1f1f;
    border-color: #1f1f1f;
  }

  .cta:hover::before {
    transform: translateX(100%);
  }

  .cta svg {
    transition: transform 0.32s cubic-bezier(0.2, 0.8, 0.2, 1);
  }

  .cta:hover svg {
    transform: translateX(4px);
  }

  .footer-ticker {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 0.9rem clamp(1.2rem, 3vw, 2.4rem);
    display: flex;
    align-items: center;
    gap: clamp(1rem, 3vw, 2.5rem);
    flex-wrap: wrap;
    border-top: 1px solid rgba(0, 0, 0, 0.08);
    background: rgba(244, 244, 244, 0.75);
    backdrop-filter: blur(8px);
    z-index: 2;
  }

  .tick {
    display: inline-flex;
    align-items: baseline;
    gap: 0.55rem;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.74rem;
  }

  .tick-label {
    color: #8c8c8c;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.66rem;
  }

  .tick-value {
    color: #1a1a1a;
    font-weight: 500;
  }

  .coord {
    margin-left: auto;
  }

  @media (max-width: 880px) {
    .content-grid {
      grid-template-columns: 1fr;
      gap: 1rem;
    }

    .text-block {
      max-width: none;
      text-align: left;
    }

    .intro-content {
      align-items: flex-end;
      padding-bottom: 5rem;
    }

    .display {
      font-size: clamp(2.2rem, 9vw, 3.4rem);
    }

    .coord {
      margin-left: 0;
    }
  }
</style>
