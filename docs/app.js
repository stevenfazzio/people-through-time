/* People Through Time -- a datamap of the most famous people of every era, with a birth-year slider.

   Data (written by pipeline/12_export_frontend.py into data/):
     core.json + points.bin  positions per (person, window), names, label centroids, ribbon sizes
     details.json            hovercard text, fetched after the first paint

   The slider value s runs over the 32 precomputed windows (0..31). Between windows w and w+1 (t = s - w),
   a person in both glides between their two positions; one in only one of them fades in or out where they
   stand. Labels do the same with their track's centroid. Releasing the slider eases s to the nearest window,
   the only states that are real layouts. The slider is drawn as a ThemeRiver of the coarse tracks: the band
   covers the 5 birth-year bins of the window on the map. */
(function () {
  "use strict";

  const DATA = "data/";
  const SETTLE_MS = 420;
  const STEP_MS = 560;
  const PLAY_WINDOWS_PER_S = 0.75;
  const DIM = 0.1; // alpha factor for points outside a highlight
  const FONT = '"Source Sans 3", "Source Sans Pro", system-ui, sans-serif';
  // Label layers, finest first (stage 09's layer 0 is the finest). As in datamapplot 0.7.3, labels have fixed
  // pixel sizes, coarse labels win collisions and finer ones appear as zooming spreads them apart -- but the
  // collisions are resolved here on the CPU (placeLabels), not with deck.gl's CollisionFilterExtension, whose
  // per-anchor test let partly overlapping labels through and which can't crossfade moving labels.
  const LABEL_STYLE = [
    { size: 12, weight: 600, wrap: 24, base: 0, color: [205, 211, 220] },
    { size: 14, weight: 600, wrap: 24, base: 300, color: [228, 232, 238] },
    { size: 17, weight: 700, wrap: 22, base: 600, color: null },
  ];
  const GAP = 2; // px between ribbons
  const LABEL_PAD = [10, 6]; // px added around each label's box (x, y) when testing overlaps
  const LABEL_FADE_MS = 180;
  // Props handed to every label layer on every frame: created once, so deck.gl sees them unchanged.
  let LABEL_PROPS = null;
  const TICK_YEARS = [-2999, -999, -499, 1, 500, 1000, 1200, 1400, 1500, 1600, 1700, 1800, 1850, 1900, 1925,
    1950, 1970, 1990, 2005];

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const easeInOut = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);

  function fmtYear(y) {
    if (y <= 0) {
      const bc = 1 - y;
      return (bc >= 10000 ? bc.toLocaleString("en-US") : bc) + " BC";
    }
    return y < 1000 ? "AD " + y : String(y);
  }
  function fmtViews(v) {
    if (v >= 1e6) return (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + "M";
    if (v >= 1e3) return Math.round(v / 1e3) + "k";
    return String(v);
  }
  function articleUrl(ref) {
    const i = ref.indexOf(":");
    const pre = ref.slice(0, i), title = ref.slice(i + 1);
    if (pre === "wd") return "https://www.wikidata.org/wiki/" + title;
    return "https://" + pre + ".wikipedia.org/wiki/" + encodeURIComponent(title);
  }
  function thumbUrl(code) {
    if (!code) return "";
    if (code.startsWith("T/")) return "https://thumb.wikimedia.org/wikipedia/" + code.slice(2);
    if (code.startsWith("U/")) return "https://upload.wikimedia.org/wikipedia/" + code.slice(2);
    return code;
  }
  function wrap(text, width) {
    const words = text.split(" ");
    const lines = [];
    let line = "";
    for (const w of words) {
      if (line && (line + " " + w).length > width) {
        lines.push(line);
        line = w;
      } else line = line ? line + " " + w : w;
    }
    if (line) lines.push(line);
    return lines.join("\n");
  }
  const normText = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

  // ------------------------------------------------------------------------------------------------------
  // State
  const S = {
    s: 0,
    anim: null, // requestAnimationFrame id of a slider animation
    playing: false,
    dragging: false,
    hover: -1, // hovered person
    hoverXY: null,
    pinned: -1,
    hlTrack: -1, // coarse track id highlighted from the chart
    mask: null, // Uint8Array of search matches, or null
    zoom: 0,
    zoom0: 0,
    renderQueued: false,
  };
  let core, D = null; // D = details, once loaded
  let NP, NW, N, X, Y, PT, W0, NWIN, OFF, RAD, members, labelsByWindow, coarseRgb, coarseOf, trackText;
  let deckgl, frame = { n: 0 };
  let nameIndex, descIndex = null;

  // ------------------------------------------------------------------------------------------------------
  // Loading
  async function main() {
    const [c, buf] = await Promise.all([
      fetch(DATA + "core.json").then((r) => r.json()),
      fetch(DATA + "points.bin").then((r) => r.arrayBuffer()),
      document.fonts.load(`600 14px ${FONT}`),
      document.fonts.load(`700 17px ${FONT}`),
    ]);
    core = c;
    decode(buf);
    initDeck();
    initTimeline();
    initSearch();
    initKeys();
    setS(16); // opens on people born 1831-1924: busy, and full of familiar names
    $("loading").hidden = true;
    // For debugging from the console; renderNow() draws without waiting for an animation frame.
    window.__ptt = { S, deck: () => deckgl, core: () => core, setS, goTo: (p) => goTo(p),
      renderNow: () => { render(); for (let i = 0; i < 3; i++) deckgl.redraw("debug"); } };
    fetch(DATA + "details.json").then((r) => r.json()).then((d) => {
      D = d;
      descIndex = D.desc.map(normText);
      if (S.hover >= 0 || S.pinned >= 0) showCard();
    });
  }

  function decode(buf) {
    NP = core.people.name.length;
    NW = core.n_windows;
    N = core.n_points;
    const q = new Uint16Array(buf, 0, 2 * N);
    PT = new Uint8Array(buf, 4 * N, N);
    const [xlo, xhi, ylo, yhi] = core.bounds;
    X = new Float32Array(N);
    Y = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      X[i] = xlo + (q[2 * i] / 65535) * (xhi - xlo);
      Y[i] = ylo + (q[2 * i + 1] / 65535) * (yhi - ylo);
    }
    W0 = core.people.w0;
    NWIN = core.people.nw;
    RAD = Float32Array.from(core.people.r, (r) => r / 10);
    OFF = new Int32Array(NP);
    const counts = new Int32Array(NW);
    let acc = 0;
    for (let p = 0; p < NP; p++) {
      OFF[p] = acc;
      acc += NWIN[p];
      for (let k = 0; k < NWIN[p]; k++) counts[W0[p] + k]++;
    }
    members = Array.from(counts, (n) => new Int32Array(n));
    const fill = new Int32Array(NW);
    for (let p = 0; p < NP; p++) for (let k = 0; k < NWIN[p]; k++) {
      const w = W0[p] + k;
      members[w][fill[w]++] = p; // ascending person index = rank order
    }
    // Tracks: coarse colours by coarse track id; wrapped label text by global index.
    const T = core.tracks;
    const pal = core.coarse.palette.map(hexRgb);
    coarseRgb = [];
    coarseOf = {}; // coarse track id -> global index
    T.layer.forEach((layer, g) => {
      if (layer === 2) {
        coarseOf[T.id[g]] = g;
        coarseRgb[T.id[g]] = pal[T.color[g]];
      }
    });
    coarseRgb[255] = hexRgb(core.coarse.unlabelled);
    trackText = T.name.map((n, g) => wrap(n, LABEL_STYLE[T.layer[g]].wrap));
    labelsByWindow = Array.from({ length: NW }, () => new Map());
    const L = core.labels;
    L.window.forEach((w, i) => labelsByWindow[w].set(L.track[i], i));
    nameIndex = core.people.name.map(normText);
  }

  // ------------------------------------------------------------------------------------------------------
  // Map
  function mapSize() {
    const el = $("map");
    return [el.clientWidth, el.clientHeight];
  }

  function fitZoom() {
    // Fit the central 99.6% of all points (normalized windows share one frame).
    const xs = Array.from(X).sort((a, b) => a - b), ys = Array.from(Y).sort((a, b) => a - b);
    const qa = (a, f) => a[Math.floor(f * (a.length - 1))];
    const x0 = qa(xs, 0.002), x1 = qa(xs, 0.998), y0 = qa(ys, 0.002), y1 = qa(ys, 0.998);
    const [w, h] = mapSize();
    const zoom = Math.log2(Math.min(w / (x1 - x0), h / (y1 - y0)) * 0.9);
    return { target: [(x0 + x1) / 2, (y0 + y1) / 2, 0], zoom };
  }

  // Fit the view once the map has a size; until then deck.gl has a zero-size canvas.
  function refit() {
    const [w, h] = mapSize();
    if (w < 80 || h < 80) return;
    const fit = fitZoom();
    S.zoom = S.zoom0 = fit.zoom;
    S.fitTarget = fit.target;
    S.fitted = true;
    deckgl.setProps({ initialViewState: { target: fit.target, zoom: fit.zoom, minZoom: fit.zoom - 1.5, maxZoom: fit.zoom + 7 } });
  }

  function initDeck() {
    LABEL_PROPS = {
      fontFamily: FONT,
      lineHeight: 1.05,
      characterSet: core.charset + " …",
      fontSettings: { sdf: true, fontSize: 64, buffer: 8 },
      outlineWidth: 3,
      outlineColor: [13, 16, 21, 235],
      getTextAnchor: "middle",
      getAlignmentBaseline: "center",
      getPixelOffset: (d) => d.off,
    };
    const [mw, mh] = mapSize();
    const fit = mw >= 80 && mh >= 80 ? fitZoom() : { target: [0, 0, 0], zoom: 6 };
    S.fitted = mw >= 80 && mh >= 80;
    S.zoom = S.zoom0 = fit.zoom;
    S.fitTarget = fit.target;
    deckgl = new deck.Deck({
      parent: $("map"),
      views: new deck.OrthographicView({ id: "ortho", flipY: true }),
      initialViewState: { target: fit.target, zoom: fit.zoom, minZoom: fit.zoom - 1.5, maxZoom: fit.zoom + 7 },
      controller: { scrollZoom: { speed: 0.04, smooth: true }, dragRotate: false, keyboard: false, inertia: false },
      getCursor: ({ isDragging, isHovering }) => (isDragging ? "grabbing" : isHovering ? "pointer" : "grab"),
      onViewStateChange: ({ viewState }) => {
        S.zoom = viewState.zoom;
        queueRender();
      },
      onHover: (info) => {
        const p = info.picked && info.layer && info.layer.id === "points" ? frame.pid[info.index] : -1;
        S.hoverXY = info.picked ? [info.x, info.y] : null;
        if (p !== S.hover) {
          S.hover = p;
          highlightRibbonOf(p);
          queueRender();
        }
        showCard();
      },
      onClick: (info) => {
        const p = info.picked && info.layer && info.layer.id === "points" ? frame.pid[info.index] : -1;
        if (p >= 0) {
          const ref = D ? D.ref[p] : null;
          if (ref) window.open(articleUrl(ref), "_blank", "noopener");
        } else if (S.pinned >= 0) {
          S.pinned = -1;
          showCard();
          queueRender();
        }
      },
      layers: [],
    });
  }

  function split(s) {
    let w = Math.floor(s);
    if (w >= NW - 1) w = NW - 2;
    return [w, s - w];
  }

  function mix(a, b, t) {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  }

  // A person's position, colour track and alpha at slider value s (null when absent).
  function personAt(p, s) {
    const [w, t] = split(s);
    const inA = W0[p] <= w && w < W0[p] + NWIN[p];
    const inB = W0[p] <= w + 1 && w + 1 < W0[p] + NWIN[p];
    if (!inA && !inB) return null;
    const a = OFF[p] + (w - W0[p]), b = a + 1;
    if (inA && inB) return { x: X[a] + (X[b] - X[a]) * t, y: Y[a] + (Y[b] - Y[a]) * t, alpha: 1, track: t < 0.5 ? PT[a] : PT[b] };
    if (inA) return { x: X[a], y: Y[a], alpha: 1 - t, track: PT[a] };
    const b0 = OFF[p]; // w + 1 is the person's first window
    return { x: X[b0], y: Y[b0], alpha: t, track: PT[b0] };
  }

  let buf = null;
  function buildPoints() {
    const [w, t] = split(S.s);
    const A = members[w], B = members[w + 1];
    const max = A.length + B.length;
    if (!buf || buf.cap < max) {
      buf = { cap: max, pos: new Float32Array(2 * max), col: new Uint8Array(4 * max), rad: new Float32Array(max),
        pid: new Int32Array(max) };
    }
    const { pos, col, rad, pid } = buf;
    let n = 0, i = A.length - 1, j = B.length - 1;
    const hl = S.hlTrack, mask = S.mask;
    // Merge A and B from the end: least famous first, so the famous are drawn on top.
    while (i >= 0 || j >= 0) {
      let p, inA = false, inB = false;
      if (j < 0 || (i >= 0 && A[i] > B[j])) { p = A[i--]; inA = true; }
      else if (i < 0 || B[j] > A[i]) { p = B[j--]; inB = true; }
      else { p = A[i--]; j--; inA = inB = true; }
      let x, y, alpha, rgb, track;
      if (inA && inB) {
        const a = OFF[p] + (w - W0[p]), b = a + 1;
        x = X[a] + (X[b] - X[a]) * t;
        y = Y[a] + (Y[b] - Y[a]) * t;
        alpha = 1;
        rgb = PT[a] === PT[b] ? coarseRgb[PT[a]] : mix(coarseRgb[PT[a]], coarseRgb[PT[b]], t);
        track = t < 0.5 ? PT[a] : PT[b];
      } else {
        const a = inA ? OFF[p] + (w - W0[p]) : OFF[p];
        x = X[a]; y = Y[a]; alpha = inA ? 1 - t : t; rgb = coarseRgb[PT[a]]; track = PT[a];
      }
      if (alpha < 0.01) continue;
      if (hl >= 0 && track !== hl) alpha *= DIM;
      if (mask && !mask[p]) alpha *= DIM;
      pos[2 * n] = x; pos[2 * n + 1] = y;
      col[4 * n] = rgb[0]; col[4 * n + 1] = rgb[1]; col[4 * n + 2] = rgb[2]; col[4 * n + 3] = Math.round(235 * alpha);
      rad[n] = RAD[p];
      pid[n] = p;
      n++;
    }
    frame = { n, pid, pos: pos.subarray(0, 2 * n), col: col.subarray(0, 4 * n), rad: rad.subarray(0, n) };
  }

  // Candidate labels for the current slider value: every track in window w or w + 1, its centroid lerped
  // when it is in both, fading with the slider when it is in only one.
  function buildLabels() {
    const [w, t] = split(S.s);
    const LA = labelsByWindow[w], LB = labelsByWindow[w + 1];
    const L = core.labels, T = core.tracks;
    const out = [];
    const push = (g, x, y, n, alpha) => {
      if (alpha < 0.04) return;
      const layer = T.layer[g];
      // Coarse beats mid beats fine, bigger regions first; a label fading with the slider yields.
      const prio = (layer * 3000 + Math.min(n, 2500)) * (alpha < 0.5 ? 0.25 : 1);
      out.push({ g, layer, x, y, alpha, prio });
    };
    for (const [g, ia] of LA) {
      const ib = LB.get(g);
      if (ib !== undefined) {
        push(g, L.x[ia] + (L.x[ib] - L.x[ia]) * t, L.y[ia] + (L.y[ib] - L.y[ia]) * t, L.n[ia] + (L.n[ib] - L.n[ia]) * t, 1);
      } else push(g, L.x[ia], L.y[ia], L.n[ia], 1 - t);
    }
    for (const [g, ib] of LB) if (!LA.has(g)) push(g, L.x[ib], L.y[ib], L.n[ib], t);
    return out;
  }

  let labelBox = null, labelVis = null, lastPlace = 0;
  function labelSize(g) {
    if (!labelBox) labelBox = new Array(core.tracks.name.length);
    if (!labelBox[g]) {
      const st = LABEL_STYLE[core.tracks.layer[g]];
      const lines = trackText[g].split("\n");
      const w = Math.max(...lines.map((ln) => textWidth(ln, st.size, st.weight)));
      labelBox[g] = [w, lines.length * st.size * 1.05];
    }
    return labelBox[g];
  }

  // Greedy placement in priority order: a label shows if its padded screen box overlaps no label placed
  // before it. Visibility eases toward shown / hidden over LABEL_FADE_MS, so labels crossfade.
  function placeLabels(cands) {
    const vp = deckgl.getViewports()[0];
    const now = performance.now();
    const step = lastPlace ? Math.min(1, (now - lastPlace) / LABEL_FADE_MS) : 1;
    lastPlace = now;
    if (!labelVis) labelVis = new Float32Array(core.tracks.name.length);
    // The title panel counts as a placed label, so no label hides under it.
    const [mw, mh] = mapSize();
    const panel = $("title").getBoundingClientRect();
    const placed = panel.width ? [[panel.left - 4, panel.right + 4, panel.top - 4, panel.bottom + 4]] : [];
    const EDGE = 4;
    const target = new Uint8Array(labelVis.length);
    cands.sort((a, b) => b.prio - a.prio);
    for (const c of cands) {
      if (!vp) { target[c.g] = c.layer === 2 ? 1 : 0; continue; }
      const [sx, sy] = vp.project([c.x, c.y]);
      if (sx < 0 || sx > mw || sy < 0 || sy > mh) continue; // anchor off screen
      const [w, h] = labelSize(c.g);
      // A label whose anchor is on screen but whose text runs off an edge is nudged back inside.
      const dx = Math.max(0, EDGE - (sx - w / 2)) - Math.max(0, sx + w / 2 - (mw - EDGE));
      const dy = Math.max(0, EDGE - (sy - h / 2)) - Math.max(0, sy + h / 2 - (mh - EDGE));
      const x = sx + dx, y = sy + dy;
      const box = [x - w / 2 - LABEL_PAD[0], x + w / 2 + LABEL_PAD[0], y - h / 2 - LABEL_PAD[1], y + h / 2 + LABEL_PAD[1]];
      c.off = [dx, dy];
      if (placed.some((b) => box[0] < b[1] && b[0] < box[1] && box[2] < b[3] && b[2] < box[3])) continue;
      placed.push(box);
      target[c.g] = 1;
    }
    const out = [[], [], []];
    let settling = false;
    const T = core.tracks;
    const live = new Set(cands.map((c) => c.g));
    for (let g = 0; g < labelVis.length; g++) {
      if (!live.has(g)) { labelVis[g] = 0; continue; }
      const v = labelVis[g], tv = target[g];
      labelVis[g] = tv > v ? Math.min(tv, v + step) : Math.max(tv, v - step);
      if (labelVis[g] !== tv) settling = true;
    }
    for (const c of cands) {
      const a = c.alpha * labelVis[c.g];
      if (a < 0.02) continue;
      const st = LABEL_STYLE[c.layer];
      let rgb = st.color || mix(coarseRgb[T.id[c.g]], [255, 255, 255], 0.35);
      let alpha = a;
      if (S.hlTrack >= 0 && c.layer === 2 && T.id[c.g] !== S.hlTrack) alpha *= 0.3;
      out[c.layer].push({ text: trackText[c.g], x: c.x, y: c.y, off: c.off || [0, 0],
        color: [rgb[0], rgb[1], rgb[2], Math.round(255 * alpha)] });
    }
    if (settling) queueRender();
    return out;
  }

  function render() {
    S.renderQueued = false;
    buildPoints();
    const labels = placeLabels(buildLabels());
    const radiusScale = clamp(Math.pow(2, (S.zoom - S.zoom0) * 0.5), 0.8, 3.2);
    const layers = [
      new deck.ScatterplotLayer({
        id: "points",
        data: { length: frame.n, attributes: {
          getPosition: { value: frame.pos, size: 2 },
          getFillColor: { value: frame.col, size: 4 },
          getRadius: { value: frame.rad, size: 1 },
        } },
        radiusUnits: "pixels",
        radiusScale,
        pickable: true,
        stroked: false,
      }),
    ];
    const ringFor = S.hover >= 0 ? S.hover : S.pinned;
    if (ringFor >= 0) {
      const at = personAt(ringFor, S.s);
      if (at) layers.push(new deck.ScatterplotLayer({
        id: "ring",
        data: [at],
        getPosition: (d) => [d.x, d.y],
        getRadius: RAD[ringFor] * radiusScale + 4,
        radiusUnits: "pixels",
        stroked: true,
        filled: false,
        getLineColor: [255, 255, 255, 230],
        lineWidthUnits: "pixels",
        getLineWidth: 1.6,
      }));
    }
    LABEL_STYLE.forEach((st, layer) => {
      layers.push(new deck.TextLayer({
        id: "labels-" + layer,
        data: labels[layer],
        getPosition: (d) => [d.x, d.y],
        getText: (d) => d.text,
        getColor: (d) => d.color,
        getSize: st.size,
        sizeUnits: "pixels",
        fontWeight: st.weight,
        ...LABEL_PROPS,
      }));
    });
    deckgl.setProps({ layers });
    drawBand();
    if (S.pinned >= 0 && S.hover < 0) showCard();
  }

  function queueRender() {
    if (S.renderQueued) return;
    S.renderQueued = true;
    requestAnimationFrame(render);
  }

  // ------------------------------------------------------------------------------------------------------
  // Slider value and animation
  function setS(v) {
    S.s = clamp(v, 0, NW - 1);
    updateReadout();
    queueRender();
  }

  function stopAnim() {
    if (S.anim) cancelAnimationFrame(S.anim);
    S.anim = null;
    S.animTarget = undefined;
    if (S.playing) setPlaying(false);
  }

  function animateTo(target, ms, done) {
    if (S.anim) cancelAnimationFrame(S.anim);
    const from = S.s, to = clamp(target, 0, NW - 1), t0 = performance.now();
    S.animTarget = to;
    if (Math.abs(to - from) < 1e-4) { setS(to); S.animTarget = undefined; if (done) done(); return; }
    const tick = (now) => {
      const u = clamp((now - t0) / ms, 0, 1);
      setS(from + (to - from) * easeInOut(u));
      if (u < 1) S.anim = requestAnimationFrame(tick);
      else { S.anim = null; S.animTarget = undefined; if (done) done(); }
    };
    S.anim = requestAnimationFrame(tick);
  }

  function setPlaying(on) {
    S.playing = on;
    $("play").innerHTML = on ? "&#10074;&#10074;" : "&#9654;";
    $("play").setAttribute("aria-label", on ? "Pause" : "Play");
    if (!on) return;
    if (S.anim) cancelAnimationFrame(S.anim);
    S.animTarget = undefined;
    if (S.s >= NW - 1 - 1e-3) S.s = 0;
    let last = performance.now();
    const tick = (now) => {
      if (!S.playing) return;
      const next = S.s + ((now - last) / 1000) * PLAY_WINDOWS_PER_S;
      last = now;
      if (next >= NW - 1) { setS(NW - 1); S.anim = null; setPlaying(false); return; }
      setS(next);
      S.anim = requestAnimationFrame(tick);
    };
    S.anim = requestAnimationFrame(tick);
  }

  function updateReadout() {
    const w = Math.round(S.s);
    const W = core.windows;
    $("readout").innerHTML = `Born ${esc(fmtYear(W.lo[w]))} – ${esc(fmtYear(W.hi[w]))}` +
      `<span class="n">${W.n[w].toLocaleString("en-US")} people · window ${w + 1} of ${NW}</span>`;
    $("tl-svg").setAttribute("aria-valuenow", String(w));
    if (S.mask && w !== S.statusWindow) { S.statusWindow = w; searchStatus(); }
  }

  function initKeys() {
    document.addEventListener("keydown", (e) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const dir = e.key === "ArrowRight" ? 1 : -1;
        const base = S.anim && S.animTarget !== undefined ? S.animTarget : Math.round(S.s);
        if (S.playing) setPlaying(false);
        animateTo(base + dir, STEP_MS);
      } else if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      } else if (e.key === "Escape") {
        clearSearch();
        S.pinned = -1;
        showCard();
        queueRender();
      }
    });
    $("play").addEventListener("click", togglePlay);
  }
  // Pausing settles on the nearest window, like releasing the band.
  function togglePlay() {
    if (!S.playing) { setPlaying(true); return; }
    setPlaying(false);
    animateTo(Math.round(S.s), SETTLE_MS);
  }

  // ------------------------------------------------------------------------------------------------------
  // Hovercard
  function cardHtml(p) {
    const P = core.people;
    const at = personAt(p, S.s);
    const tr = at ? at.track : 255;
    let pill = "";
    if (tr !== 255) {
      const g = coarseOf[tr];
      const rgb = coarseRgb[tr];
      const name = core.tracks.short[g] || core.tracks.name[g];
      pill = `<span class="pc-pill" style="background:rgba(${rgb.join(",")},0.18);color:rgb(${rgb.join(",")})">${esc(name)}</span>`;
    }
    const d = D;
    const img = d && d.thumb[p] ? `<img class="pc-img" src="${esc(thumbUrl(d.thumb[p]))}" alt="" onerror="this.remove()">` : "";
    const rows = [["Born", P.born[p]]];
    if (d && d.died[p]) rows.push(["Died", d.died[p]]);
    if (d) rows.push(["Views", fmtViews(d.views[p]) + " a month"]);
    const grid = rows.map(([k, v]) => `<div class="k">${k}</div><div>${esc(v)}</div>`).join("");
    let foot;
    if (!d) foot = "Loading details…";
    else if (d.from[p]) foot = `Translated by Claude from the <b>${esc(d.from[p])} Wikipedia</b> · click to open it`;
    else if (d.ref[p].startsWith("wd:")) foot = "From Wikidata · click to open the item";
    else foot = "Click to open the Wikipedia article";
    return `${img}${pill}<div class="pc-title">${esc(P.name[p])}</div>` +
      (d && d.desc[p] ? `<div class="pc-desc">${esc(d.desc[p])}</div>` : "") +
      `<div class="pc-grid">${grid}</div>` +
      `<div class="pc-lead">${d ? esc(d.excerpt[p]) : ""}</div><div class="pc-foot">${foot}</div>`;
  }

  let cardFor = -2, cardLoaded = null;
  function showCard() {
    const card = $("card");
    const p = S.hover >= 0 ? S.hover : S.pinned;
    if (p < 0) { card.hidden = true; cardFor = -2; return; }
    if (p !== cardFor || cardLoaded !== !!D) {
      card.innerHTML = cardHtml(p);
      cardFor = p;
      cardLoaded = !!D;
    }
    card.classList.toggle("pinned", S.hover < 0);
    card.hidden = false;
    let xy = S.hover >= 0 ? S.hoverXY : null;
    if (!xy) {
      const at = personAt(p, S.s);
      const vp = deckgl.getViewports()[0];
      if (!at || !vp) { card.hidden = true; return; }
      xy = vp.project([at.x, at.y]);
    }
    const [mw, mh] = mapSize();
    const cw = card.offsetWidth, ch = card.offsetHeight;
    let x = xy[0] + 18, y = xy[1] + 14;
    if (x + cw > mw - 8) x = xy[0] - cw - 18;
    if (y + ch > mh - 8) y = Math.max(8, mh - ch - 8);
    card.style.left = Math.max(8, x) + "px";
    card.style.top = Math.max(8, y) + "px";
  }

  // ------------------------------------------------------------------------------------------------------
  // Timeline (ThemeRiver of the coarse tracks) -- also the slider
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const TL = { w: 0, h: 0, padL: 14, padR: 14, top: 4, axis: 18, step: 2, g: {}, ribbons: {} };

  function binToX(b) { return TL.padL + (b / core.n_bins) * (TL.w - TL.padL - TL.padR); }
  function xToBin(x) { return ((x - TL.padL) / (TL.w - TL.padL - TL.padR)) * core.n_bins; }
  function yearToBin(y) {
    const by = core.bin_years;
    if (y < by[0] || y >= by[by.length - 1]) return null;
    let k = 0;
    while (by[k + 1] <= y) k++;
    return k + (y - by[k]) / (by[k + 1] - by[k]);
  }

  // Thickness per coarse track per window = its people in that window (stage 09's coarse layer).
  function trackCounts() {
    const L = core.labels, T = core.tracks;
    const ids = Object.keys(coarseOf).map(Number);
    const cnt = {};
    ids.forEach((t) => (cnt[t] = new Float64Array(NW)));
    L.window.forEach((w, i) => {
      const g = L.track[i];
      if (T.layer[g] === 2) cnt[T.id[g]][w] = L.n[i];
    });
    return cnt;
  }

  // Knots every half bin from 0 to n_bins: window w's value sits at its centre bin w + 2.5, the midpoint
  // between windows at w + 3 carries the mean (0 if the track is absent from either: ribbons taper there),
  // and the first and last windows extend flat to the chart's edges.
  function knotValues(c) {
    const K = 2 * core.n_bins + 1, v = new Float64Array(K), half = core.window_bins / 2;
    for (let k = 0; k < K; k++) {
      const x = k / 2;
      if (x <= half) v[k] = c[0];
      else if (x >= core.n_bins - half) v[k] = c[NW - 1];
      else {
        const d = x - half; // windows from the first centre
        if (Number.isInteger(d)) v[k] = c[d];
        else {
          const a = c[Math.floor(d)], b = c[Math.ceil(d)];
          v[k] = a > 0 && b > 0 ? (a + b) / 2 : 0;
        }
      }
    }
    return v;
  }

  function layoutRibbons() {
    const order = core.coarse.order.map((g) => core.tracks.id[g]);
    const cnt = (TL.counts = trackCounts());
    const vals = {};
    order.forEach((t) => (vals[t] = knotValues(cnt[t])));
    // A drift cut (stage 09) hands a region from one track straight to the next: at the midpoint between the
    // old track's last window and the new one's first, each carries half the band, so the river changes colour
    // there instead of pinching to nothing. (Cuts across a window the region is absent from still taper.)
    const T = core.tracks, half = core.window_bins / 2;
    T.from.forEach((f, g) => {
      if (f < 0 || T.layer[g] !== 2 || T.first[g] !== T.last[f] + 1) return;
      const t = T.id[f], s = T.id[g], d = T.last[f];
      const k = 2 * (d + half) + 1; // knot of the midpoint between windows d and d + 1
      const m = (cnt[t][d] + cnt[s][d + 1]) / 4;
      vals[t][k] = m;
      vals[s][k] = m;
    });
    const K = 2 * core.n_bins + 1;
    let maxSum = 0, maxAlive = 0;
    for (let k = 0; k < K; k++) {
      let s = 0, a = 0;
      for (const t of order) if (vals[t][k] > 0) { s += vals[t][k]; a++; }
      maxSum = Math.max(maxSum, s);
      maxAlive = Math.max(maxAlive, a);
    }
    const H = TL.h - TL.top - TL.axis;
    const scale = (H - maxAlive * GAP) / maxSum;
    // Control points (y down): centred stack, ribbons in order from the top.
    const top = {}, bot = {};
    order.forEach((t) => { top[t] = new Float64Array(K); bot[t] = new Float64Array(K); });
    for (let k = 0; k < K; k++) {
      let total = 0;
      for (const t of order) if (vals[t][k] > 0) total += vals[t][k] * scale + GAP;
      let y = TL.top + (H - total) / 2;
      for (const t of order) {
        const v = vals[t][k];
        if (v > 0) {
          top[t][k] = y + GAP / 2;
          bot[t][k] = y + GAP / 2 + v * scale;
          y += v * scale + GAP;
        } else top[t][k] = bot[t][k] = y;
      }
    }
    // Sample the uniform cubic B-spline of each boundary every TL.step px (C2-smooth, linear in the
    // control points, so stacked boundaries stay stacked and never overshoot below zero thickness).
    const xs = [];
    for (let x = TL.padL; x <= TL.w - TL.padR + 0.01; x += TL.step) xs.push(x);
    const spl = (P, u) => {
      const j = Math.floor(u), f = u - j;
      const at = (i) => P[clamp(i, 0, K - 1)];
      const f2 = f * f, f3 = f2 * f;
      return (at(j - 1) * (1 - 3 * f + 3 * f2 - f3) + at(j) * (4 - 6 * f2 + 3 * f3) +
        at(j + 1) * (1 + 3 * f + 3 * f2 - 3 * f3) + at(j + 2) * f3) / 6;
    };
    const rib = {};
    for (const t of order) {
      const T0 = new Float64Array(xs.length), B0 = new Float64Array(xs.length);
      xs.forEach((x, i) => {
        const u = clamp(xToBin(x), 0, core.n_bins) * 2;
        T0[i] = spl(top[t], u);
        B0[i] = spl(bot[t], u);
      });
      rib[t] = { top: T0, bot: B0 };
    }
    return { order, xs, rib };
  }

  let measureCtx = null;
  function textWidth(text, size, weight = 600) {
    if (!measureCtx) measureCtx = document.createElement("canvas").getContext("2d");
    measureCtx.font = `${weight} ${size}px ${FONT}`;
    return measureCtx.measureText(text).width;
  }

  // Sliding-window max / min over m consecutive samples (monotonic deque).
  function sliding(a, m, isMax) {
    const out = new Float64Array(Math.max(0, a.length - m + 1));
    const dq = [];
    for (let i = 0; i < a.length; i++) {
      while (dq.length && (isMax ? a[dq[dq.length - 1]] <= a[i] : a[dq[dq.length - 1]] >= a[i])) dq.pop();
      dq.push(i);
      if (dq[0] <= i - m) dq.shift();
      if (i >= m - 1) out[i - m + 1] = a[dq[0]];
    }
    return out;
  }

  // The best horizontal spot for `text` inside a ribbon: the span where the ribbon leaves the most room.
  function fitLabel(r, text, size) {
    const w = textWidth(text, size) + 8, h = size * 1.02 + 2;
    const m = Math.ceil(w / TL.step);
    if (m >= r.top.length) return null;
    const maxTop = sliding(r.top, m, true), minBot = sliding(r.bot, m, false);
    let best = -1, bi = -1;
    for (let i = 0; i < maxTop.length; i++) {
      const slack = minBot[i] - maxTop[i] - h;
      if (slack >= 0 && slack > best) { best = slack; bi = i; }
    }
    if (bi < 0) return null;
    return { x: TL.padL + (bi + m / 2) * TL.step, y: (maxTop[bi] + minBot[bi]) / 2, size };
  }

  function drawTimeline() {
    const svg = $("tl-svg");
    const rect = svg.getBoundingClientRect();
    TL.w = rect.width;
    TL.h = rect.height;
    if (TL.w < 80 || TL.h < 60) return false; // not laid out yet: the ResizeObserver draws it later
    svg.setAttribute("viewBox", `0 0 ${TL.w} ${TL.h}`);
    svg.innerHTML = "";
    const gR = el("g", {}, svg), gL = el("g", {}, svg), gA = el("g", {}, svg), gB = el("g", {}, svg);
    TL.g = { gR, gL, gA, gB };
    const { order, xs, rib } = layoutRibbons();
    TL.ribbons = {};
    for (const t of order) {
      const r = rib[t];
      let i0 = 0, i1 = xs.length - 1;
      while (i0 < i1 && r.bot[i0] - r.top[i0] < 0.05) i0++;
      while (i1 > i0 && r.bot[i1] - r.top[i1] < 0.05) i1--;
      let d = "M" + xs[i0].toFixed(1) + "," + r.top[i0].toFixed(2);
      for (let i = i0 + 1; i <= i1; i++) d += "L" + xs[i].toFixed(1) + "," + r.top[i].toFixed(2);
      for (let i = i1; i >= i0; i--) d += "L" + xs[i].toFixed(1) + "," + r.bot[i].toFixed(2);
      const g = coarseOf[t];
      const path = el("path", { d: d + "Z", class: "ribbon", fill: `rgb(${coarseRgb[t].join(",")})` }, gR);
      path.addEventListener("pointerenter", (e) => { if (!S.dragging) hoverRibbon(t, e); });
      path.addEventListener("pointermove", (e) => { if (!S.dragging) moveTip(e); });
      path.addEventListener("pointerleave", () => { if (!S.dragging) hoverRibbon(-1); });
      TL.ribbons[t] = path;
      // Label: full name if it fits somewhere at 13..9 px, else the short name, else none.
      const T = core.tracks;
      let spot = null, text = null;
      for (const cand of [T.name[g], T.short[g]]) {
        if (!cand) continue;
        for (const size of [13, 12, 11, 10, 9]) {
          spot = fitLabel(r, cand, size);
          if (spot) { text = cand; break; }
        }
        if (spot) break;
      }
      if (spot) {
        const tx = el("text", { x: spot.x.toFixed(1), y: spot.y.toFixed(1), class: "rlabel", "font-size": spot.size }, gL);
        tx.textContent = text;
      }
    }
    // Year axis: ticks at their true positions along the equal-people bins.
    let lastX = -1e9;
    const yAxis = TL.h - TL.axis;
    for (const yr of TICK_YEARS) {
      const b = yearToBin(yr);
      if (b === null) continue;
      const x = binToX(b);
      if (x - lastX < 46) continue;
      lastX = x;
      const g = el("g", { class: "tick" }, gA);
      el("line", { x1: x, x2: x, y1: yAxis, y2: yAxis + 4 }, g);
      el("text", { x, y: yAxis + 14 }, g).textContent = fmtYear(yr);
    }
    // Lens: shade outside the 5 bins on the map, outline the band.
    TL.shadeL = el("rect", { class: "shade", y: 0, height: yAxis }, gB);
    TL.shadeR = el("rect", { class: "shade", y: 0, height: yAxis }, gB);
    TL.band = el("rect", { class: "band", y: 1, height: yAxis - 2, rx: 3 }, gB);
    TL.grip = el("rect", { class: "grip", y: 0, height: 3, rx: 1.5 }, gB);
    drawBand();
    return true;
  }

  function drawBand() {
    if (!TL.band) return;
    const x0 = binToX(S.s), x1 = binToX(S.s + core.window_bins);
    const yAxis = TL.h - TL.axis;
    TL.shadeL.setAttribute("x", 0);
    TL.shadeL.setAttribute("width", Math.max(0, x0));
    TL.shadeR.setAttribute("x", x1);
    TL.shadeR.setAttribute("width", Math.max(0, TL.w - x1));
    TL.band.setAttribute("x", x0);
    TL.band.setAttribute("width", Math.max(0, x1 - x0));
    TL.band.setAttribute("height", yAxis - 2);
    TL.grip.setAttribute("x", (x0 + x1) / 2 - 14);
    TL.grip.setAttribute("width", 28);
  }

  function hoverRibbon(t, e) {
    S.hlTrack = t;
    $("tl-svg").classList.toggle("has-hl", t >= 0);
    for (const k in TL.ribbons) TL.ribbons[k].classList.toggle("hl", Number(k) === t);
    const tip = $("tl-tip");
    if (t < 0) { tip.hidden = true; queueRender(); return; }
    const g = coarseOf[t], T = core.tracks, W = core.windows;
    const peak = Math.max(...TL.counts[t]);
    tip.innerHTML = `<b>${esc(T.name[g])}</b><br><span class="d">In the windows of people born ` +
      `${esc(fmtYear(W.lo[T.first[g]]))} – ${esc(fmtYear(W.hi[T.last[g]]))} · up to ` +
      `${peak.toLocaleString("en-US")} people</span>`;
    tip.hidden = false;
    moveTip(e);
    queueRender();
  }
  function moveTip(e) {
    const tip = $("tl-tip");
    const box = $("timeline").getBoundingClientRect();
    let x = e.clientX - box.left + 14, y = e.clientY - box.top - tip.offsetHeight - 10;
    if (x + tip.offsetWidth > box.width - 8) x = e.clientX - box.left - tip.offsetWidth - 14;
    tip.style.left = Math.max(4, x) + "px";
    tip.style.top = Math.max(-tip.offsetHeight - 6, y) + "px";
  }
  function highlightRibbonOf(p) {
    if (S.hlTrack >= 0) return; // a ribbon hovered in the chart wins
    for (const k in TL.ribbons) TL.ribbons[k].classList.remove("hl");
    const at = p >= 0 ? personAt(p, S.s) : null;
    const t = at && at.track !== 255 ? at.track : -1;
    $("tl-svg").classList.toggle("has-hl", t >= 0);
    if (t >= 0 && TL.ribbons[t]) TL.ribbons[t].classList.add("hl");
  }

  function initTimeline() {
    const svg = $("tl-svg");
    drawTimeline();
    let offset = 0;
    svg.addEventListener("pointerdown", (e) => {
      const b = xToBin(e.clientX - svg.getBoundingClientRect().left);
      stopAnim();
      offset = b >= S.s && b <= S.s + core.window_bins ? b - S.s : core.window_bins / 2;
      S.dragging = true;
      $("tl-tip").hidden = true;
      svg.setPointerCapture(e.pointerId);
      setS(b - offset);
    });
    svg.addEventListener("pointermove", (e) => {
      if (!S.dragging) return;
      setS(xToBin(e.clientX - svg.getBoundingClientRect().left) - offset);
    });
    const release = () => {
      if (!S.dragging) return;
      S.dragging = false;
      animateTo(Math.round(S.s), SETTLE_MS);
    };
    svg.addEventListener("pointerup", release);
    svg.addEventListener("pointercancel", release);
    // Redraw on any size change, including the first real one (a pane can load the page at zero size).
    let pending = null, lastSize = "";
    new ResizeObserver(() => {
      const r = svg.getBoundingClientRect();
      const size = `${Math.round(r.width)}x${Math.round(r.height)}`;
      if (size === lastSize) return;
      lastSize = size;
      clearTimeout(pending);
      pending = setTimeout(() => {
        if (!S.fitted) refit();
        drawTimeline();
        queueRender();
      }, 80);
    }).observe(svg);
  }

  // ------------------------------------------------------------------------------------------------------
  // Search: names (and short descriptions once loaded) across all eras. Picking a result moves the slider
  // to the person's middle window and flies to them; matches in view are highlighted while typing.
  let results = [], active = -1;
  function initSearch() {
    const input = $("search");
    input.addEventListener("input", () => runSearch(input.value));
    input.addEventListener("focus", () => { if (input.value.trim().length >= 2) runSearch(input.value); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!results.length) return;
        active = clamp(active + (e.key === "ArrowDown" ? 1 : -1), 0, results.length - 1);
        renderResults();
      } else if (e.key === "Enter") {
        const p = results[active >= 0 ? active : 0];
        if (p !== undefined) goTo(p);
      } else if (e.key === "Escape") {
        clearSearch();
        input.blur();
      }
    });
  }

  function runSearch(raw) {
    const q = normText(raw.trim());
    if (q.length < 2) { clearSearch(false); return; }
    const toks = q.split(/\s+/);
    const mask = new Uint8Array(NP);
    const scored = [];
    for (let p = 0; p < NP; p++) {
      const name = nameIndex[p];
      let score = -1;
      // Tiers, each in fame order: every token starts a word of the name; every token is in the name;
      // every token is in the name or short description.
      const spaced = " " + name;
      if (toks.every((t) => spaced.includes(" " + t))) score = 0;
      else if (toks.every((t) => name.includes(t))) score = 1;
      else if (descIndex) {
        const both = name + " " + descIndex[p];
        if (toks.every((t) => both.includes(t))) score = 2;
      }
      if (score >= 0) { mask[p] = 1; scored.push(score * 1e6 + p); }
    }
    scored.sort((a, b) => a - b);
    results = scored.slice(0, 12).map((v) => v % 1e6);
    active = -1;
    S.mask = mask;
    S.nMatches = scored.length;
    searchStatus();
    renderResults();
    queueRender();
  }

  function searchStatus() {
    if (!S.mask) { $("search-status").textContent = ""; return; }
    let inView = 0;
    for (const p of members[Math.round(S.s)]) if (S.mask[p]) inView++;
    const n = S.nMatches;
    $("search-status").textContent = `${n.toLocaleString("en-US")} match${n === 1 ? "" : "es"} · ` +
      `${inView.toLocaleString("en-US")} in this window` + (descIndex ? "" : " (names only until details load)");
  }

  function renderResults() {
    const ol = $("results");
    const P = core.people;
    ol.innerHTML = results.map((p, i) =>
      `<li data-p="${p}" class="${i === active ? "active" : ""}"><span class="r-name">${esc(P.name[p])}</span>` +
      `<span class="r-year">${esc(P.born[p])}</span>` +
      (D && D.desc[p] ? `<span class="r-desc">${esc(D.desc[p])}</span>` : "") + "</li>").join("");
    ol.hidden = results.length === 0;
    ol.querySelectorAll("li").forEach((li) => li.addEventListener("click", () => goTo(Number(li.dataset.p))));
  }

  function clearSearch(clearInput = true) {
    if (clearInput) $("search").value = "";
    results = [];
    S.mask = null;
    $("search-status").textContent = "";
    $("results").hidden = true;
    queueRender();
  }

  function goTo(p) {
    const target = W0[p] + Math.floor((NWIN[p] - 1) / 2);
    stopAnim();
    S.pinned = p;
    S.mask = null; // the pinned card and ring mark the person; the query text stays for another pick
    $("search-status").textContent = "";
    $("results").hidden = true;
    const ms = Math.min(1600, 350 + 70 * Math.abs(target - S.s));
    animateTo(target, ms, () => {
      const at = personAt(p, target);
      deckgl.setProps({
        initialViewState: {
          target: [at.x, at.y, 0],
          zoom: Math.max(S.zoom, S.zoom0 + 2.2),
          minZoom: S.zoom0 - 1.5,
          maxZoom: S.zoom0 + 7,
          transitionDuration: 700,
          transitionInterpolator: new deck.LinearInterpolator({ transitionProps: ["target", "zoom"] }),
        },
      });
      setTimeout(() => { showCard(); queueRender(); }, 720);
    });
    showCard();
  }

  main().catch((err) => {
    $("loading").textContent = "Could not load the map data: " + err.message;
    console.error(err);
  });
})();
