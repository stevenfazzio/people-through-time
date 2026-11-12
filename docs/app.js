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
  const AGED_SPAN_MAX = 150; // years: the readout's "aged 40 in" line shows for windows no wider (1773-1914 on)
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
  const TICK_SIZE = 11; // px, year ticks on the timeline's axis (as in style.css)
  const EDGE_SIZE = 13; // px, the band's edge years (as in style.css)
  const EDGE_GAP = 8; // px kept clear between an edge year and a tick or the other edge year
  const EDGE_INSET = 8; // px kept between an edge year and the chart's sides (2 px clipped "2013" in one browser)
  const HANDLE_W = 46, HANDLE_H = 12; // px, the tab on top of the band; it grabs across the band's whole width
  const LABEL_PAD = [10, 6]; // px added around each label's box (x, y) when testing overlaps
  const LABEL_FADE_MS = 180;
  // Props handed to every label layer on every frame: created once, so deck.gl sees them unchanged.
  let LABEL_PROPS = null;
  const TICK_YEARS = [-2999, -999, -499, 1, 500, 1000, 1200, 1400, 1500, 1600, 1700, 1800, 1850, 1900, 1925,
    1950, 1970, 1990, 2005];

  // Phone-sized screens dock the card (as in style.css): along the bottom in portrait, at the right when short.
  const DOCKED = window.matchMedia("(max-width: 640px), (max-height: 500px)");
  const NO_HOVER = window.matchMedia("(hover: none)");

  const $ = (id) => document.getElementById(id);
  // Touch and pen have no hover: a tap pins the card instead of opening the article. Per event, not per
  // device, so a touchscreen laptop's mouse still opens articles on click.
  const isTouch = (ev) => !!ev && (ev.pointerType === "touch" || ev.pointerType === "pen");
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
    hoverHold: null, // [x, y] where the last drag was released: ribbon hovers wait until the pointer moves
    mask: null, // Uint8Array of search matches, or null
    zoom: 0,
    zoom0: 0,
    userMoved: false, // the reader panned, zoomed or flew somewhere: a resize no longer refits the view
    lastTouch: -1e9, // performance.now() of the last touch on the map
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
    initCard();
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

  let qs = null;
  function fitZoom() {
    // Fit the central 99.6% of all points (normalized windows share one frame).
    if (!qs) {
      const xs = Array.from(X).sort((a, b) => a - b), ys = Array.from(Y).sort((a, b) => a - b);
      const qa = (a, f) => a[Math.floor(f * (a.length - 1))];
      qs = [qa(xs, 0.002), qa(xs, 0.998), qa(ys, 0.002), qa(ys, 0.998)];
    }
    const [x0, x1, y0, y1] = qs;
    const [w, h] = mapSize();
    // On a phone the title panel spans the map's width: fit the points into the space below it instead.
    const panel = $("title").getBoundingClientRect();
    const top = panel.width > 0.6 * w && panel.bottom < 0.5 * h ? panel.bottom : 0;
    const zoom = Math.log2(Math.min(w / (x1 - x0), (h - top) / (y1 - y0)) * 0.9);
    return { target: [(x0 + x1) / 2, (y0 + y1) / 2 - top / 2 / Math.pow(2, zoom), 0], zoom };
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

  // Move the view (zoom unchanged unless given) so world point [x, y] lands on screen point [fx, fy].
  function panTo(x, y, [fx, fy], zoom = S.zoom, ms = 450) {
    const [mw, mh] = mapSize();
    const k = Math.pow(2, zoom);
    S.userMoved = true;
    deckgl.setProps({
      initialViewState: {
        target: [x + (mw / 2 - fx) / k, y + (mh / 2 - fy) / k, 0],
        zoom,
        minZoom: S.zoom0 - 1.5,
        maxZoom: S.zoom0 + 7,
        transitionDuration: ms,
        transitionInterpolator: new deck.LinearInterpolator({ transitionProps: ["target", "zoom"] }),
      },
    });
  }

  // Where a focused person should sit on screen: the middle of the map's part not under the title panel or
  // a docked card (the map's centre when the card floats beside its person).
  function focusPoint() {
    const [mw, mh] = mapSize();
    let x0 = 0, x1 = mw, y0 = 0, y1 = mh;
    const card = $("card");
    if (DOCKED.matches && !card.hidden) {
      const r = card.getBoundingClientRect();
      if (r.left > 0.3 * mw) x1 = r.left; // docked at the right
      else y1 = r.top; // docked along the bottom
    }
    const panel = $("title").getBoundingClientRect();
    if (panel.width > 0.6 * (x1 - x0) && y1 - panel.bottom > 80) y0 = panel.bottom;
    return [(x0 + x1) / 2, (y0 + y1) / 2];
  }

  // A tapped person under the title panel or the docked card moves into the free part of the map.
  function revealPinned() {
    const at = personAt(S.pinned, S.s);
    const vp = deckgl.getViewports()[0];
    if (!at || !vp) return;
    const [sx, sy] = vp.project([at.x, at.y]);
    const M = 16;
    const covered = ["title", "card"].some((id) => {
      const el = $(id);
      if (el.hidden) return false;
      const r = el.getBoundingClientRect();
      return sx > r.left - M && sx < r.right + M && sy > r.top - M && sy < r.bottom + M;
    });
    if (covered) panTo(at.x, at.y, focusPoint());
  }

  function pin(p) {
    S.pinned = p;
    showCard();
    queueRender();
  }
  function unpin() {
    if (S.pinned < 0) return;
    S.pinned = -1;
    showCard();
    queueRender();
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
      onViewStateChange: ({ viewState, interactionState }) => {
        S.zoom = viewState.zoom;
        if (interactionState && (interactionState.isDragging || interactionState.isPanning ||
          interactionState.isZooming)) S.userMoved = true;
        queueRender();
      },
      onHover: (info, ev) => {
        // A touch has no hover; its tap pins (onClick). deck.gl's hover comes from mouse events, labelled
        // "mouse" whatever made them, so a tap's compatibility mouse events (if touchend's cancel below
        // didn't stop them) are told apart by time.
        if (isTouch(ev) || performance.now() - S.lastTouch < 1000) return;
        const p = info.picked && info.layer && info.layer.id === "points" ? frame.pid[info.index] : -1;
        S.hoverXY = info.picked ? [info.x, info.y] : null;
        if (p !== S.hover) {
          S.hover = p;
          queueRender();
        }
        showCard();
      },
      onClick: (info, ev) => {
        // deck.gl also calls onClick for a double click, right after its first click (a second article tab).
        if (ev && ev.type === "dblclick") return;
        const p = info.picked && info.layer && info.layer.id === "points" ? frame.pid[info.index] : -1;
        if (p >= 0 && isTouch(ev)) {
          // A tap shows the card (pinned, with a link to the article); tapping empty map closes it.
          pin(p);
          revealPinned();
        } else if (p >= 0) {
          const ref = D ? D.ref[p] : null;
          if (ref) window.open(articleUrl(ref), "_blank", "noopener");
        } else unpin();
      },
      layers: [],
    });
    // deck.gl fires onClick on a tap's pointerup (in 9.1.15 its click recognizer never waits for a double
    // tap: it passes requestFailure where mjolnir reads requireFailure), so the pinned card is up before the
    // browser's own click for the tap -- which then lands on whatever is now under the finger: the card, or
    // its link to Wikipedia. Cancelling touchend stops that click and the tap's compatibility mouse events
    // (which deck.gl would take for a mouse hover); deck.gl reads pointer events, so taps, pans and pinches
    // still work.
    const mapEl = $("map");
    mapEl.addEventListener("pointerdown", (e) => { if (isTouch(e)) S.lastTouch = performance.now(); }, true);
    mapEl.addEventListener("touchend", (e) => {
      S.lastTouch = performance.now();
      if (e.cancelable) e.preventDefault();
      if (document.activeElement === $("search")) $("search").blur(); // a cancelled tap moves no focus
    }, { passive: false });
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
    // The title panel and a docked card count as placed labels, so no label hides under them.
    const [mw, mh] = mapSize();
    const placed = [];
    for (const id of ["title", "card"]) {
      const el = $(id);
      if (el.hidden || (id === "card" && !el.classList.contains("docked"))) continue;
      const r = el.getBoundingClientRect();
      if (r.width) placed.push([r.left - 4, r.right + 4, r.top - 4, r.bottom + 4]);
    }
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
    markRibbon();
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

  // The range on the map. "Aged 40 in" moves the reader from birth years to the years these people were
  // active (the slider is about birth, not events); only where 40 years is a real shift against the window's
  // span -- in the earliest windows, spanning centuries, it would be noise.
  function updateReadout() {
    const w = Math.round(S.s);
    if (w === S.readoutWindow) return;
    S.readoutWindow = w;
    const W = core.windows;
    const range = `${fmtYear(W.lo[w])} – ${fmtYear(W.hi[w])}`;
    $("era-years").textContent = range;
    let sub = `${W.n[w].toLocaleString("en-US")} people`;
    if (W.hi[w] - W.lo[w] <= AGED_SPAN_MAX) sub += ` · aged 40 in ${fmtYear(W.lo[w] + 40)} – ${fmtYear(W.hi[w] + 40)}`;
    $("era-sub").textContent = sub;
    $("tl-svg").setAttribute("aria-valuenow", String(w));
    $("tl-svg").setAttribute("aria-valuetext", "Born " + range);
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
        clearSearch(); // also unpins
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
  function cardTrack(p) {
    const at = personAt(p, S.s);
    return at ? at.track : 255;
  }

  // pinned: the card stays up (a search pick or a tap) and takes clicks, so it gets a close button and a link
  // to the article in place of the hover card's "click to open".
  function cardHtml(p, tr, pinned) {
    const P = core.people;
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
    const link = (text) => `<a class="pc-link" href="${esc(articleUrl(d.ref[p]))}" target="_blank" rel="noopener">${text} →</a>`;
    if (!d) foot = "Loading details…";
    else if (d.from[p]) {
      foot = `Translated by Claude from the <b>${esc(d.from[p])} Wikipedia</b>` +
        (pinned ? `<br>${link(`Open the ${esc(d.from[p])} article`)}` : " · click to open it");
    } else if (d.ref[p].startsWith("wd:")) foot = pinned ? `From Wikidata · ${link("Open the item")}` : "From Wikidata · click to open the item";
    else foot = pinned ? link("Open the Wikipedia article") : "Click to open the Wikipedia article";
    const close = pinned ? `<button class="pc-close" type="button" aria-label="Close">×</button>` : "";
    return `${close}${img}${pill}<div class="pc-title">${esc(P.name[p])}</div>` +
      (d && d.desc[p] ? `<div class="pc-desc">${esc(d.desc[p])}</div>` : "") +
      `<div class="pc-grid">${grid}</div>` +
      `<div class="pc-lead">${d ? esc(d.excerpt[p]) : ""}</div><div class="pc-foot">${foot}</div>`;
  }

  // The card is rebuilt when its person, their region (the pill follows the slider), the details or its
  // pinned state change. On a phone-sized screen it docks (style.css places it) instead of following its person.
  let cardFor = -2, cardLoaded = null, cardTr = -1, cardPinned = null;
  function showCard() {
    const card = $("card");
    const p = S.hover >= 0 ? S.hover : S.pinned;
    if (p < 0) { card.hidden = true; cardFor = -2; return; }
    const tr = cardTrack(p), pinned = S.hover < 0;
    if (p !== cardFor || cardLoaded !== !!D || tr !== cardTr || pinned !== cardPinned) {
      card.innerHTML = cardHtml(p, tr, pinned);
      cardFor = p;
      cardLoaded = !!D;
      cardTr = tr;
      cardPinned = pinned;
    }
    card.classList.toggle("pinned", pinned);
    const docked = DOCKED.matches;
    card.classList.toggle("docked", docked);
    card.hidden = false;
    let xy = S.hover >= 0 ? S.hoverXY : null;
    if (!xy) {
      const at = personAt(p, S.s);
      const vp = deckgl.getViewports()[0];
      if (!at || !vp) { card.hidden = true; return; }
      xy = vp.project([at.x, at.y]);
    }
    if (docked) {
      card.style.left = card.style.top = "";
      return;
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
  // top: the strip above the ribbons holds the band's handle (HANDLE_H tall), so grabbing it touches no ribbon.
  const TL = { w: 0, h: 0, padL: 14, padR: 14, top: HANDLE_H + 4, axis: 18, step: 2, g: {}, ribbons: {} };

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
    // Each knot's ribbons fill the chart's height: a ribbon is its region's share of the window's people in a
    // coarse region. Windows hold ~equal numbers of people by design, so a raw total would only show coarse
    // coverage (58-92% by window), a clustering artifact that reads as "fewer famous people born then".
    const K = 2 * core.n_bins + 1;
    const H = TL.h - TL.top - TL.axis;
    const top = {}, bot = {};
    order.forEach((t) => { top[t] = new Float64Array(K); bot[t] = new Float64Array(K); });
    for (let k = 0; k < K; k++) {
      let sum = 0, alive = 0;
      for (const t of order) if (vals[t][k] > 0) { sum += vals[t][k]; alive++; }
      const scale = sum > 0 ? (H - alive * GAP) / sum : 0;
      let y = TL.top;
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

  // Where no horizontal box fits -- mostly a ribbon running diagonally, whose vertical thickness a flat box
  // can't use -- the label follows the ribbon's centre line instead, as a map labels a river (fitCurved). The
  // text is a strip h tall centred on that line: at every sample of the stretch both edges of the strip, taken
  // along the line's normal, must lie inside the ribbon, the line within CURVE_MAX_ANGLE of horizontal and
  // bending no tighter than CURVE_MIN_RADIUS text heights (tighter, the letters on the inside of the bend
  // crowd). Of the stretches that fit, the one leaving the most room at its tightest sample wins.
  const CURVE_MAX_ANGLE = (35 * Math.PI) / 180;
  const CURVE_MIN_RADIUS = 2.5;

  function centreLine(r) {
    const n = r.top.length, st = TL.step;
    const c = new Float64Array(n), ang = new Float64Array(n), radius = new Float64Array(n), S = new Float64Array(n);
    for (let i = 0; i < n; i++) c[i] = (r.top[i] + r.bot[i]) / 2;
    for (let i = 0; i < n; i++) {
      const a = c[Math.max(0, i - 1)], b = c[Math.min(n - 1, i + 1)];
      const s = (b - a) / (st * (Math.min(n - 1, i + 1) - Math.max(0, i - 1)));
      const s2 = i > 0 && i < n - 1 ? (c[i + 1] - 2 * c[i] + c[i - 1]) / (st * st) : 0;
      ang[i] = Math.atan(s);
      radius[i] = s2 === 0 ? Infinity : Math.pow(1 + s * s, 1.5) / Math.abs(s2);
      if (i > 0) S[i] = S[i - 1] + Math.hypot(st, c[i] - c[i - 1]);
    }
    return { c, ang, radius, S };
  }

  function fitCurved(r, text, size) {
    const w = textWidth(text, size) + 8, h = size * 1.02 + 2;
    const n = r.top.length, st = TL.step;
    const L = r.line || (r.line = centreLine(r));
    const at = (arr, x) => {
      const u = clamp((x - TL.padL) / st, 0, n - 1), i = Math.min(Math.floor(u), n - 2), f = u - i;
      return arr[i] * (1 - f) + arr[i + 1] * f;
    };
    // Room at each sample: vertical px between each strip edge and the ribbon's boundary beyond it (< 0 outside).
    const room = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      if (Math.abs(L.ang[i]) > CURVE_MAX_ANGLE || L.radius[i] < CURVE_MIN_RADIUS * h) { room[i] = -Infinity; continue; }
      const x = TL.padL + i * st, sn = Math.sin(L.ang[i]) * h / 2, cs = Math.cos(L.ang[i]) * h / 2;
      const up = L.c[i] - cs - at(r.top, x + sn); // upper edge of the strip: (x + sin, c - cos) * h / 2
      const down = at(r.bot, x - sn) - (L.c[i] + cs); // lower edge: (x - sin, c + cos) * h / 2
      room[i] = Math.min(up, down);
    }
    // Every stretch i..j whose arc length first reaches w, with the least room in it (monotonic deque).
    let best = -1, bi = -1, bj = -1, jn = 0;
    const dq = [];
    for (let i = 0; i < n; i++) {
      while (jn < n && (jn <= i || L.S[jn - 1] - L.S[i] < w)) {
        while (dq.length && room[dq[dq.length - 1]] >= room[jn]) dq.pop();
        dq.push(jn++);
      }
      while (dq.length && dq[0] < i) dq.shift();
      if (L.S[jn - 1] - L.S[i] < w) break;
      const m = room[dq[0]];
      if (m >= 0 && m > best) { best = m; bi = i; bj = jn - 1; }
    }
    if (bi < 0) return null;
    let d = "";
    for (let i = bi; i <= bj; i++) {
      d += (i === bi ? "M" : "L") + (TL.padL + i * st).toFixed(1) + "," + L.c[i].toFixed(2);
    }
    return { d, size };
  }

  function drawTimeline() {
    const svg = $("tl-svg");
    const rect = svg.getBoundingClientRect();
    TL.w = rect.width;
    TL.h = rect.height;
    if (TL.w < 80 || TL.h < 60) return false; // not laid out yet: the ResizeObserver draws it later
    svg.setAttribute("viewBox", `0 0 ${TL.w} ${TL.h}`);
    svg.innerHTML = "";
    TL.marked = undefined;
    const defs = el("defs", {}, svg);
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
      // Touch has no hover: a press on the chart only moves the band.
      path.addEventListener("pointerenter", (e) => { if (!S.dragging && !S.hoverHold && !isTouch(e)) hoverRibbon(t, e); });
      path.addEventListener("pointermove", (e) => {
        if (S.dragging || S.hoverHold || isTouch(e)) return;
        if (S.hlTrack !== t) hoverRibbon(t, e); // first move after a drag ended over this ribbon
        else moveTip(e);
      });
      path.addEventListener("pointerleave", () => { if (!S.dragging) hoverRibbon(-1); });
      TL.ribbons[t] = path;
      // Label: full name if a horizontal box fits somewhere at 13..9 px, else the short name; where neither
      // fits flat, the same along the ribbon's centre line; else none.
      const T = core.tracks;
      let spot = null, text = null;
      for (const fit of [fitLabel, fitCurved]) {
        for (const cand of [T.name[g], T.short[g]]) {
          if (!cand) continue;
          for (const size of [13, 12, 11, 10, 9]) {
            spot = fit(r, cand, size);
            if (spot) { text = cand; break; }
          }
          if (spot) break;
        }
        if (spot) break;
      }
      if (spot && spot.d) {
        const id = `tl-rl-${t}`;
        el("path", { id, d: spot.d }, defs);
        const tx = el("text", { class: "rlabel", "font-size": spot.size }, gL);
        el("textPath", { href: `#${id}`, startOffset: "50%" }, tx).textContent = text;
      } else if (spot) {
        const tx = el("text", { x: spot.x.toFixed(1), y: spot.y.toFixed(1), class: "rlabel", "font-size": spot.size }, gL);
        tx.textContent = text;
      }
    }
    // Year axis: ticks at their true positions along the equal-people bins.
    let lastX = -1e9;
    const yAxis = TL.h - TL.axis;
    TL.ticks = [];
    for (const yr of TICK_YEARS) {
      const b = yearToBin(yr);
      if (b === null) continue;
      const x = binToX(b);
      if (x - lastX < 46) continue;
      lastX = x;
      const g = el("g", { class: "tick" }, gA);
      el("line", { x1: x, x2: x, y1: yAxis, y2: yAxis + 4 }, g);
      const text = fmtYear(yr);
      el("text", { x, y: yAxis + 14 }, g).textContent = text;
      TL.ticks.push({ g, x, half: textWidth(text, TICK_SIZE, 400) / 2 });
    }
    // Lens: shade outside the 5 bins on the map, outline the band, and label its edges with the window's
    // first and last birth years (the ticks under those labels hide).
    TL.shadeL = el("rect", { class: "shade", y: 0, height: yAxis }, gB);
    TL.shadeR = el("rect", { class: "shade", y: 0, height: yAxis }, gB);
    const bandY = TL.top - 3;
    TL.band = el("rect", { class: "band", y: bandY, height: yAxis - 1 - bandY, rx: 3 }, gB);
    TL.edges = [0, 1].map(() => ({
      line: el("line", { class: "edge-tick", y1: yAxis - 1, y2: yAxis + 4 }, gB),
      text: el("text", { class: "edge", y: yAxis + 14 }, gB),
      label: null,
    }));
    // Handle: a tab on the band's top edge (drawn round x = 0, moved by drawBand), with a hit area across the
    // band's width above the ribbons. Pressing it starts the same drag as pressing the band.
    TL.handle = el("g", { class: "handle" }, gB);
    el("rect", { class: "tab", x: -HANDLE_W / 2, y: 1, width: HANDLE_W, height: bandY + 2, rx: 3 }, TL.handle);
    for (const dx of [-4, 0, 4]) {
      el("line", { class: "grip-line", x1: dx, x2: dx, y1: 4, y2: bandY - 2 }, TL.handle);
    }
    TL.handleHit = el("rect", { class: "hit", y: 0, height: TL.top }, TL.handle);
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
    const cx = (x0 + x1) / 2, hitW = Math.max(HANDLE_W + 20, x1 - x0);
    TL.handle.setAttribute("transform", `translate(${cx.toFixed(1)},0)`);
    TL.handleHit.setAttribute("x", -hitW / 2);
    TL.handleHit.setAttribute("width", hitW);
    // Edge labels: the years of the window the map shows (the nearest one while dragging, as the readout),
    // centred under the band's edges, kept inside the chart and apart from each other.
    const w = Math.round(S.s), W = core.windows;
    const texts = [fmtYear(W.lo[w]), fmtYear(W.hi[w])];
    const half = texts.map((t) => textWidth(t, EDGE_SIZE, 700) / 2);
    let c0 = x0, c1 = x1;
    const short = c0 + half[0] + EDGE_GAP - (c1 - half[1]);
    if (short > 0) { c0 -= short / 2; c1 += short / 2; }
    const inL = Math.max(0, EDGE_INSET - (c0 - half[0])), inR = Math.max(0, c1 + half[1] - (TL.w - EDGE_INSET));
    c0 += inL - inR;
    c1 += inL - inR;
    [x0, x1].forEach((x, i) => {
      const e = TL.edges[i], c = i ? c1 : c0;
      e.line.setAttribute("x1", x);
      e.line.setAttribute("x2", x);
      e.text.setAttribute("x", c.toFixed(1));
      if (e.label !== texts[i]) { e.label = texts[i]; e.text.textContent = texts[i]; }
    });
    const spans = [[c0 - half[0], c0 + half[0]], [c1 - half[1], c1 + half[1]]];
    for (const t of TL.ticks) {
      const hide = spans.some(([a, b]) => t.x + t.half + EDGE_GAP > a && t.x - t.half - EDGE_GAP < b);
      if (t.hidden !== hide) { t.hidden = hide; t.g.style.display = hide ? "none" : ""; }
    }
  }

  function hoverRibbon(t, e) {
    S.hlTrack = t;
    TL.marked = undefined; // markRibbon redoes the person's ribbon once the chart hover ends
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
  // The hovered or pinned person's ribbon stands out in the chart (on a phone, where few ribbons have room
  // for a label, this is how a pinned person's region shows its run through time).
  function markRibbon() {
    if (S.hlTrack >= 0) return; // a ribbon hovered in the chart wins
    const p = S.hover >= 0 ? S.hover : S.pinned;
    const at = p >= 0 ? personAt(p, S.s) : null;
    const t = at && at.track !== 255 ? at.track : -1;
    if (t === TL.marked) return;
    TL.marked = t;
    for (const k in TL.ribbons) TL.ribbons[k].classList.toggle("hl", Number(k) === t);
    $("tl-svg").classList.toggle("has-hl", t >= 0);
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
      svg.classList.add("dragging");
      // Grabbing the band means pressing on a ribbon, which hover has just highlighted: drop that highlight
      // for the drag (ribbon hovers are ignored while dragging, and resume when the pointer next moves).
      hoverRibbon(-1);
      svg.setPointerCapture(e.pointerId);
      setS(b - offset);
    });
    svg.addEventListener("pointermove", (e) => {
      if (!S.dragging) return;
      setS(xToBin(e.clientX - svg.getBoundingClientRect().left) - offset);
    });
    // Releasing the capture makes the browser enter the ribbon under a still pointer; hold hovers off until
    // the pointer really moves (a capture listener, so the ribbon's own handler sees the hold cleared).
    svg.addEventListener("pointermove", (e) => {
      const h = S.hoverHold;
      if (h && Math.abs(e.clientX - h[0]) + Math.abs(e.clientY - h[1]) > 3) S.hoverHold = null;
    }, true);
    svg.addEventListener("pointerleave", () => { S.hoverHold = null; });
    const release = (e) => {
      if (!S.dragging) return;
      S.dragging = false;
      S.hoverHold = [e.clientX, e.clientY];
      svg.classList.remove("dragging");
      animateTo(Math.round(S.s), SETTLE_MS);
    };
    svg.addEventListener("pointerup", release);
    svg.addEventListener("pointercancel", release);
    // Redraw on any size change, including the first real one (a pane can load the page at zero size). A new
    // width (a phone turned sideways) refits the map too, unless the reader has moved it; a new height alone
    // may be a phone's keyboard, which shouldn't rescale the map.
    let pending = null, lastSize = "", lastW = 0;
    new ResizeObserver(() => {
      const r = svg.getBoundingClientRect();
      const size = `${Math.round(r.width)}x${Math.round(r.height)}`;
      if (size === lastSize) return;
      lastSize = size;
      clearTimeout(pending);
      pending = setTimeout(() => {
        const w = Math.round(svg.getBoundingClientRect().width);
        if (!S.fitted || (w !== lastW && !S.userMoved)) refit();
        lastW = w;
        drawTimeline();
        if (S.hover >= 0 || S.pinned >= 0) showCard(); // docked or not may have changed
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
    const input = $("search");
    if (clearInput) input.value = "";
    results = [];
    S.mask = null;
    $("search-status").textContent = "";
    $("results").hidden = true;
    // An empty box (its x, Escape, or deleting the query) drops the pin too, whether a pick or a tap made it.
    if (!input.value) unpin();
    queueRender();
  }

  function goTo(p) {
    const target = W0[p] + Math.floor((NWIN[p] - 1) / 2);
    stopAnim();
    S.pinned = p;
    S.mask = null; // the pinned card and ring mark the person; the query text stays for another pick
    $("search-status").textContent = "";
    $("results").hidden = true;
    if (NO_HOVER.matches) $("search").blur(); // put the phone's keyboard away
    const ms = Math.min(1600, 350 + 70 * Math.abs(target - S.s));
    animateTo(target, ms, () => {
      const at = personAt(p, target);
      showCard(); // the person is in this window: a docked card is up, so focusPoint can leave room for it
      // Centred on the map, or in the part a docked card leaves free.
      panTo(at.x, at.y, focusPoint(), Math.max(S.zoom, S.zoom0 + 2.2), 700);
      setTimeout(() => { showCard(); queueRender(); }, 720);
    });
    showCard();
  }

  function initCard() {
    $("card").addEventListener("click", (e) => {
      if (e.target.closest(".pc-close")) unpin();
    });
  }

  main().catch((err) => {
    $("loading").textContent = "Could not load the map data: " + err.message;
    console.error(err);
  });
})();
