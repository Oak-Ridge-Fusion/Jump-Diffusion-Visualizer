// One factory function per section, registered on window.Scenes by id.
// Each factory receives (sectionEl, canvas) and returns { start(), stop() }.
// start()/stop() are driven by an IntersectionObserver in main.js so
// animations only run while their section is on screen.
window.Scenes = (() => {
  const U = Util;
  const C = {
    blue: "#4fc3f7", blue2: "#2979ff", yellow: "#ffd54f",
    orange: "#ff8a65", teal: "#4fd1c5", green: "#66bb6a",
    red: "#ff5f56", purple: "#b39ddb", muted: "#5b6577", bg2: "#0b0f1a"
  };

  function floatLabel(sceneEl, styleOverrides) {
    const el = document.createElement("div");
    el.className = "float-label";
    Object.assign(el.style, { top: "12px", left: "12px" }, styleOverrides || {});
    sceneEl.appendChild(el);
    return {
      set(text, color) {
        el.textContent = text;
        el.style.color = color || "#eef2f7";
        el.classList.add("show");
      },
      hide() { el.classList.remove("show"); }
    };
  }

  function bgFieldArrows(ctx, w, h, t, dir = 1, color = C.teal, alpha = 0.22) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    const rows = 3;
    const cols = 6;
    for (let r = 0; r < rows; r++) {
      const y = h * (0.22 + r * 0.3);
      for (let c = 0; c < cols; c++) {
        const offset = ((t * 40 * dir) % (w / cols));
        const x = (c * (w / cols)) + offset;
        U.drawArrow(ctx, x, y, x + 24 * dir, y, color, 2);
      }
    }
    ctx.restore();
  }

  // Offscreen-generated cat-silhouette point cloud, cached and reused by
  // the normalizing-flow scenes.
  let _catCache = null;
  function getCatPoints(n) {
    if (_catCache && _catCache.length >= n) return _catCache.slice(0, n);
    const size = 240;
    const off = document.createElement("canvas");
    off.width = size; off.height = size;
    const c = off.getContext("2d");
    c.clearRect(0, 0, size, size);
    c.fillStyle = "#fff";
    c.beginPath();
    c.arc(size * 0.5, size * 0.58, size * 0.28, 0, Math.PI * 2);
    c.fill();
    c.beginPath();
    c.moveTo(size * 0.27, size * 0.42);
    c.lineTo(size * 0.33, size * 0.13);
    c.lineTo(size * 0.47, size * 0.36);
    c.closePath();
    c.fill();
    c.beginPath();
    c.moveTo(size * 0.73, size * 0.42);
    c.lineTo(size * 0.67, size * 0.13);
    c.lineTo(size * 0.53, size * 0.36);
    c.closePath();
    c.fill();
    c.globalCompositeOperation = "destination-out";
    c.beginPath(); c.arc(size * 0.40, size * 0.55, size * 0.028, 0, Math.PI * 2); c.fill();
    c.beginPath(); c.arc(size * 0.60, size * 0.55, size * 0.028, 0, Math.PI * 2); c.fill();
    c.globalCompositeOperation = "source-over";
    const img = c.getImageData(0, 0, size, size).data;
    const candidates = [];
    for (let y = 0; y < size; y += 2) {
      for (let x = 0; x < size; x += 2) {
        const idx = (y * size + x) * 4;
        if (img[idx + 3] > 128) candidates.push([x / size - 0.5, y / size - 0.5]);
      }
    }
    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }
    const pts = candidates.slice(0, Math.min(n, candidates.length));
    _catCache = pts;
    return pts;
  }

  const Scenes = {};

  // ============================================================
  // Section 1 — the particle's journey (field, collisions, jumps, boundary)
  // ============================================================
  Scenes.s1 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    const scene = canvas.parentElement;
    const label = floatLabel(scene);
    let p, trail, nextJumpAt, wallX, t;

    function reset(w, h) {
      p = { x: w * 0.08, y: h * 0.5, alpha: 1 };
      trail = [];
      nextJumpAt = 1.2 + Math.random() * 1.4;
      wallX = w * 0.88;
    }

    const loop = U.makeLoop((dt, tt) => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!p) reset(w, h);
      t = tt;
      ctx.clearRect(0, 0, w, h);
      bgFieldArrows(ctx, w, h, tt, 1, C.teal, 0.18);

      // wall
      ctx.save();
      ctx.strokeStyle = C.red;
      ctx.setLineDash([6, 6]);
      ctx.globalAlpha = 0.6;
      ctx.beginPath(); ctx.moveTo(wallX, 0); ctx.lineTo(wallX, h); ctx.stroke();
      ctx.restore();

      if (p.alpha > 0) {
        // drift + jitter (collisions)
        p.x += w * 0.028 * dt * 4;
        p.x += U.randn() * 14 * dt * 10;
        p.y += U.randn() * 10 * dt * 10;
        p.y = U.clamp(p.y, h * 0.15, h * 0.85);
        label.set("random collisions", C.blue);

        trail.push({ x: p.x, y: p.y });
        if (trail.length > 40) trail.shift();

        if (tt > nextJumpAt && p.x < wallX - 30) {
          const jumpDist = w * (0.09 + Math.random() * 0.08);
          const from = { x: p.x, y: p.y };
          p.x += jumpDist;
          trail.push({ x: from.x, y: from.y });
          trail.push({ x: p.x, y: p.y });
          nextJumpAt = tt + 1.4 + Math.random() * 1.6;
          label.set("JUMP!", C.orange);
          ctx.save();
          ctx.strokeStyle = C.orange;
          ctx.lineWidth = 3;
          ctx.globalAlpha = 0.9;
          ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(p.x, p.y); ctx.stroke();
          ctx.restore();
        }

        if (p.x >= wallX) {
          p.alpha = 0.999;
          label.set("boundary → absorbed", C.red);
        }
      } else {
        p.alpha -= dt * 2;
        if (p.alpha <= 0) {
          reset(w, h);
        }
      }

      // trail
      ctx.save();
      for (let i = 0; i < trail.length; i++) {
        ctx.globalAlpha = (i / trail.length) * 0.5;
        ctx.fillStyle = C.blue;
        ctx.beginPath();
        ctx.arc(trail[i].x, trail[i].y, 2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // particle
      ctx.save();
      ctx.globalAlpha = Math.max(0, p.alpha);
      ctx.fillStyle = C.yellow;
      ctx.shadowColor = C.yellow;
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });

    return {
      start() { reset(canvas.clientWidth, canvas.clientHeight); loop.start(); },
      stop() { loop.stop(); }
    };
  };

  // ============================================================
  // Section 2 — Brownian motion: 100 particles, time slider, histogram
  // ============================================================
  Scenes.s2 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    const N = 100, STEPS = 240;
    const slider = section.querySelector('[data-role="time-slider"]');
    const eq = section.querySelector('[data-role="eq2"]');
    let paths = null;
    let playing = false;
    let interacted = false;

    function regen() {
      paths = [];
      for (let i = 0; i < N; i++) {
        const arr = new Float32Array(STEPS);
        let x = 0;
        for (let s = 0; s < STEPS; s++) {
          x += U.randn() * 0.35;
          arr[s] = x;
        }
        paths.push(arr);
      }
    }

    function markInteracted() {
      if (!interacted) { interacted = true; eq && eq.classList.add("visible"); }
    }

    function render() {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      const idx = Math.round((slider ? slider.value / 100 : 0) * (STEPS - 1));
      const rowY = h * 0.32, histY = h * 0.5, histH = h * 0.42;
      const cx = w * 0.5;
      const scale = w * 0.028;

      const xs = paths.map(p => p[idx]);
      ctx.save();
      ctx.fillStyle = C.muted;
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText(`t = ${(idx / (STEPS - 1) * 3).toFixed(2)}`, 14, 20);
      ctx.restore();

      // dots
      for (let i = 0; i < xs.length; i++) {
        ctx.save();
        ctx.globalAlpha = 0.8;
        ctx.fillStyle = C.blue;
        ctx.beginPath();
        ctx.arc(cx + xs[i] * scale, rowY + (i % 7) * 3 - 9, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // histogram
      const lo = -6, hi = 6;
      U.drawHistogram(ctx, xs, 28, lo, hi, w * 0.06, histY, w * 0.88, histH, C.teal);
      ctx.save();
      ctx.strokeStyle = C.border || "#1e2635";
      ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(w * 0.06, histY + histH); ctx.lineTo(w * 0.94, histY + histH); ctx.stroke();
      ctx.restore();
    }

    const loop = U.makeLoop(() => {
      if (playing && slider) {
        let v = Number(slider.value) + 0.6;
        if (v > 100) { v = 100; playing = false; }
        slider.value = v;
        markInteracted();
      }
      render();
    });

    let wiredOnce = false;
    function wireOnce() {
      if (wiredOnce) return; wiredOnce = true;
      if (slider) slider.addEventListener("input", markInteracted);
      const playBtn = section.querySelector('[data-role="play2"]');
      if (playBtn) playBtn.addEventListener("click", () => { playing = true; markInteracted(); });
    }

    return {
      start() { wireOnce(); if (!paths) regen(); loop.start(); },
      stop() { loop.stop(); }
    };
  };

  // ============================================================
  // Section 3 — SDE: drift + noise, hoverable legend
  // ============================================================
  Scenes.s3 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    let emphasis = null; // 'drift' | 'noise' | null
    const particles = [];
    const NP = 6;

    function reset(w, h) {
      particles.length = 0;
      for (let i = 0; i < NP; i++) {
        particles.push({ x: w * (0.08 + Math.random() * 0.1), y: h * (0.2 + (i / NP) * 0.6) });
      }
    }

    const loop = U.makeLoop((dt, tt) => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (particles.length === 0) reset(w, h);
      ctx.clearRect(0, 0, w, h);
      const driftBoost = emphasis === "drift" ? 2.2 : 1;
      const noiseBoost = emphasis === "noise" ? 2.4 : 1;

      bgFieldArrows(ctx, w, h, tt, 1, C.teal, emphasis === "drift" ? 0.4 : 0.14);

      particles.forEach((p, i) => {
        p.x += w * 0.02 * dt * 4 * driftBoost;
        p.y += U.randn() * 9 * dt * 10 * noiseBoost;
        p.y = U.clamp(p.y, h * 0.12, h * 0.88);
        if (p.x > w * 0.94) p.x = w * (0.06 + Math.random() * 0.05);

        if (emphasis === "noise") {
          ctx.save();
          ctx.globalAlpha = 0.18;
          ctx.strokeStyle = C.blue;
          ctx.beginPath(); ctx.arc(p.x, p.y, 16, 0, Math.PI * 2); ctx.stroke();
          ctx.restore();
        }
        ctx.save();
        ctx.fillStyle = C.yellow;
        ctx.shadowColor = C.yellow;
        ctx.shadowBlur = emphasis ? 14 : 6;
        ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
      });
    });

    let wiredOnce = false;
    function wireOnce() {
      if (wiredOnce) return; wiredOnce = true;
      section.querySelectorAll(".legend .item").forEach(item => {
        const term = item.dataset.term;
        const setOn = () => {
          emphasis = term;
          section.querySelectorAll(".legend .item").forEach(i2 => i2.classList.toggle("active", i2 === item));
          section.querySelectorAll(`.eq .term.${term}`).forEach(t2 => t2.classList.add("hot"));
        };
        const setOff = () => {
          emphasis = null;
          item.classList.remove("active");
          section.querySelectorAll(`.eq .term.${term}`).forEach(t2 => t2.classList.remove("hot"));
        };
        item.addEventListener("mouseenter", setOn);
        item.addEventListener("mouseleave", setOff);
        item.addEventListener("click", setOn);
      });
    }

    return {
      start() { wireOnce(); reset(canvas.clientWidth, canvas.clientHeight); loop.start(); },
      stop() { loop.stop(); }
    };
  };

  // ============================================================
  // Section 4 — jumps appear on top of Brownian motion
  // ============================================================
  Scenes.s4 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    let p, trail, nextJumpAt, jumpFlash;
    const eqExtra = section.querySelector('[data-role="jump-term"]');

    function reset(w, h) {
      p = { x: w * 0.12, y: h * 0.5 };
      trail = [];
      nextJumpAt = 1.4 + Math.random() * 1.2;
      jumpFlash = 0;
    }

    const loop = U.makeLoop((dt, tt) => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!p) reset(w, h);
      ctx.clearRect(0, 0, w, h);

      p.x += U.randn() * 8 * dt * 10;
      p.y += U.randn() * 8 * dt * 10;
      p.x = U.clamp(p.x, w * 0.05, w * 0.95);
      p.y = U.clamp(p.y, h * 0.1, h * 0.9);
      trail.push({ x: p.x, y: p.y, big: false });

      if (tt > nextJumpAt) {
        const from = { x: p.x, y: p.y };
        const ang = Math.random() * Math.PI * 2;
        const dist = Math.min(w, h) * 0.28;
        p.x = U.clamp(p.x + Math.cos(ang) * dist, w * 0.05, w * 0.95);
        p.y = U.clamp(p.y + Math.sin(ang) * dist, h * 0.1, h * 0.9);
        trail.push({ x: from.x, y: from.y, big: true, tx: p.x, ty: p.y });
        nextJumpAt = tt + 1.6 + Math.random() * 1.4;
        jumpFlash = 1;
        if (eqExtra) eqExtra.classList.add("visible");
      }

      jumpFlash = Math.max(0, jumpFlash - dt * 1.5);
      if (trail.length > 60) trail.shift();

      ctx.save();
      for (let i = 0; i < trail.length; i++) {
        const pt = trail[i];
        if (pt.big) {
          ctx.globalAlpha = 0.55;
          ctx.strokeStyle = C.orange;
          ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(pt.x, pt.y); ctx.lineTo(pt.tx, pt.ty); ctx.stroke();
        } else {
          ctx.globalAlpha = (i / trail.length) * 0.45;
          ctx.fillStyle = C.blue;
          ctx.beginPath(); ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2); ctx.fill();
        }
      }
      ctx.restore();

      if (jumpFlash > 0) {
        ctx.save();
        ctx.globalAlpha = jumpFlash * 0.5;
        ctx.fillStyle = C.orange;
        ctx.beginPath(); ctx.arc(p.x, p.y, 22, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
      }

      ctx.save();
      ctx.fillStyle = C.yellow;
      ctx.shadowColor = C.yellow;
      ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    });

    return {
      start() { reset(canvas.clientWidth, canvas.clientHeight); loop.start(); },
      stop() { loop.stop(); }
    };
  };

  // ============================================================
  // Section 5 — transport: 1000 particles spreading from a point
  // ============================================================
  Scenes.s5 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    const N = 900;
    let xs, t0;
    const restartBtn = section.querySelector('[data-role="restart5"]');

    function reset() {
      xs = new Float32Array(N).fill(0);
      t0 = performance.now();
    }

    const loop = U.makeLoop((dt) => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!xs) reset();
      ctx.clearRect(0, 0, w, h);

      const elapsed = (performance.now() - t0) / 1000;
      if (elapsed < 6) {
        for (let i = 0; i < N; i++) xs[i] += U.randn() * 0.6 * dt * 6 + dt * 0.4;
      }

      const cx = w * 0.5, scale = w * 0.02;
      const rowY = h * 0.28;
      ctx.save();
      for (let i = 0; i < N; i += 3) {
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = C.blue;
        const jitterY = ((i * 2654435761) % 1000) / 1000 * h * 0.22;
        ctx.beginPath();
        ctx.arc(cx + xs[i] * scale, rowY + jitterY, 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      U.drawHistogram(ctx, Array.from(xs), 34, -14, 14, w * 0.04, h * 0.56, w * 0.92, h * 0.36, C.purple);

      ctx.save();
      ctx.fillStyle = C.muted;
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText(elapsed < 6 ? "spreading…" : "settled distribution", 14, 18);
      ctx.restore();
    });

    let wiredOnce = false;
    function wireOnce() {
      if (wiredOnce) return; wiredOnce = true;
      if (restartBtn) restartBtn.addEventListener("click", reset);
    }

    return {
      start() { wireOnce(); if (!xs) reset(); loop.start(); },
      stop() { loop.stop(); }
    };
  };

  // ============================================================
  // Section 6 — why not solve the PDE directly (matrix fill + CPU melt)
  // ============================================================
  Scenes.s6 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    const caption = section.querySelector('[data-role="melt-caption"]');
    const CYCLE = 7.2;
    const GRID = 18;

    const loop = U.makeLoop((dt, tt) => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      const phase = tt % CYCLE;

      const gridArea = { x: w * 0.06, y: h * 0.12, w: w * 0.42, h: h * 0.76 };
      const cell = gridArea.w / GRID;
      const fillFrac = U.clamp(phase / 3.6, 0, 1);
      const filledCount = Math.floor(fillFrac * GRID * GRID);

      ctx.save();
      ctx.font = "11px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.fillText(`unknowns solved: ${filledCount} / ${GRID * GRID}`, gridArea.x, gridArea.y - 8);
      for (let i = 0; i < GRID * GRID; i++) {
        const gx = i % GRID, gy = Math.floor(i / GRID);
        const seeded = ((i * 9301 + 49297) % 233280) / 233280;
        const on = seeded < fillFrac;
        ctx.fillStyle = on ? C.blue2 : "#141b28";
        ctx.globalAlpha = on ? 0.85 : 1;
        ctx.fillRect(gridArea.x + gx * cell + 1, gridArea.y + gy * cell + 1, cell - 2, cell - 2);
      }
      ctx.restore();

      // CPU + melt
      const cpuX = w * 0.62, cpuY = h * 0.16, cpuW = w * 0.3, cpuH = h * 0.5;
      const meltPhase = U.clamp((phase - 3.8) / 2.6, 0, 1);
      const meltAmp = meltPhase * 22;

      ctx.save();
      ctx.fillStyle = "#20293b";
      ctx.strokeStyle = C.teal;
      ctx.lineWidth = 2;
      ctx.beginPath();
      const steps = 24;
      ctx.moveTo(cpuX, cpuY);
      ctx.lineTo(cpuX + cpuW, cpuY);
      ctx.lineTo(cpuX + cpuW, cpuY + cpuH);
      for (let i = steps; i >= 0; i--) {
        const fx = cpuX + (i / steps) * cpuW;
        const droop = Math.sin(i * 1.3 + tt * 2) * meltAmp * (i % 3 === 0 ? 1 : 0.5);
        ctx.lineTo(fx, cpuY + cpuH + Math.max(0, droop));
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // pins
      ctx.fillStyle = C.teal;
      for (let i = 0; i < 6; i++) {
        const px = cpuX + cpuW * 0.12 + i * (cpuW * 0.76 / 5);
        ctx.fillRect(px - 2, cpuY - 8, 4, 8);
      }
      ctx.restore();

      // drips
      if (meltPhase > 0.15) {
        ctx.save();
        ctx.fillStyle = C.teal;
        for (let i = 0; i < 4; i++) {
          const dripT = ((tt * 0.6 + i * 0.27) % 1);
          const dx = cpuX + cpuW * (0.15 + i * 0.22);
          const dy = cpuY + cpuH + dripT * 40 * meltPhase;
          ctx.globalAlpha = (1 - dripT) * meltPhase * 0.8;
          ctx.beginPath(); ctx.arc(dx, dy, 3, 0, Math.PI * 2); ctx.fill();
        }
        ctx.restore();
      }

      if (caption) caption.style.opacity = meltPhase > 0.35 ? 1 : 0;
    });

    return { start() { loop.start(); }, stop() { loop.stop(); } };
  };

  // ============================================================
  // Section 7 — particles → histogram → smooth curve
  // ============================================================
  Scenes.s7 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    const N = 260;
    let samples = null;
    const CYCLE = 7;

    function regen() {
      samples = [];
      for (let i = 0; i < N; i++) samples.push(U.randn() * 1.6);
    }

    const loop = U.makeLoop((dt, tt) => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!samples) regen();
      ctx.clearRect(0, 0, w, h);
      const phase = tt % CYCLE;
      const p1 = U.clamp(phase / 2.2, 0, 1);       // scatter -> stack
      const p2 = U.clamp((phase - 2.6) / 2.2, 0, 1); // bars -> curve
      const bins = 24, lo = -5, hi = 5;
      const areaX = w * 0.06, areaY = h * 0.14, areaW = w * 0.88, areaH = h * 0.72;
      const counts = new Array(bins).fill(0);
      const span = hi - lo;

      samples.forEach((v) => {
        let idx = Math.floor(((v - lo) / span) * bins);
        idx = U.clamp(idx, 0, bins - 1);
        counts[idx]++;
      });
      const maxCount = Math.max(1, ...counts);
      const barW = areaW / bins;

      ctx.save();
      if (p1 < 1) {
        // scatter falling into place
        samples.forEach((v, i) => {
          const idx = U.clamp(Math.floor(((v - lo) / span) * bins), 0, bins - 1);
          const seatIdx = counts.slice(0, idx).length ? i % 30 : i % 30;
          const targetX = areaX + idx * barW + barW / 2;
          const targetY = areaY + areaH - ((seatIdx + 1) * 3);
          const startY = areaY - 20;
          const y = U.lerp(startY, targetY, U.easeOutCubic(p1));
          ctx.globalAlpha = 0.55;
          ctx.fillStyle = C.blue;
          ctx.beginPath(); ctx.arc(targetX, y, 2, 0, Math.PI * 2); ctx.fill();
        });
      } else {
        // bars, then morph into curve
        ctx.globalAlpha = 1 - p2 * 0.7;
        for (let i = 0; i < bins; i++) {
          const bh = (counts[i] / maxCount) * areaH;
          ctx.fillStyle = C.teal;
          ctx.fillRect(areaX + i * barW + 1, areaY + areaH - bh, barW - 2, bh);
        }
        if (p2 > 0) {
          ctx.globalAlpha = p2;
          ctx.strokeStyle = C.yellow;
          ctx.lineWidth = 3;
          ctx.beginPath();
          for (let i = 0; i <= bins; i++) {
            const x = areaX + i * barW;
            const cIdx = U.clamp(i, 0, bins - 1);
            const bh = (counts[cIdx] / maxCount) * areaH;
            const y = areaY + areaH - bh;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }
      ctx.restore();

      if (phase > CYCLE - 0.3) regen();
    });

    return { start() { regen(); loop.start(); }, stop() { loop.stop(); } };
  };

  // ============================================================
  // Section 8 — why generative AI (simulation → NN → distribution)
  // ============================================================
  Scenes.s8 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    const particles = [];
    const N = 140;

    function reset(w, h) {
      particles.length = 0;
      for (let i = 0; i < N; i++) {
        particles.push({ x: Math.random() * w * 0.28, y: Math.random() * h, vx: (Math.random() - 0.5) * 30, vy: (Math.random() - 0.5) * 30 });
      }
    }

    const loop = U.makeLoop((dt, tt) => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (particles.length === 0) reset(w, h);
      ctx.clearRect(0, 0, w, h);

      // left: chaotic particle simulation
      ctx.save();
      ctx.fillStyle = C.muted;
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText("expensive: simulate millions of particles", 12, 18);
      particles.forEach(p => {
        p.x += p.vx * dt; p.y += p.vy * dt;
        if (p.x < 0 || p.x > w * 0.28) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
        ctx.globalAlpha = 0.6;
        ctx.fillStyle = C.blue;
        ctx.beginPath(); ctx.arc(p.x, p.y, 1.8, 0, Math.PI * 2); ctx.fill();
      });
      ctx.restore();

      // arrow
      U.drawArrow(ctx, w * 0.30, h * 0.5, w * 0.40, h * 0.5, C.muted, 2);

      // middle: NN diagram
      const nnX = w * 0.5, nnW = w * 0.16;
      const layers = [3, 4, 3];
      const pulse = (Math.sin(tt * 3) + 1) / 2;
      const nodePos = [];
      layers.forEach((count, li) => {
        const lx = nnX + li * (nnW / (layers.length - 1));
        const col = [];
        for (let i = 0; i < count; i++) {
          const ly = h * 0.5 + (i - (count - 1) / 2) * 26;
          col.push({ x: lx, y: ly });
        }
        nodePos.push(col);
      });
      ctx.save();
      ctx.strokeStyle = C.teal;
      ctx.globalAlpha = 0.35;
      for (let li = 0; li < nodePos.length - 1; li++) {
        nodePos[li].forEach(a => nodePos[li + 1].forEach(b => {
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }));
      }
      ctx.restore();
      nodePos.forEach((col, li) => col.forEach(n => {
        ctx.save();
        ctx.fillStyle = C.yellow;
        ctx.globalAlpha = 0.6 + pulse * 0.4;
        ctx.shadowColor = C.yellow; ctx.shadowBlur = 8;
        ctx.beginPath(); ctx.arc(n.x, n.y, 5, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
      }));

      // arrow
      U.drawArrow(ctx, w * 0.62, h * 0.5, w * 0.72, h * 0.5, C.muted, 2);

      // right: learned distribution (smooth curve)
      ctx.save();
      ctx.fillStyle = C.muted;
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText("cheap: one trained network", w * 0.74, 18);
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 3;
      ctx.beginPath();
      const areaX = w * 0.74, areaW = w * 0.22, baseY = h * 0.66, curveH = h * 0.4;
      for (let i = 0; i <= 40; i++) {
        const x = areaX + (i / 40) * areaW;
        const u = (i / 40 - 0.5) * 4;
        const y = baseY - Math.exp(-u * u / 2) * curveH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    });

    return { start() { reset(canvas.clientWidth, canvas.clientHeight); loop.start(); }, stop() { loop.stop(); } };
  };

  // ============================================================
  // Section 9 — normalizing flow: Gaussian cloud warps into a shape
  // ============================================================
  function makeFlowWarp(canvas, opts = {}) {
    const { ctx } = U.setupCanvas(canvas);
    const N = opts.n || 260;
    let gauss = null, target = null;
    let manualT = 0;
    let autoT = null; // when set, animating automatically

    function regen() {
      gauss = [];
      for (let i = 0; i < N; i++) gauss.push({ x: U.randn(), y: U.randn() });
      target = getCatPoints(N);
    }

    function render(tVal) {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      const cx = w * 0.5, cy = h * 0.52;
      const gScale = Math.min(w, h) * 0.11;
      const tScale = Math.min(w, h) * 0.86;
      const te = U.easeInOutCubic(tVal);

      ctx.save();
      ctx.fillStyle = C.muted;
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText(tVal < 0.05 ? "z ~ N(0, I)" : (tVal > 0.95 ? "x = f(z)" : "warping…"), 12, 18);
      ctx.restore();

      for (let i = 0; i < N; i++) {
        const gx = cx + gauss[i].x * gScale;
        const gy = cy + gauss[i].y * gScale;
        const tx = cx + (target[i % target.length][0]) * tScale;
        const ty = cy + (target[i % target.length][1]) * tScale;
        const x = U.lerp(gx, tx, te);
        const y = U.lerp(gy, ty, te);
        ctx.save();
        ctx.globalAlpha = 0.75;
        ctx.fillStyle = U.lerp(0, 1, te) > 0.5 ? C.yellow : C.blue;
        ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
      }
    }

    const loop = U.makeLoop((dt) => {
      if (autoT !== null) {
        autoT += dt / 2.2;
        if (autoT >= 1) { autoT = 1; }
        render(autoT);
        if (autoT >= 1) autoT = null;
      } else {
        render(manualT);
      }
    });

    return {
      start() { if (!gauss) regen(); loop.start(); },
      stop() { loop.stop(); },
      setManual(v) { manualT = v; },
      playAuto() { autoT = 0; },
      regen
    };
  }

  Scenes.s9 = function (section, canvas) {
    const warp = makeFlowWarp(canvas, { n: 260 });
    const slider = section.querySelector('[data-role="warp-slider"]');
    const playBtn = section.querySelector('[data-role="warp-play"]');
    const nextBtn = section.querySelector('[data-role="eq-next-9"]');
    const eqs = section.querySelectorAll('[data-role^="eq9-"]');
    let step = 0;

    let wiredOnce = false;
    function wireOnce() {
      if (wiredOnce) return; wiredOnce = true;
      eqs.forEach((e, i) => e.classList.toggle("visible", i <= step));
      if (slider) slider.addEventListener("input", () => warp.setManual(Number(slider.value) / 100));
      if (playBtn) playBtn.addEventListener("click", () => { warp.playAuto(); if (slider) slider.value = 100; });
      if (nextBtn) nextBtn.addEventListener("click", () => {
        step = Math.min(step + 1, eqs.length - 1);
        eqs.forEach((e, i) => e.classList.toggle("visible", i <= step));
        if (step >= eqs.length - 1) nextBtn.textContent = "shown: x = f(z), z = f⁻¹(x), Jacobian";
        else nextBtn.textContent = `Next (${step + 1}/${eqs.length})`;
      });
    }

    return { start() { wireOnce(); warp.start(); }, stop() { warp.stop(); } };
  };

  // ============================================================
  // Section 10 — diffusion: forward noising, reverse denoising
  // ============================================================
  Scenes.s10 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    const SIZE = 52;
    let base = null, noiseField = null;
    let k = 0; // 0..K clean->noisy
    const K = 40;
    let dir = 0; // 1 forward auto, -1 reverse auto, 0 idle
    const phaseLabel = section.querySelector('[data-role="diff-phase"]');

    function buildBase() {
      const off = document.createElement("canvas");
      off.width = SIZE; off.height = SIZE;
      const c = off.getContext("2d");
      c.fillStyle = "#0b0f1a"; c.fillRect(0, 0, SIZE, SIZE);
      c.fillStyle = "#ffd54f";
      c.beginPath(); c.arc(SIZE * 0.5, SIZE * 0.56, SIZE * 0.28, 0, Math.PI * 2); c.fill();
      c.beginPath();
      c.moveTo(SIZE * 0.27, SIZE * 0.40); c.lineTo(SIZE * 0.33, SIZE * 0.12); c.lineTo(SIZE * 0.47, SIZE * 0.34);
      c.closePath(); c.fill();
      c.beginPath();
      c.moveTo(SIZE * 0.73, SIZE * 0.40); c.lineTo(SIZE * 0.67, SIZE * 0.12); c.lineTo(SIZE * 0.53, SIZE * 0.34);
      c.closePath(); c.fill();
      c.fillStyle = "#0b0f1a";
      c.beginPath(); c.arc(SIZE * 0.40, SIZE * 0.53, SIZE * 0.028, 0, Math.PI * 2); c.fill();
      c.beginPath(); c.arc(SIZE * 0.60, SIZE * 0.53, SIZE * 0.028, 0, Math.PI * 2); c.fill();
      base = c.getImageData(0, 0, SIZE, SIZE);
      noiseField = new Float32Array(SIZE * SIZE * 3);
      for (let i = 0; i < noiseField.length; i++) noiseField[i] = U.randn();
    }

    function render() {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      const alphaBar = Math.pow(1 - k / K, 2); // signal fraction remaining
      const out = ctx.createImageData(SIZE, SIZE);
      for (let p = 0; p < SIZE * SIZE; p++) {
        for (let ch = 0; ch < 3; ch++) {
          const signal = base.data[p * 4 + ch];
          const noiseVal = 128 + noiseField[p * 3 + ch] * 60;
          const v = Math.sqrt(alphaBar) * signal + Math.sqrt(1 - alphaBar) * noiseVal;
          out.data[p * 4 + ch] = U.clamp(v, 0, 255);
        }
        out.data[p * 4 + 3] = 255;
      }
      const off2 = document.createElement("canvas");
      off2.width = SIZE; off2.height = SIZE;
      off2.getContext("2d").putImageData(out, 0, 0);

      const side = Math.min(w, h) * 0.72;
      ctx.save();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off2, (w - side) / 2, (h - side) / 2, side, side);
      ctx.restore();

      ctx.save();
      ctx.fillStyle = C.muted;
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText(`step ${k} / ${K}`, 12, 18);
      ctx.restore();

      if (phaseLabel) phaseLabel.textContent = dir >= 0
        ? (k === 0 ? "clean" : (k >= K ? "pure noise" : "forward process — adding noise"))
        : (k <= 0 ? "clean" : "reverse process — denoising");
    }

    const loop = U.makeLoop((dt) => {
      if (!base) buildBase();
      if (dir !== 0) {
        k += dir * dt * 22;
        k = U.clamp(k, 0, K);
        if (k <= 0 || k >= K) dir = 0;
      }
      render();
    });

    let wiredOnce = false;
    function wireOnce() {
      if (wiredOnce) return; wiredOnce = true;
      const fBtn = section.querySelector('[data-role="diff-forward"]');
      const rBtn = section.querySelector('[data-role="diff-reverse"]');
      const resetBtn = section.querySelector('[data-role="diff-reset"]');
      if (fBtn) fBtn.addEventListener("click", () => { dir = 1; });
      if (rBtn) rBtn.addEventListener("click", () => { dir = -1; });
      if (resetBtn) resetBtn.addEventListener("click", () => { k = 0; dir = 0; });
    }

    return { start() { wireOnce(); if (!base) buildBase(); loop.start(); }, stop() { loop.stop(); } };
  };

  // ============================================================
  // Section 11 — split screen: NF vs Diffusion, race
  // ============================================================
  Scenes.s11 = function (section, canvas) {
    // canvas here is actually two canvases; section stores both.
    const leftCanvas = section.querySelector('[data-role="split-nf"]');
    const rightCanvas = section.querySelector('[data-role="split-diff"]');
    const nfLabel = section.querySelector('[data-role="nf-step-label"]');
    const diffLabel = section.querySelector('[data-role="diff-step-label"]');

    const nf = makeFlowWarp(leftCanvas, { n: 180 });
    const { ctx: rctx } = U.setupCanvas(rightCanvas);

    let diffProgress = null; // 0..1 or null idle
    const rSteps = ["noise", "step", "step", "step", "answer"];

    function renderRight() {
      const w = rightCanvas.clientWidth, h = rightCanvas.clientHeight;
      rctx.clearRect(0, 0, w, h);
      const t = diffProgress === null ? 0 : diffProgress;
      const cx = w * 0.5, cy = h * 0.52;
      const n = 180;
      const target = getCatPoints(n);
      for (let i = 0; i < n; i++) {
        const gx = cx + U.randn() * Math.min(w, h) * 0.02;
        const gy = cy + U.randn() * Math.min(w, h) * 0.02;
        // stepwise easing: quantize t into 4 discrete jumps for a "steps" feel
        const steps = 4;
        const qt = Math.floor(t * steps) / steps;
        const smooth = U.lerp(qt, t, 0.35);
        const tx = cx + target[i % target.length][0] * Math.min(w, h) * 0.86;
        const ty = cy + target[i % target.length][1] * Math.min(w, h) * 0.86;
        const noiseX = cx + (Math.random() - 0.5) * Math.min(w, h) * 0.9;
        const noiseY = cy + (Math.random() - 0.5) * Math.min(w, h) * 0.9;
        const x = U.lerp(noiseX, tx, U.easeInOutCubic(smooth));
        const y = U.lerp(noiseY, ty, U.easeInOutCubic(smooth));
        rctx.save();
        rctx.globalAlpha = 0.75;
        rctx.fillStyle = smooth > 0.5 ? C.yellow : C.orange;
        rctx.beginPath(); rctx.arc(x, y, 2.4, 0, Math.PI * 2); rctx.fill();
        rctx.restore();
      }
    }

    const loop = U.makeLoop((dt) => {
      if (diffProgress !== null) {
        diffProgress += dt / 2.6;
        if (diffProgress >= 1) diffProgress = 1;
        const idx = Math.min(rSteps.length - 1, Math.floor(diffProgress * rSteps.length));
        if (diffLabel) diffLabel.textContent = rSteps[idx];
      }
      renderRight();
    });

    let wiredOnce = false;
    function wireOnce() {
      if (wiredOnce) return; wiredOnce = true;
      const raceBtn = section.querySelector('[data-role="race-btn"]');
      if (raceBtn) raceBtn.addEventListener("click", () => {
        nf.playAuto();
        if (nfLabel) { nfLabel.textContent = "warp"; setTimeout(() => { if (nfLabel) nfLabel.textContent = "answer"; }, 900); }
        diffProgress = 0;
        if (diffLabel) diffLabel.textContent = "noise";
      });
    }

    return {
      start() { wireOnce(); nf.start(); loop.start(); },
      stop() { nf.stop(); loop.stop(); }
    };
  };

  // ============================================================
  // Section 12 — decision table (no canvas, pure DOM)
  // ============================================================
  Scenes.s12 = function () {
    return { start() {}, stop() {} };
  };

  // ============================================================
  // Section 13 — our research: SDE + conditional transition fan-out
  // ============================================================
  Scenes.s13 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    const NP = 26;
    let outcomes = null;

    function reset(w, h) {
      outcomes = [];
      for (let i = 0; i < NP; i++) {
        const ang = (i / NP) * Math.PI * 2;
        outcomes.push({ ang, r: 0.5 + Math.random() * 0.5, phase: Math.random() * Math.PI * 2 });
      }
    }

    const loop = U.makeLoop((dt, tt) => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!outcomes) reset(w, h);
      ctx.clearRect(0, 0, w, h);
      const x0 = w * 0.22, y0 = h * 0.5;
      const x1 = w * 0.68, y1 = h * 0.5;

      ctx.save();
      ctx.fillStyle = C.muted;
      ctx.font = "13px Inter, sans-serif";
      ctx.fillText("X_t", x0 - 10, y0 - 18);
      ctx.fillText("possible X_(t+Δt)", x1 - 40, y1 - h * 0.28);
      ctx.restore();

      outcomes.forEach(o => {
        const wob = Math.sin(tt * 1.4 + o.phase) * 6;
        const ox = x1 + Math.cos(o.ang) * (o.r * w * 0.14 + wob);
        const oy = y1 + Math.sin(o.ang) * (o.r * h * 0.24 + wob);
        ctx.save();
        ctx.globalAlpha = 0.18;
        ctx.strokeStyle = C.blue;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(ox, oy); ctx.stroke();
        ctx.restore();
        ctx.save();
        ctx.fillStyle = C.blue;
        ctx.globalAlpha = 0.8;
        ctx.beginPath(); ctx.arc(ox, oy, 3, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
      });

      ctx.save();
      ctx.fillStyle = C.yellow;
      ctx.shadowColor = C.yellow; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(x0, y0, 7, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    });

    return { start() { reset(canvas.clientWidth, canvas.clientHeight); loop.start(); }, stop() { loop.stop(); } };
  };

  // ============================================================
  // Section 14 — ablation panels (static real images, staggered reveal)
  // ============================================================
  Scenes.s14 = function () {
    return { start() {}, stop() {} };
  };

  // ============================================================
  // Section 15 — exit network decision
  // ============================================================
  Scenes.s15 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    const label = floatLabel(canvas.parentElement, { top: "12px", right: "12px", left: "auto" });
    let p, state, stateT, wallX, outcome;

    function reset(w, h) {
      p = { x: w * 0.1, y: h * 0.5, alpha: 1 };
      state = "approach"; stateT = 0;
      wallX = w * 0.82;
      outcome = null;
      label.set("random walk toward wall", C.blue);
    }

    const loop = U.makeLoop((dt) => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!p) reset(w, h);
      ctx.clearRect(0, 0, w, h);
      stateT += dt;

      ctx.save();
      ctx.strokeStyle = C.red; ctx.setLineDash([6, 6]); ctx.globalAlpha = 0.6;
      ctx.beginPath(); ctx.moveTo(wallX, 0); ctx.lineTo(wallX, h); ctx.stroke();
      ctx.restore();

      if (state === "approach") {
        p.x += w * 0.02 * dt * 4 + U.randn() * 10 * dt * 8;
        p.y += U.randn() * 8 * dt * 8;
        p.y = U.clamp(p.y, h * 0.15, h * 0.85);
        if (p.x >= wallX) { state = "decide"; stateT = 0; label.set("Exit?", C.yellow); }
      } else if (state === "decide") {
        if (stateT > 0.8) {
          outcome = Math.random() < 0.5 ? "yes" : "no";
          state = outcome; stateT = 0;
          label.set(outcome === "yes" ? "YES → remove" : "NO → flow map continues", outcome === "yes" ? C.red : C.green);
        }
      } else if (state === "yes") {
        p.alpha -= dt * 1.6;
        if (p.alpha <= 0) { reset(w, h); }
      } else if (state === "no") {
        if (stateT < 0.05) p.x = w * (0.35 + Math.random() * 0.2);
        if (stateT > 0.5) { state = "approach"; label.set("random walk toward wall", C.blue); }
      }

      // decision box
      ctx.save();
      ctx.strokeStyle = C.border || "#1e2635";
      ctx.fillStyle = "rgba(16,22,36,0.9)";
      const bx = wallX - 118, by = h * 0.14, bw = 106, bh = 76;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(bx, by, bw, bh, 8) : ctx.rect(bx, by, bw, bh);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = C.text || "#eef2f7";
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText("Exit?", bx + 10, by + 18);
      ctx.fillStyle = state === "yes" ? C.red : (C.muted);
      ctx.fillText("YES → remove", bx + 8, by + 40);
      ctx.fillStyle = state === "no" ? C.green : (C.muted);
      ctx.fillText("NO → flow map", bx + 8, by + 60);
      ctx.restore();

      ctx.save();
      ctx.globalAlpha = Math.max(0, p.alpha);
      ctx.fillStyle = C.yellow;
      ctx.shadowColor = C.yellow; ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    });

    return { start() { reset(canvas.clientWidth, canvas.clientHeight); loop.start(); }, stop() { loop.stop(); } };
  };

  // ============================================================
  // Section 16 — speed race + real 39x speedup counter
  // ============================================================
  Scenes.s16 = function (section, canvas) {
    const leftCanvas = section.querySelector('[data-role="euler-canvas"]');
    const rightCanvas = section.querySelector('[data-role="flowmap-canvas"]');
    const stepCountEl = section.querySelector('[data-role="euler-count"]');
    const flowCountEl = section.querySelector('[data-role="flow-count"]');
    const bigNumberEl = section.querySelector('[data-role="big-number"]');
    const REAL_SPEEDUP = 39.11270324740738;

    const { ctx: lctx } = U.setupCanvas(leftCanvas);
    const { ctx: rctx } = U.setupCanvas(rightCanvas);

    let raceT = null; // seconds since race start, or null idle
    let counterShown = false;

    function renderEuler(t) {
      const w = leftCanvas.clientWidth, h = leftCanvas.clientHeight;
      lctx.clearRect(0, 0, w, h);
      const totalSteps = 200;
      const dur = 2.2;
      const frac = t === null ? 0 : U.clamp(t / dur, 0, 1);
      const stepsDone = Math.floor(frac * totalSteps);
      if (stepCountEl) stepCountEl.textContent = stepsDone;
      lctx.save();
      lctx.strokeStyle = C.muted; lctx.globalAlpha = 0.3;
      lctx.beginPath(); lctx.moveTo(w * 0.05, h * 0.5); lctx.lineTo(w * 0.95, h * 0.5); lctx.stroke();
      lctx.restore();
      const x = U.lerp(w * 0.05, w * 0.95, frac);
      // tiny hop marks
      lctx.save();
      lctx.fillStyle = C.blue;
      for (let i = 0; i < stepsDone; i += 4) {
        const hx = U.lerp(w * 0.05, w * 0.95, i / totalSteps);
        lctx.globalAlpha = 0.25;
        lctx.beginPath(); lctx.arc(hx, h * 0.5, 1.6, 0, Math.PI * 2); lctx.fill();
      }
      lctx.restore();
      lctx.save();
      lctx.fillStyle = C.blue;
      lctx.shadowColor = C.blue; lctx.shadowBlur = 8;
      lctx.beginPath(); lctx.arc(x, h * 0.5, 6, 0, Math.PI * 2); lctx.fill();
      lctx.restore();
    }

    function renderFlow(t) {
      const w = rightCanvas.clientWidth, h = rightCanvas.clientHeight;
      rctx.clearRect(0, 0, w, h);
      const jumpAt = 0.15;
      const jumped = t !== null && t > jumpAt;
      if (flowCountEl) flowCountEl.textContent = jumped ? "1" : "0";
      rctx.save();
      rctx.strokeStyle = C.muted; rctx.globalAlpha = 0.3;
      rctx.beginPath(); rctx.moveTo(w * 0.05, h * 0.5); rctx.lineTo(w * 0.95, h * 0.5); rctx.stroke();
      rctx.restore();
      const x = jumped ? w * 0.95 : w * 0.05;
      if (jumped && t < jumpAt + 0.4) {
        rctx.save();
        rctx.strokeStyle = C.yellow; rctx.lineWidth = 3; rctx.globalAlpha = 0.8;
        rctx.beginPath(); rctx.moveTo(w * 0.05, h * 0.5); rctx.lineTo(w * 0.95, h * 0.5); rctx.stroke();
        rctx.restore();
      }
      rctx.save();
      rctx.fillStyle = C.yellow;
      rctx.shadowColor = C.yellow; rctx.shadowBlur = 12;
      rctx.beginPath(); rctx.arc(x, h * 0.5, 6, 0, Math.PI * 2); rctx.fill();
      rctx.restore();
      if (jumped) {
        rctx.save();
        rctx.fillStyle = C.green;
        rctx.font = "13px Inter, sans-serif";
        rctx.fillText("done!", w * 0.82, h * 0.3);
        rctx.restore();
      }
    }

    const loop = U.makeLoop((dt) => {
      if (raceT !== null) {
        raceT += dt;
        if (raceT > 2.4 && !counterShown) {
          counterShown = true;
          animateCounter();
        }
      }
      renderEuler(raceT);
      renderFlow(raceT);
    });

    function animateCounter() {
      if (!bigNumberEl) return;
      const dur = 1.4, start = performance.now();
      function step() {
        const t = U.clamp((performance.now() - start) / 1000 / dur, 0, 1);
        const val = 1 + U.easeOutCubic(t) * (REAL_SPEEDUP - 1);
        bigNumberEl.textContent = val.toFixed(1) + "×";
        if (t < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }

    let wiredOnce = false;
    function wireOnce() {
      if (wiredOnce) return; wiredOnce = true;
      const raceBtn = section.querySelector('[data-role="race16-btn"]');
      if (raceBtn) raceBtn.addEventListener("click", () => {
        raceT = 0; counterShown = false;
        if (bigNumberEl) bigNumberEl.textContent = "1.0×";
      });
    }

    return { start() { wireOnce(); loop.start(); }, stop() { loop.stop(); } };
  };

  // ============================================================
  // Section 17 — runaway electrons / tokamak
  // ============================================================
  Scenes.s17 = function (section, canvas) {
    const { ctx } = U.setupCanvas(canvas);
    let ang = 0, nextJumpAt = 3, flashT = 0;

    const loop = U.makeLoop((dt, tt) => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      const cx = w * 0.5, cy = h * 0.5;
      const rx = Math.min(w, h) * 0.42, ry = Math.min(w, h) * 0.3;
      const sepScale = 0.62;

      // vessel wall
      ctx.save();
      ctx.strokeStyle = C.muted; ctx.globalAlpha = 0.5; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2); ctx.stroke();
      ctx.restore();

      // separatrix
      ctx.save();
      ctx.strokeStyle = C.orange; ctx.globalAlpha = 0.6; ctx.setLineDash([5, 6]); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.ellipse(cx, cy, rx * sepScale, ry * sepScale, 0, 0, Math.PI * 2); ctx.stroke();
      ctx.restore();

      ang += dt * 0.9;
      let radiusFrac = 0.78 + Math.sin(tt * 2.3) * 0.03;
      let jumped = false;
      if (tt > nextJumpAt) {
        radiusFrac = 0.3;
        jumped = true;
        flashT = 1;
        nextJumpAt = tt + 3 + Math.random() * 2;
      }
      const px = cx + Math.cos(ang) * rx * radiusFrac;
      const py = cy + Math.sin(ang) * ry * radiusFrac;

      flashT = Math.max(0, flashT - dt * 0.8);
      if (flashT > 0) {
        ctx.save();
        ctx.fillStyle = C.orange;
        ctx.globalAlpha = flashT * 0.5;
        ctx.font = "13px Inter, sans-serif";
        ctx.fillText("↯ jump across separatrix", cx - 70, cy - ry - 14);
        ctx.restore();
      }

      ctx.save();
      ctx.fillStyle = jumped ? C.orange : C.blue;
      ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    });

    return { start() { loop.start(); }, stop() { loop.stop(); } };
  };

  return Scenes;
})();
