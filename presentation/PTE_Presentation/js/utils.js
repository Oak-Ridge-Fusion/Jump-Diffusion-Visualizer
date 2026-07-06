// Shared math + canvas helpers used by every scene in scenes.js
const Util = (() => {
  function randn() {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function easeInOutCubic(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

  // Sets up a canvas for crisp rendering at devicePixelRatio, sized to its
  // parent's CSS box. Returns {ctx, resize} — call resize() on layout change.
  function setupCanvas(canvas) {
    const ctx = canvas.getContext("2d");
    function resize() {
      const rect = canvas.parentElement.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);
    return { ctx, resize, get width() { return canvas.getBoundingClientRect().width; }, get height() { return canvas.getBoundingClientRect().height; } };
  }

  function drawArrow(ctx, x1, y1, x2, y2, color, width = 2) {
    const headlen = 8;
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headlen * Math.cos(angle - Math.PI / 6), y2 - headlen * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - headlen * Math.cos(angle + Math.PI / 6), y2 - headlen * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // Histogram of `values` into `bins` buckets across [lo, hi], drawn as bars
  // inside the rect (x, y, w, h), bars growing upward from y+h.
  function drawHistogram(ctx, values, bins, lo, hi, x, y, w, h, color) {
    const counts = new Array(bins).fill(0);
    const span = hi - lo;
    for (const v of values) {
      let idx = Math.floor(((v - lo) / span) * bins);
      idx = clamp(idx, 0, bins - 1);
      counts[idx]++;
    }
    const maxCount = Math.max(1, ...counts);
    const barW = w / bins;
    ctx.save();
    ctx.fillStyle = color;
    for (let i = 0; i < bins; i++) {
      const bh = (counts[i] / maxCount) * h;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x + i * barW + 1, y + h - bh, barW - 2, bh);
    }
    ctx.restore();
    return counts;
  }

  function gaussianPdf(x, mu, sigma) {
    return Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
  }

  class Particle {
    constructor(x, y) {
      this.x = x; this.y = y;
      this.homeX = x; this.homeY = y;
      this.alive = true;
      this.trail = [];
    }
  }

  // Runs cb(dt, t) every frame while active; caller controls start/stop.
  function makeLoop(cb) {
    let raf = null, last = null, t0 = null, running = false;
    function frame(ts) {
      if (!running) return;
      if (last === null) { last = ts; t0 = ts; }
      const dt = Math.min((ts - last) / 1000, 0.05);
      last = ts;
      cb(dt, (ts - t0) / 1000);
      raf = requestAnimationFrame(frame);
    }
    return {
      start() { if (running) return; running = true; last = null; raf = requestAnimationFrame(frame); },
      stop() { running = false; if (raf) cancelAnimationFrame(raf); raf = null; },
      get isRunning() { return running; }
    };
  }

  return { randn, clamp, lerp, easeInOutCubic, easeOutCubic, setupCanvas, drawArrow, drawHistogram, gaussianPdf, Particle, makeLoop };
})();
