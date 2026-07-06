(() => {
  const sections = Array.from(document.querySelectorAll("main > section"));

  // ---------------- progress bar ----------------
  const progressBar = document.getElementById("progress-bar");
  function updateProgress() {
    const h = document.documentElement;
    const scrollable = h.scrollHeight - h.clientHeight;
    const pct = scrollable > 0 ? (h.scrollTop / scrollable) * 100 : 0;
    if (progressBar) progressBar.style.width = pct + "%";
  }
  document.addEventListener("scroll", updateProgress, { passive: true });
  updateProgress();

  // ---------------- nav dots ----------------
  const dotsWrap = document.getElementById("nav-dots");
  if (dotsWrap) {
    sections.forEach((s) => {
      const dot = document.createElement("div");
      dot.className = "dot";
      dot.dataset.target = s.id;
      const tip = document.createElement("div");
      tip.className = "tip";
      tip.textContent = s.dataset.nav || s.id;
      dot.appendChild(tip);
      dot.addEventListener("click", () => s.scrollIntoView({ behavior: "smooth" }));
      dotsWrap.appendChild(dot);
    });
  }

  const navObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const dot = dotsWrap && dotsWrap.querySelector(`[data-target="${entry.target.id}"]`);
        if (dot) {
          dotsWrap.querySelectorAll(".dot").forEach((d) => d.classList.remove("active"));
          dot.classList.add("active");
        }
      }
    });
  }, { threshold: 0.5 });
  sections.forEach((s) => navObserver.observe(s));

  // ---------------- generic reveal-on-scroll ----------------
  const revealTargets = document.querySelectorAll(".reveal, .eq:not([data-manual])");
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.25 });
  revealTargets.forEach((el) => revealObserver.observe(el));

  // ---------------- staggered ablation-panel reveal (section 14) ----------------
  document.querySelectorAll(".panels").forEach((panelsWrap) => {
    const cards = Array.from(panelsWrap.querySelectorAll(".panelcard"));
    const panelObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          cards.forEach((card, i) => {
            setTimeout(() => card.classList.add("visible"), i * 180);
          });
          panelObserver.disconnect();
        }
      });
    }, { threshold: 0.2 });
    panelObserver.observe(panelsWrap);
  });

  // ---------------- roadmap smooth-scroll ----------------
  document.querySelectorAll(".roadmap .rstep[data-target]").forEach((el) => {
    el.addEventListener("click", () => {
      const target = document.getElementById(el.dataset.target);
      if (target) target.scrollIntoView({ behavior: "smooth" });
    });
  });

  // ---------------- lightbox ----------------
  const lightbox = document.getElementById("lightbox");
  if (lightbox) {
    const lightboxImg = lightbox.querySelector("img");
    document.querySelectorAll(".zoomable").forEach((img) => {
      img.addEventListener("click", () => {
        lightboxImg.src = img.src;
        lightbox.classList.add("active");
      });
    });
    lightbox.addEventListener("click", () => {
      lightbox.classList.remove("active");
      lightboxImg.src = "";
    });
  }

  // ---------------- scene lifecycle (canvas animations) ----------------
  const instances = {};
  sections.forEach((s) => {
    const factory = window.Scenes && window.Scenes[s.id];
    if (!factory) return;
    const canvas = s.querySelector(".scene canvas");
    try {
      instances[s.id] = factory(s, canvas);
    } catch (err) {
      console.error(`Scene ${s.id} failed to initialize`, err);
    }
  });

  const sceneObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const inst = instances[entry.target.id];
      if (!inst) return;
      if (entry.isIntersecting) inst.start();
      else inst.stop();
    });
  }, { threshold: 0.1 });
  sections.forEach((s) => { if (instances[s.id]) sceneObserver.observe(s); });
})();
