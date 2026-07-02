(function () {
  const tabs = document.querySelectorAll(".tab-btn");
  const decks = document.querySelectorAll(".deck");
  const counter = document.getElementById("slide-counter");
  const dotsWrap = document.getElementById("dots");
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = lightbox.querySelector("img");

  let activeDeck = document.querySelector(".deck.active");

  function buildDots(deck) {
    dotsWrap.innerHTML = "";
    const slides = deck.querySelectorAll(".slide");
    slides.forEach((s, i) => {
      const d = document.createElement("div");
      d.className = "dot";
      d.title = s.dataset.title || `Slide ${i + 1}`;
      d.addEventListener("click", () => s.scrollIntoView({ behavior: "smooth" }));
      dotsWrap.appendChild(d);
    });
    updateCounter(deck);
  }

  function updateCounter(deck) {
    const slides = [...deck.querySelectorAll(".slide")];
    const container = deck.querySelector(".slides");
    const scrollTop = container.scrollTop;
    let idx = 0;
    slides.forEach((s, i) => {
      if (s.offsetTop - container.offsetTop <= scrollTop + 10) idx = i;
    });
    counter.textContent = `${idx + 1} / ${slides.length}`;
    dotsWrap.querySelectorAll(".dot").forEach((d, i) => d.classList.toggle("active", i === idx));
  }

  function activateDeck(name) {
    decks.forEach((d) => d.classList.toggle("active", d.dataset.deck === name));
    tabs.forEach((t) => t.classList.toggle("active", t.dataset.deck === name));
    activeDeck = document.querySelector(`.deck[data-deck="${name}"]`);
    buildDots(activeDeck);
    localStorage.setItem("activeDeck", name);
  }

  tabs.forEach((t) => t.addEventListener("click", () => activateDeck(t.dataset.deck)));

  decks.forEach((deck) => {
    deck.querySelector(".slides").addEventListener("scroll", () => {
      if (deck.classList.contains("active")) updateCounter(deck);
    });
  });

  document.addEventListener("keydown", (e) => {
    if (lightbox.classList.contains("active")) {
      if (e.key === "Escape") closeLightbox();
      return;
    }
    const container = activeDeck.querySelector(".slides");
    const slides = [...activeDeck.querySelectorAll(".slide")];
    const scrollTop = container.scrollTop;
    let idx = 0;
    slides.forEach((s, i) => {
      if (s.offsetTop - container.offsetTop <= scrollTop + 10) idx = i;
    });
    if (e.key === "ArrowDown" || e.key === "PageDown" || e.key === " ") {
      e.preventDefault();
      slides[Math.min(idx + 1, slides.length - 1)].scrollIntoView({ behavior: "smooth" });
    } else if (e.key === "ArrowUp" || e.key === "PageUp") {
      e.preventDefault();
      slides[Math.max(idx - 1, 0)].scrollIntoView({ behavior: "smooth" });
    } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      const other = [...tabs].find((t) => !t.classList.contains("active"));
      if (other) activateDeck(other.dataset.deck);
    }
  });

  document.querySelectorAll(".figcard img").forEach((img) => {
    img.addEventListener("click", () => {
      lightboxImg.src = img.src;
      lightbox.classList.add("active");
    });
  });

  function closeLightbox() {
    lightbox.classList.remove("active");
    lightboxImg.src = "";
  }
  lightbox.addEventListener("click", closeLightbox);

  // init
  const saved = localStorage.getItem("activeDeck") || "comparison";
  activateDeck(saved);
})();
