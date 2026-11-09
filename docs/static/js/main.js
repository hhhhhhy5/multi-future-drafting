/* Page behaviour: external links, theme toggle, BibTeX copy, animation controls. */
(function () {
  "use strict";

  // Fill these in once the release is public. Empty string = button shown as "soon".
  var LINKS = {
    arxiv: "",                       // e.g. "https://arxiv.org/abs/2611.xxxxx"
    code: "",                        // e.g. "https://github.com/ORG/multi-future-drafting"
    models: "",                      // e.g. "https://huggingface.co/collections/ORG/..."
    modelBase: ""                    // e.g. "https://huggingface.co/ORG/" + name
  };

  // --- buttons -------------------------------------------------------------
  document.querySelectorAll("#link-buttons a[data-link]").forEach(function (a) {
    var key = a.getAttribute("data-link");
    var url = LINKS[key];
    if (url) {
      a.href = url;
      if (/^https?:/.test(url)) { a.target = "_blank"; a.rel = "noopener"; }
    } else {
      a.classList.add("soon");
      a.removeAttribute("href");
      a.setAttribute("aria-disabled", "true");
      a.title = "Available once the release is public";
    }
  });
  document.querySelectorAll("a[data-model]").forEach(function (a) {
    if (LINKS.modelBase) {
      a.href = LINKS.modelBase + a.getAttribute("data-model");
      a.target = "_blank"; a.rel = "noopener";
    } else {
      a.removeAttribute("href");
      a.style.color = "inherit";
      a.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      a.style.fontSize = ".88em";
    }
  });
  if (LINKS.code && LINKS.modelBase) {
    var note = document.getElementById("links-note");
    if (note) note.remove();
  }

  // --- theme ---------------------------------------------------------------
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem("mfd-theme");
    if (saved === "dark" || saved === "light") root.setAttribute("data-theme", saved);
  } catch (e) { /* storage may be unavailable */ }
  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var dark = root.getAttribute("data-theme") === "dark" ||
        (!root.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
      var next = dark ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("mfd-theme", next); } catch (e) { /* ignore */ }
    });
  }

  // --- BibTeX copy -----------------------------------------------------------
  var copy = document.getElementById("copy-bib");
  if (copy) {
    copy.addEventListener("click", function () {
      var text = document.getElementById("bib-text").textContent;
      var done = function () { copy.textContent = "Copied"; setTimeout(function () { copy.textContent = "Copy"; }, 1600); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(); });
      } else { fallback(); }
      function fallback() {
        var r = document.createRange(); r.selectNodeContents(document.getElementById("bib-text"));
        var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
        try { document.execCommand("copy"); done(); } catch (e) { /* ignore */ }
      }
    });
  }

  // --- animation controls ---------------------------------------------------
  var svg = document.getElementById("mfd-anim");
  if (!svg || !window.MFDAnimation) return;
  var anim = window.MFDAnimation.mount(svg, { autoplay: false });
  var playBtn = document.getElementById("anim-play");
  var restartBtn = document.getElementById("anim-restart");
  var scenesBox = document.getElementById("anim-scenes");
  var progress = document.getElementById("anim-progress");
  var bar = document.getElementById("anim-progress-bar");
  var speed = document.getElementById("anim-speed");
  var sceneButtons = [];

  anim.scenes.forEach(function (s, i) {
    var b = document.createElement("button");
    b.type = "button";
    b.textContent = (i + 1) + " " + s.name;
    b.addEventListener("click", function () { anim.seek(s.t); anim.play(); });
    scenesBox.appendChild(b);
    sceneButtons.push(b);
  });

  playBtn.addEventListener("click", function () { if (anim.playing()) anim.pause(); else anim.play(); });
  restartBtn.addEventListener("click", function () { anim.seek(0); anim.play(); });
  speed.addEventListener("change", function () { anim.setSpeed(parseFloat(speed.value)); });
  progress.addEventListener("click", function (ev) {
    var r = progress.getBoundingClientRect();
    anim.seek(anim.duration * (ev.clientX - r.left) / r.width);
  });

  anim.onFrame(function (t) {
    bar.style.width = (100 * t / anim.duration).toFixed(2) + "%";
    playBtn.textContent = anim.playing() ? "Pause" : "Play";
    var cur = 0;
    anim.scenes.forEach(function (s, i) { if (t >= s.t - 1e-6) cur = i; });
    sceneButtons.forEach(function (b, i) { b.classList.toggle("on", i === cur); });
  });

  // Play while visible, pause when scrolled away.
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { if (!anim.userPaused()) anim.play(); }
        else anim.pause(true);
      });
    }, { threshold: 0.35 });
    io.observe(svg);
  } else {
    anim.play();
  }
})();
