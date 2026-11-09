/* The highlight animation: a product-ranked draft tree splices two futures into chimeras and accepts 2 tokens;
 * MFD's mixture tree keeps the futures apart and accepts 6. Everything is a pure function of the time t, so the
 * same code drives the page (requestAnimationFrame) and the video export (seek + screenshot per frame).
 *
 * Data: the real GSM8K context of the paper's Figure 1 (Qwen3-4B, conversation 25, anchor 89 of
 * eval_results/teaser/candidates_gsm8k.json and example_modes.json): the one-pass drafter's per-position
 * probabilities, DDTree's 15-node tree in best-first order, the eight modes' argmax chains with each token's
 * probability, and the 15-node mixture tree in best-first order with, per node, the modes whose own score for
 * that prefix reaches the common cut-off (the 15th best score; Proposition 1 of the paper).
 *
 * API: MFDAnimation.mount(svgElement, {autoplay}) -> {scenes, duration, times, seek, play, pause, playing,
 * userPaused, setSpeed, onFrame, render}.
 */
(function (global) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var W = 960, H = 540;
  var X0 = 176, PITCH = 76, CW = 70, CH = 24;   // token columns: position i sits at X0 + i * PITCH
  var FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
  var SERIF = "Tinos, 'Times New Roman', Times, serif";
  var C = {
    blue: "#2563a8", blueL: "#dbe8f6", orange: "#d9731a", orangeL: "#fbe6d3", red: "#c62828", redL: "#fde7e7",
    green: "#1b7a3d", greenL: "#dff3e6", purple: "#6b3fa0", ink: "#1d1f23", muted: "#5b6068", faint: "#8a8f98",
    rule: "#c9cdd3", grayL: "#eceef1", gray: "#8a8f98", white: "#ffffff"
  };

  var DATA = {
    context: "… how much profit does he make on his car?",
    prefix: "Let’s break",
    A: ["this", "down", "step", "by", "step", "to", "find", "out"],
    B: ["down", "the", "problem", "step", "by", "step", "to", "determine", "how"],
    pA: 0.61, pB: 0.37,
    // the one-pass drafter: probability of A's token and of B's token at each position
    marg: [[0.517, 0.456], [0.221, 0.772], [0.179, 0.804], [0.240, 0.738], [0.812, 0.181], [0.920, 0.067]],
    // DDTree's tree (B = 15), in the order the best-first search creates the nodes
    product: [
      ["this"], ["down"], ["this", "the"], ["down", "the"], ["this", "the", "problem"], ["down", "the", "problem"],
      ["this", "the", "problem", "step"], ["down", "the", "problem", "step"],
      ["this", "the", "problem", "step", "step"], ["this", "the", "problem", "step", "step", "to"],
      ["down", "the", "problem", "step", "step"], ["down", "the", "problem", "step", "step", "to"],
      ["this", "down"], ["down", "down"], ["this", "down", "problem"]
    ],
    // the mixture tree (B = 15): the 15 best prefixes over all modes, each scored by its best mode
    mixture: [
      ["down"], ["down", "the"], ["down", "the", "problem"], ["down", "the", "problem", "step"],
      ["down", "the", "problem", "step", "by"], ["down", "the", "problem", "step", "by", "step"],
      ["down", "the", "problem", "step", "by", "step", "to"],
      ["this"], ["this", "down"], ["this", "down", "step"], ["this", "down", "step", "by"],
      ["this", "down", "step", "by", "step"],
      ["down", "the", "problem", "step", "by", "step", "to", "determine"],
      ["this", "down", "step", "by", "step", "to"],
      ["down", "the", "problem", "step", "by", "step", "to", "determine", "how"]
    ],
    // the eight modes' argmax chains (first seven positions)
    modes: [
      ["down", "the", "problem", "step", "by", "step", "to"],
      ["this", "down", "step", "by", "step", "to", "determine"],
      ["this", "down", "step", "by", "step", "to", "---"],
      ["down", "the", "problem", "step", "by", "step", "to"],
      ["down", "the", "problem", "by", "step", "step", "to"],
      ["down", "the", "problem", "step", "by", "step", "to"],
      ["this", "down", "step", "by", "by", "step", "to"],
      ["down", "the", "problem", "step", "by", "step", "to"]
    ],
    // each mode's probability of the tokens above (positions 1-7)
    modeP: [
      [0.9959, 0.9626, 0.9974, 0.9986, 0.9949, 0.9988, 0.9991],
      [0.8340, 0.9730, 0.9115, 0.9730, 0.9949, 0.9979, 0.6465],
      [0.9341, 0.9898, 0.9957, 0.9900, 0.9996, 0.5821, 0.5864],
      [0.9783, 0.9680, 0.9932, 0.9953, 0.9500, 0.9997, 0.9989],
      [0.8536, 0.8157, 0.8795, 0.5090, 0.5555, 0.5287, 0.9907],
      [0.9948, 0.9994, 0.9967, 0.9994, 0.9736, 0.9994, 0.9959],
      [0.7481, 0.9748, 0.7750, 0.7424, 0.6745, 0.5606, 0.9277],
      [0.6447, 0.9034, 0.8655, 0.7225, 0.7695, 0.9380, 0.9661]
    ],
    // the common cut-off (score of the 15th prefix), how many prefixes of its own block each mode keeps at or
    // above it (mode 6's run to position 9, beyond the seven shown), and the product of each mode's deepest kept
    // prefix; keep sums to 37, the 15 distinct prefixes are the tree
    theta: 0.698,
    keep: [7, 6, 5, 7, 1, 9, 2, 0],
    keepQ: [0.948, 0.715, 0.911, 0.888, 0.854, 0.698, 0.729, 0],
    // per mixture-tree node (same order as `mixture`): the modes whose score for the prefix reaches the cut-off
    mixtureSup: [
      [1, 4, 5, 6], [1, 4, 6], [1, 4, 6], [1, 4, 6], [1, 4, 6], [1, 4, 6], [1, 4, 6],
      [2, 3, 7], [2, 3, 7], [2, 3], [2, 3], [2, 3],
      [6], [2], [6]
    ]
  };

  // Timeline (seconds).
  var T = {};
  T.ctx = 0.0; T.futA = 1.0; T.futB = 1.9;
  T.s1 = 3.4; T.bars = 4.2; T.barStep = 0.42; T.path = 7.6; T.pathDur = 1.6; T.cap1 = 8.9;
  T.tree1 = 10.2; T.nodeStep = 0.27; T.ver1 = 15.0; T.s1end = 17.3;
  T.s2 = 17.7; T.modes = 18.4; T.modeStep = 0.26; T.cut = 21.6; T.cap2 = 22.9;
  T.tree2 = 23.6; T.nodeStep2 = 0.42; T.fly = 0.34; T.ver2 = 30.4; T.s2end = 34.1;
  T.s3 = 34.5; T.end = 41.6;

  var SCENES = [
    { name: "Two futures", t: 0 },
    { name: "Product tree", t: T.s1 },
    { name: "MFD", t: T.s2 },
    { name: "Summary", t: T.s3 }
  ];

  // ---- helpers ---------------------------------------------------------------------------------------------
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function easeOut(v) { return 1 - Math.pow(1 - v, 3); }
  function ez(t, t0, d) { return easeOut(clamp01((t - t0) / (d || 0.4))); }
  function lin(t, t0, d) { return clamp01((t - t0) / d); }
  function mk(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs.hasOwnProperty(k) && attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, s, o) {
    o = o || {};
    var e = mk("text", {
      x: x, y: y, "font-size": o.size || 13, fill: o.fill || C.ink, "font-family": o.serif ? SERIF : FONT,
      "font-weight": o.weight || 400, "text-anchor": o.anchor || "start",
      "dominant-baseline": o.baseline || "middle",
      "font-style": o.italic ? "italic" : null, "letter-spacing": o.spacing || null
    }, parent);
    e.textContent = s;
    return e;
  }
  // Texts registered with fit() shrink their font until they fit maxW (re-run once the web fonts are in).
  var FITS = [];
  function fit(e, maxW) { FITS.push([e, maxW, parseFloat(e.getAttribute("font-size"))]); return e; }
  function fitAll() {
    FITS.forEach(function (f) {
      var e = f[0], maxW = f[1], base = f[2], len;
      e.setAttribute("font-size", base);
      try { len = e.getComputedTextLength(); } catch (err) { return; }
      if (len > maxW && len > 0) e.setAttribute("font-size", (base * maxW / len).toFixed(2));
    });
  }
  function setOp(e, v) {
    e.setAttribute("opacity", v.toFixed(3));
    e.style.visibility = v <= 0.002 ? "hidden" : "visible";
  }
  function prefixOf(p, arr) { for (var i = 0; i < p.length; i++) if (arr[i] !== p[i]) return false; return true; }
  function status(p) { return prefixOf(p, DATA.A) ? "A" : prefixOf(p, DATA.B) ? "B" : "X"; }
  function style(kind) {
    if (kind === "A") return { fill: C.blueL, stroke: C.blue, text: C.ink, dash: null };
    if (kind === "B") return { fill: C.orangeL, stroke: C.orange, text: C.ink, dash: null };
    if (kind === "X") return { fill: C.redL, stroke: C.red, text: C.red, dash: "4 3" };
    return { fill: C.grayL, stroke: C.gray, text: C.muted, dash: null };
  }
  function colX(i) { return X0 + i * PITCH; }
  function pct(p) { var n = Math.round(p * 100); return n >= 100 ? "1.0" : "." + (n < 10 ? "0" : "") + n; }
  function pct3(p) { var n = Math.round(p * 1000); return n >= 1000 ? "1.00" : "." + (n < 10 ? "00" : n < 100 ? "0" : "") + n; }

  // A token cell centred at (cx, cy); the group is transformed, so it can pop and slide.
  function cell(parent, cx, cy, w, h, token, kind, o) {
    o = o || {};
    var st = style(kind);
    var g = mk("g", { transform: "translate(" + cx + "," + cy + ")" }, parent);
    var r = mk("rect", {
      x: -w / 2, y: -h / 2, width: w, height: h, rx: 5, ry: 5, fill: st.fill, stroke: st.stroke,
      "stroke-width": o.strokeWidth || 1.2, "stroke-dasharray": st.dash
    }, g);
    var l = text(g, 0, 1, token, { serif: true, size: o.size || 13.5, fill: st.text, anchor: "middle", weight: o.weight || 400 });
    var out = { g: g, rect: r, label: l, cx: cx, cy: cy, w: w, h: h, kind: kind };
    out.place = function (s, dx, dy) {
      g.setAttribute("transform", "translate(" + (cx + (dx || 0)) + "," + (cy + (dy || 0)) + ") scale(" + s + ")");
    };
    return out;
  }

  // "the target says A; accepted  N  drafted tokens" on one baseline, the number large.
  function acceptedLine(parent, x, y, n, color) {
    var e = mk("text", { x: x, y: y, "font-family": FONT, "font-size": 13.5, fill: C.muted, "dominant-baseline": "alphabetic" }, parent);
    var a = mk("tspan", {}, e); a.textContent = "the target says A;  accepted ";
    var b = mk("tspan", { "font-size": 44, "font-weight": 800, fill: color, dx: 6 }, e); b.textContent = String(n);
    var c = mk("tspan", { dx: 8 }, e); c.textContent = "drafted tokens this round";
    return e;
  }

  // ---- the scene ------------------------------------------------------------------------------------------
  function build(svg) {
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.style.fontFamily = FONT;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    mk("rect", { x: 0, y: 0, width: W, height: H, fill: C.white }, svg);

    var U = [];   // update functions of t

    // ---------- persistent header: context and the two futures ----------
    var hd = mk("g", {}, svg);
    var ctxLabel = text(hd, 20, 24, "Context:", { weight: 700, size: 13.5 });
    var ctxText = fit(text(hd, 88, 24, DATA.context, { italic: true, fill: C.muted, size: 13.5, serif: true }), 275);
    var prefixText = text(hd, 372, 24, DATA.prefix, { weight: 700, size: 15, serif: true });
    var cursor = mk("rect", { x: 455, y: 15, width: 7, height: 18, fill: C.gray }, hd);
    var brand = text(hd, 940, 24, "Multi-Future Drafting (MFD)", { fill: C.faint, size: 11.5, anchor: "end", weight: 600, spacing: ".02em" });
    var posLabel = text(hd, 20, 50, "next tokens", { fill: C.faint, size: 11 });
    var posNums = [];
    for (var i = 0; i < 9; i++) posNums.push(text(hd, colX(i) + CW / 2, 50, String(i + 1), { fill: C.faint, size: 11, anchor: "middle" }));
    var yA = 72, yB = 100;
    var labA = text(hd, 20, yA, "target says A", { weight: 700, fill: C.blue, size: 13.5 });
    var pA = text(hd, 122, yA, "p = " + DATA.pA.toFixed(2), { fill: C.blue, size: 12 });
    var labB = text(hd, 113, yB, "or B", { weight: 700, fill: C.orange, size: 13.5, anchor: "end" });
    var pB = text(hd, 122, yB, "p = " + DATA.pB.toFixed(2), { fill: C.orange, size: 12 });
    var cellsA = DATA.A.map(function (tok, i) { return cell(hd, colX(i) + CW / 2, yA, CW, CH, tok, "A"); });
    var cellsB = DATA.B.map(function (tok, i) { return cell(hd, colX(i) + CW / 2, yB, CW, CH, tok, "B"); });
    var rule = mk("line", { x1: 20, y1: 118, x2: 940, y2: 118, stroke: C.rule, "stroke-width": 1 }, hd);
    var foot = fit(text(svg, 20, 527, "GSM8K context, Qwen3-4B.  (a) D-PACE drafter + DDTree, B = 15.  (b) The same drafter fine-tuned as MFD, K = 8, mixture tree, B = 15.  Trees and mode probabilities from the paper’s Figure 1 example.",
      { fill: C.faint, size: 10.5 }), 900);

    U.push(function (t) {
      var k = ez(t, T.ctx, 0.6);
      [ctxLabel, ctxText, prefixText].forEach(function (e) { setOp(e, k); });
      setOp(cursor, k * ((Math.floor(t * 2) % 2 === 0) ? 1 : 0.25));
      setOp(brand, k);
      var kp = ez(t, T.futA - 0.3, 0.4);
      setOp(posLabel, kp); posNums.forEach(function (e) { setOp(e, kp); });
      setOp(labA, ez(t, T.futA, 0.4)); setOp(pA, ez(t, T.futA + 0.2, 0.4));
      setOp(labB, ez(t, T.futB, 0.4)); setOp(pB, ez(t, T.futB + 0.2, 0.4));
      cellsA.forEach(function (c, i) { var q = ez(t, T.futA + 0.1 + i * 0.07, 0.35); setOp(c.g, q); c.place(0.7 + 0.3 * q); });
      cellsB.forEach(function (c, i) { var q = ez(t, T.futB + 0.1 + i * 0.07, 0.35); setOp(c.g, q); c.place(0.7 + 0.3 * q); });
      setOp(rule, ez(t, T.futB + 0.6, 0.5));
      setOp(foot, ez(t, T.s1, 0.8));
    });

    // ---------- scene 1: the one-pass drafter and the product tree ----------
    var g1 = mk("g", {}, svg);
    text(g1, 20, 146, "(a) One-pass drafter + DDTree", { weight: 700, size: 17 });
    text(g1, 20, 167, "one distribution per position; prefixes ranked by their product, the 15 best verified", { fill: C.muted, size: 12.5 });

    var yD1 = 198, yD2 = 232, DH = 30;
    var dist = DATA.marg.map(function (pq, i) {
      var g = mk("g", {}, g1);
      var ca = cell(g, colX(i) + CW / 2, yD1, CW, DH, DATA.A[i] + "  " + pct(pq[0]), "A", { size: 12.5 });
      var cb = cell(g, colX(i) + CW / 2, yD2, CW, DH, DATA.B[i] + "  " + pct(pq[1]), "B", { size: 12.5 });
      [ca, cb].forEach(function (c) { c.label.setAttribute("y", -2); });
      var ba = mk("rect", { x: -CW / 2 + 5, y: DH / 2 - 7, width: 0, height: 4, rx: 1, fill: C.blue }, ca.g);
      var bb = mk("rect", { x: -CW / 2 + 5, y: DH / 2 - 7, width: 0, height: 4, rx: 1, fill: C.orange }, cb.g);
      return { g: g, a: ca, b: cb, ba: ba, bb: bb, pa: pq[0], pb: pq[1], best: pq[0] >= pq[1] ? "A" : "B" };
    });
    var distLab = text(g1, X0 - 10, (yD1 + yD2) / 2, "drafter", { fill: C.muted, size: 11.5, anchor: "end" });
    var pathPts = dist.map(function (d, i) { return (colX(i) + CW / 2) + "," + (d.best === "A" ? yD1 : yD2); });
    var path = mk("polyline", { points: pathPts.join(" "), fill: "none", stroke: C.red, "stroke-width": 3, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: 0.9 }, g1);
    var pathLen = 0;
    var cap1a = fit(text(g1, X0, 276, "DDTree’s best path “this the problem step step to” jumps between A and B: a chimera the target never says.", { fill: C.red, size: 12.5 }), 760);
    var cap1b = text(g1, X0, 304, "The 15 best prefixes under the product form the tree. Best first:", { fill: C.muted, size: 12.5 });

    // the product tree
    var rowsY = [334, 372, 410, 448];
    function rowOf1(p) {
      if (p.length === 1) return p[0] === "this" ? (rowsY[0] + rowsY[1]) / 2 : (rowsY[2] + rowsY[3]) / 2;
      if (p[0] === "this") return p[1] === "down" ? rowsY[0] : rowsY[1];
      return p[1] === "down" ? rowsY[2] : rowsY[3];
    }
    var tree1 = buildTree(g1, DATA.product, rowOf1, (rowsY[1] + rowsY[2]) / 2);
    var cnt1n = text(g1, 940, 334, "", { fill: C.muted, size: 12.5, anchor: "end" });
    var cnt1c = text(g1, 940, 354, "", { fill: C.red, size: 12.5, anchor: "end", weight: 600 });
    var ghost1 = cell(g1, colX(2) + CW / 2, rowsY[0], CW, CH, "step", "X", { strokeWidth: 1.4 });
    var ghost1t = text(g1, colX(3) + 2, rowsY[0], "not in the tree", { fill: C.red, size: 11.5 });
    var acc1 = acceptedLine(g1, X0, 503, 2, C.red);

    U.push(function (t) {
      var vis = ez(t, T.s1, 0.5) * (1 - ez(t, T.s1end, 0.4));
      setOp(g1, vis);
      if (vis <= 0) return;
      setOp(distLab, ez(t, T.bars, 0.4));
      dist.forEach(function (d, i) {
        var q = ez(t, T.bars + i * T.barStep, 0.45);
        setOp(d.g, q);
        d.a.place(0.75 + 0.25 * q); d.b.place(0.75 + 0.25 * q);
        d.ba.setAttribute("width", ((CW - 10) * d.pa * q).toFixed(2));
        d.bb.setAttribute("width", ((CW - 10) * d.pb * q).toFixed(2));
      });
      if (!pathLen) { try { pathLen = path.getTotalLength(); } catch (e) { pathLen = 0; } }
      var kp = lin(t, T.path, T.pathDur);
      if (pathLen) { path.setAttribute("stroke-dasharray", pathLen); path.setAttribute("stroke-dashoffset", (pathLen * (1 - kp)).toFixed(2)); }
      setOp(path, kp > 0 ? 0.9 : 0);
      dist.forEach(function (d, i) {
        var on = kp > 0 && kp >= i / 5 - 1e-6;
        var c = d.best === "A" ? d.a : d.b;
        c.rect.setAttribute("stroke", on ? C.red : style(c.kind).stroke);
        c.rect.setAttribute("stroke-width", on ? 2.2 : 1.2);
      });
      setOp(cap1a, ez(t, T.cap1, 0.5));
      setOp(cap1b, ez(t, T.tree1 - 0.5, 0.5));
      var shown = tree1.update(t, T.tree1, T.nodeStep);
      cnt1n.textContent = shown.n ? "nodes  " + shown.n : "";
      cnt1c.textContent = shown.n ? "chimeras  " + shown.x : "";
      setOp(cnt1n, shown.n ? 1 : 0); setOp(cnt1c, shown.n ? 1 : 0);
      // verification: the target says A; the tree holds "this" and "this down", not "this down step"
      tree1.mark(["this"], ez(t, T.ver1 + 0.2, 0.3));
      tree1.mark(["this", "down"], ez(t, T.ver1 + 0.7, 0.3));
      var kg = ez(t, T.ver1 + 1.25, 0.35);
      setOp(ghost1.g, kg); ghost1.place(0.8 + 0.2 * kg); setOp(ghost1t, ez(t, T.ver1 + 1.45, 0.3));
      var ka = ez(t, T.ver1 + 1.75, 0.45);
      setOp(acc1, ka);
      acc1.setAttribute("transform", "translate(0," + (8 * (1 - ka)).toFixed(2) + ")");
    });

    // ---------- scene 2: MFD ----------
    var g2 = mk("g", {}, svg);
    text(g2, 20, 146, "(b) Multi-Future Drafting", { weight: 700, size: 17 });
    fit(text(g2, 20, 167, "8 mode embeddings at the drafter’s input, 8 blocks in one forward pass. Row = a mode’s most likely token per position, bar = that token’s probability", { fill: C.muted, size: 12.5 }), 920);
    var MH = 18, MP = 21, yM0 = 194, xDots = colX(6) + CW + 6, xTag = colX(7) + 16;
    var hdrP = text(g2, 940, 180, "product of the 7 tokens", { fill: C.faint, size: 11, anchor: "end" });
    var hdrK = text(g2, 940, 180, "prefixes kept  ·  product of the deepest", { fill: C.faint, size: 11, anchor: "end" });
    var modeRows = DATA.modes.map(function (chain, m) {
      var g = mk("g", {}, g2);
      var y = yM0 + m * MP, ps = DATA.modeP[m], prod = 1, keep = DATA.keep[m];
      text(g, X0 - 10, y, "mode " + (m + 1), { fill: C.muted, size: 11.5, anchor: "end" });
      var cells = chain.map(function (tok, i) {
        var kind = tok === DATA.A[i] ? "A" : tok === DATA.B[i] ? "B" : "G";
        var c = cell(g, colX(i) + CW / 2, y, CW, MH, tok, kind, { size: 12 });
        c.label.setAttribute("y", -2);
        mk("rect", { x: -CW / 2 + 5, y: MH / 2 - 4.5, width: ((CW - 10) * ps[i]).toFixed(2), height: 3, rx: 1, fill: style(kind).stroke, opacity: 0.8 }, c.g);
        prod *= ps[i];
        return c;
      });
      var fol = prefixOf(chain.slice(0, 6), DATA.A) ? "A" : prefixOf(chain.slice(0, 6), DATA.B) ? "B" : "G";
      var col = fol === "A" ? C.blue : fol === "B" ? C.orange : C.faint;
      var dots = text(g, xDots, y, "…", { fill: C.faint, size: 13, serif: true });
      var tag = text(g, xTag, y, fol === "A" ? "follows A" : fol === "B" ? "follows B" : "neither", { fill: col, size: 11.5, weight: fol === "G" ? 400 : 600 });
      var labP = text(g, 940, y, pct3(prod), { fill: C.muted, size: 11.5, anchor: "end" });
      var labK = text(g, 940, y, keep ? "keeps " + keep + "  ·  " + pct3(DATA.keepQ[m]) : "keeps 0", { fill: keep ? C.ink : C.faint, size: 11.5, anchor: "end", weight: keep ? 600 : 400 });
      return { g: g, y: y, cells: cells, dots: dots, tag: tag, labP: labP, labK: labK, keep: keep, col: col };
    });
    var cap2a = fit(text(g2, 20, 360, "A prefix scores the product of its tokens under its best mode. The 15th best score is " + pct3(DATA.theta) + ": each mode keeps the prefixes of its block that reach it, 37 in all.", { fill: C.muted, size: 12.5 }), 920);
    var cap2b = fit(text(g2, 20, 378, "Best first. A prefix kept by several modes becomes one node, so the 37 kept prefixes make 15 nodes:", { fill: C.muted, size: 12.5 }), 920);
    var yA2 = 414, yB2 = 452;
    function rowOf2(p) { return p[0] === "this" ? yA2 : yB2; }
    var tree2 = buildTree(g2, DATA.mixture, rowOf2, (yA2 + yB2) / 2);
    var badgeA = text(g2, X0, yA2 - 20, "", { fill: C.blue, size: 11.5, weight: 600 });
    var badgeB = text(g2, 940, yB2 + 20, "", { fill: C.orange, size: 11.5, weight: 600, anchor: "end" });   // right-aligned: the accepted line sits under the tree's root
    var cnt2n = text(g2, 940, yA2 - 22, "", { fill: C.muted, size: 12.5, anchor: "end" });
    var cnt2c = text(g2, 940, yA2 - 4, "", { fill: C.green, size: 12.5, anchor: "end", weight: 600 });
    var ghost2 = cell(g2, colX(6) + CW / 2, yA2, CW, CH, "find", "X", { strokeWidth: 1.4 });
    var ghost2t = text(g2, colX(7) + 2, yA2, "not in the tree", { fill: C.red, size: 11.5 });
    var acc2 = acceptedLine(g2, X0, 503, 6, C.green);
    // each node is pulled from the rows that keep its prefix: one flying copy per such mode, merging at the node
    var flyLayer = mk("g", {}, g2);
    var flights = [];
    DATA.mixture.forEach(function (p, i) {
      var d = p.length, x1 = colX(d - 1) + CW / 2, y1 = rowOf2(p);
      DATA.mixtureSup[i].forEach(function (m) {
        var r = modeRows[m - 1];
        var x0 = d <= 7 ? x1 : xDots + 8;   // positions 8-9 are beyond the seven shown: they leave from the dots
        var c = cell(flyLayer, x0, r.y, CW, MH, p[d - 1], status(p), { size: 12 });
        flights.push({ i: i, c: c, dx: x1 - x0, dy: y1 - r.y });
      });
    });
    // "depth 1: modes 1, 4, 5, 6 · 2–7: modes 1, 4, 6 · 8–9: mode 6" for the nodes of one row built so far
    function supText(first, built) {
      var segs = [], cur = null;
      DATA.mixture.forEach(function (p, i) {
        if (p[0] !== first || !built[i]) return;
        var key = DATA.mixtureSup[i].join(",");
        if (cur && cur.key === key && cur.d1 === p.length - 1) { cur.d1 = p.length; return; }
        cur = { key: key, d0: p.length, d1: p.length, s: DATA.mixtureSup[i] };
        segs.push(cur);
      });
      return segs.map(function (sg, j) {
        var range = sg.d0 === sg.d1 ? String(sg.d0) : sg.d0 + "–" + sg.d1;
        return (j === 0 ? "depth " : "") + range + ": " + (sg.s.length > 1 ? "modes " : "mode ") + sg.s.join(", ");
      }).join("  ·  ");
    }

    U.push(function (t) {
      var vis = ez(t, T.s2, 0.5) * (1 - ez(t, T.s2end, 0.4));
      setOp(g2, vis);
      if (vis <= 0) return;
      var kc = ez(t, T.cut, 0.6);   // the common cut-off: kept prefixes framed, the rest dimmed
      setOp(hdrP, ez(t, T.modes, 0.4) * (1 - kc)); setOp(hdrK, kc);
      modeRows.forEach(function (r, m) {
        var q = ez(t, T.modes + m * T.modeStep, 0.4);
        setOp(r.g, q);
        r.g.setAttribute("transform", "translate(" + (-14 * (1 - q)).toFixed(2) + ",0)");
        r.cells.forEach(function (c, i) {
          var kept = i < r.keep;
          c.rect.setAttribute("stroke-width", (kept ? 1.2 + 1.1 * kc : 1.2).toFixed(2));
          setOp(c.g, kept ? 1 : 1 - 0.65 * kc);
        });
        var more = r.keep > 7 && kc > 0.5;
        setOp(r.dots, r.keep > 7 ? 1 : 1 - 0.65 * kc);
        r.dots.setAttribute("fill", more ? r.col : C.faint);
        r.dots.setAttribute("font-weight", more ? 700 : 400);
        setOp(r.labP, 1 - kc); setOp(r.labK, kc);
      });
      setOp(cap2a, ez(t, T.cut, 0.5));
      setOp(cap2b, ez(t, T.cap2, 0.5));
      var shown = tree2.update(t, T.tree2, T.nodeStep2);
      flights.forEach(function (f) {
        var a = (t - (T.tree2 + f.i * T.nodeStep2 - T.fly)) / T.fly;
        if (a <= 0 || a >= 1) { setOp(f.c.g, 0); return; }
        var e = easeOut(a);
        f.c.place(1, f.dx * e, f.dy * e);
        setOp(f.c.g, 0.9 * clamp01(a / 0.1) * (1 - Math.pow(a, 6)));
      });
      var built = tree2.nodes.map(function (nd, i) { return t >= T.tree2 + i * T.nodeStep2; });
      var sA = supText("this", built), sB = supText("down", built);
      badgeA.textContent = sA; setOp(badgeA, sA ? 1 : 0);
      badgeB.textContent = sB; setOp(badgeB, sB ? 1 : 0);
      cnt2n.textContent = shown.n ? "nodes  " + shown.n : "";
      cnt2c.textContent = shown.n ? "chimeras  " + shown.x : "";
      setOp(cnt2n, shown.n ? 1 : 0); setOp(cnt2c, shown.n ? 1 : 0);
      for (var d = 1; d <= 6; d++) tree2.mark(DATA.A.slice(0, d), ez(t, T.ver2 + 0.15 + (d - 1) * 0.3, 0.3));
      var kg = ez(t, T.ver2 + 2.05, 0.35);
      setOp(ghost2.g, kg); ghost2.place(0.8 + 0.2 * kg); setOp(ghost2t, ez(t, T.ver2 + 2.25, 0.3));
      var ka = ez(t, T.ver2 + 2.5, 0.45);
      setOp(acc2, ka);
      acc2.setAttribute("transform", "translate(0," + (8 * (1 - ka)).toFixed(2) + ")");
    });

    // ---------- scene 3: summary ----------
    var g3 = mk("g", {}, svg);
    var head = mk("text", { x: 480, y: 178, "font-size": 27, "font-weight": 700, "text-anchor": "middle", "dominant-baseline": "middle", fill: C.ink, "font-family": FONT }, g3);
    var hs1 = mk("tspan", {}, head); hs1.textContent = "This round: ";
    var hs2 = mk("tspan", { fill: C.red, "font-weight": 800 }, head); hs2.textContent = "2";
    var hs3 = mk("tspan", {}, head); hs3.textContent = " → ";
    var hs4 = mk("tspan", { fill: C.green, "font-weight": 800 }, head); hs4.textContent = "6";
    var hs5 = mk("tspan", {}, head); hs5.textContent = " accepted tokens, one verification pass";
    var tiles = [
      ["27–33%", "of a 64-node product tree’s nodes are chimeras", "628 prompts, Qwen3-4B and Qwen3-8B"],
      ["+8–10%", "more of the target’s probability mass covered", "at the same budget (Qwen3-4B)"],
      ["up to 7.8×", "speedup; 5–8% over the best single-mode tree", "lossless, K+1 extra vectors, no extra pass"]
    ].map(function (tt, i) {
      var x = 20 + i * 315, y = 222, w = 290, h = 112;
      var g = mk("g", {}, g3);
      mk("rect", { x: x, y: y, width: w, height: h, rx: 12, fill: C.white, stroke: C.rule, "stroke-width": 1 }, g);
      text(g, x + 18, y + 34, tt[0], { size: 32, weight: 800, fill: C.purple });
      fit(text(g, x + 18, y + 68, tt[1], { size: 12.5, fill: C.ink }), w - 32);
      fit(text(g, x + 18, y + 88, tt[2], { size: 11.5, fill: C.muted }), w - 32);
      return { g: g, x: x + w / 2, y: y + h / 2 };
    });
    var sum1 = fit(text(g3, 480, 394, "Multi-Future Drafting: K learned modes at the drafter’s input, one exact tree over all of them.", { size: 13, weight: 500, fill: C.muted, anchor: "middle" }), 900);
    var sum2 = text(g3, 480, 431, "Branch where the futures diverge.", { size: 30, weight: 800, fill: C.purple, anchor: "middle", spacing: "-.01em" });  // the closing line (user 2026-11-06)
    U.push(function (t) {
      var vis = ez(t, T.s3, 0.5) * (1 - ez(t, T.end - 0.6, 0.5));
      setOp(g3, vis);
      if (vis <= 0) return;
      setOp(head, ez(t, T.s3 + 0.1, 0.5));
      tiles.forEach(function (tl, i) {
        var q = ez(t, T.s3 + 0.6 + i * 0.35, 0.5);
        setOp(tl.g, q);
        tl.g.setAttribute("transform", "translate(" + tl.x + "," + tl.y + ") scale(" + (0.85 + 0.15 * q) + ") translate(" + (-tl.x) + "," + (-tl.y) + ")");
      });
      setOp(sum1, ez(t, T.s3 + 1.9, 0.5));
      setOp(sum2, ez(t, T.s3 + 2.2, 0.5));
    });

    return U;
  }

  // A horizontal tree: depth along the token columns, rows given by rowOf(prefix). Nodes appear in list order.
  function buildTree(parent, prefixes, rowOf, rootY) {
    var g = mk("g", {}, parent);
    var root = cell(g, X0 - 36, rootY, 54, CH, "break", "G", { size: 12.5 });
    var edges = mk("g", {}, g);
    var byKey = {};
    var nodes = prefixes.map(function (p, i) {
      var kind = status(p);
      var x = colX(p.length - 1) + CW / 2, y = rowOf(p);
      var c = cell(g, x, y, CW, CH, p[p.length - 1], kind, { strokeWidth: 1.2 });
      var par = p.length > 1 ? byKey[p.slice(0, -1).join("\u0001")] : { cx: root.cx, cy: root.cy, w: 54 };
      var e = mk("line", { x1: par.cx + par.w / 2, y1: par.cy, x2: x - CW / 2, y2: y, stroke: kind === "X" ? C.red : C.gray, "stroke-width": 1.4, "stroke-dasharray": kind === "X" ? "3 3" : null }, edges);
      var n = { p: p, kind: kind, cell: c, edge: e, cx: x, cy: y, w: CW, i: i, marked: 0 };
      byKey[p.join("\u0001")] = n;
      return n;
    });
    function update(t, t0, step) {
      var kr = ez(t, t0 - 0.3, 0.4);
      setOp(root.g, kr); root.place(0.8 + 0.2 * kr);
      var n = 0, x = 0;
      nodes.forEach(function (nd, i) {
        var q = ez(t, t0 + i * step, 0.38);
        setOp(nd.cell.g, q); setOp(nd.edge, q);
        nd.cell.place(0.6 + 0.4 * q);
        if (q > 0) { n++; if (nd.kind === "X") x++; }
        // reset verification marks; mark() re-applies them afterwards
        if (nd.marked) {
          nd.cell.rect.setAttribute("stroke", style(nd.kind).stroke);
          nd.cell.rect.setAttribute("stroke-width", 1.2);
          nd.marked = 0;
        }
      });
      return { n: n, x: x };
    }
    function mark(prefix, k) {
      if (k <= 0) return;
      var nd = byKey[prefix.join("\u0001")];
      if (!nd) return;
      nd.cell.rect.setAttribute("stroke", C.green);
      nd.cell.rect.setAttribute("stroke-width", (1.2 + 1.6 * k).toFixed(2));
      nd.marked = 1;
    }
    return { update: update, mark: mark, nodes: nodes };
  }

  // ---- controller ------------------------------------------------------------------------------------------
  function mount(svg, opts) {
    opts = opts || {};
    var U = build(svg);
    var t = 0, speed = 1, playing = false, userPaused = false, last = null, raf = null, listeners = [];
    function render(tt) {
      t = tt;
      for (var i = 0; i < U.length; i++) U[i](t);
      for (var j = 0; j < listeners.length; j++) listeners[j](t);
    }
    function frame(now) {
      if (!playing) return;
      if (last === null) last = now;
      var dt = (now - last) / 1000 * speed;
      last = now;
      var nt = t + dt;
      if (nt >= T.end) nt = nt - T.end;
      render(nt);
      raf = global.requestAnimationFrame(frame);
    }
    var api = {
      scenes: SCENES,
      duration: T.end,
      times: T,
      render: render,
      seek: function (tt) { render(Math.max(0, Math.min(T.end - 1e-3, tt))); },
      play: function () { userPaused = false; if (playing) return; playing = true; last = null; raf = global.requestAnimationFrame(frame); },
      pause: function (auto) { if (!auto) userPaused = true; playing = false; if (raf) global.cancelAnimationFrame(raf); raf = null; render(t); },
      playing: function () { return playing; },
      userPaused: function () { return userPaused; },
      setSpeed: function (v) { speed = v; },
      onFrame: function (cb) { listeners.push(cb); },
      refit: fitAll
    };
    render(0);
    fitAll();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { fitAll(); render(t); });
    if (opts.autoplay) api.play();
    return api;
  }

  global.MFDAnimation = { mount: mount, DATA: DATA, T: T };
})(window);
