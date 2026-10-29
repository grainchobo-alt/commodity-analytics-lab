// 발표 v4 HTML: Reveal 초기화 + 장이 보일 때 Plotly 차트를 그림(숨은 장은 크기가 0이라 미리 못 그림)
const C = { g: "#1E3A2F", gold: "#C8A24A", red: "#D9582B", blue: "#2F6E8F", ink: "#222", muted: "#6B6B66", grey: "#C9CBC4" };
const FONT = { family: "Malgun Gothic, sans-serif", size: 13, color: C.ink };
const CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["toImage", "lasso2d", "select2d"] };
const base = (extra) => Object.assign({ font: FONT, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { l: 55, r: 15, t: 10, b: 45 } }, extra);
const $ = (id) => document.getElementById(id);

// ── 자료 기준일 표시 + 열람일 기준 7일 지나면 경고
(function stamp() {
  const days = Math.floor((Date.now() - new Date(D.asof + "T00:00:00")) / 864e5), el = $("stamp");
  el.textContent = `자료 기준일 ${D.asof} · 구매 판단 ${D.buy_asof} · 생성 ${D.built}`;
  if (days > 7) { el.classList.add("stale"); el.textContent += ` · ⚠ 자료가 ${days}일 지남 — 15·16장·요약 숫자 갱신 필요`; }
})();

// ── 5장: 품목 × 월 3D 막대 / 2D 표
let xsMode = "3d", xsHl = null;
function xsBars() {
  const X = D.xs, x = [], y = [], z = [], i = [], j = [], k = [], fc = [], tx = [], ty = [], tz = [], tt = [], hv = [];
  const col = (v) => { const t = Math.max(-1, Math.min(1, v / 12)); return t >= 0 ? `rgb(${Math.round(255 - 160 * t)},${Math.round(235 - 60 * t)},${Math.round(160 - 90 * t)})` : `rgb(${Math.round(235 + 20 * t)},${Math.round(235 + 140 * t)},${Math.round(160 + 80 * t)})`; };
  X.items.forEach((it, a) => X.months.forEach((m, b) => {
    const v = X.z[a][b]; if (v === null) return;
    const h = Math.max(-20, Math.min(20, v)), z0 = Math.min(0, h), z1 = Math.max(0, h) || 0.15, x0 = b - 0.4, x1 = b + 0.4, y0 = a - 0.38, y1 = a + 0.38, n = x.length;
    [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]].forEach((p) => { x.push(p[0]); y.push(p[1]); z.push(p[2]); hv.push(`${it} · ${m}<br>${v > 0 ? "+" : ""}${v}%`); });
    const F = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]];
    const dim = xsHl === "ts" ? a !== 0 : xsHl === "cs" ? b !== X.months.length - 1 : false;
    F.forEach((f) => { i.push(n + f[0]); j.push(n + f[1]); k.push(n + f[2]); fc.push(dim ? "rgba(225,225,220,0.55)" : col(v)); });
    tx.push(b); ty.push(a); tz.push(Math.max(h, 0) + 1.3); tt.push(Math.round(v) === 0 ? "0" : (v > 0 ? "+" : "") + Math.round(v));
  }));
  return [{ type: "mesh3d", x, y, z, i, j, k, facecolor: fc, flatshading: true, hovertext: hv, hoverinfo: "text", lighting: { ambient: 0.75, diffuse: 0.6 } },
          { type: "scatter3d", mode: "text", x: tx, y: ty, z: tz, text: tt, textfont: { size: 9, color: "#222" }, hoverinfo: "skip" }];
}
function drawXs() {
  const X = D.xs, el = $("c-xs");
  if (xsMode === "3d") {
    Plotly.react(el, xsBars(), base({ margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false, scene: {
      xaxis: { title: "", tickvals: X.months.map((_, b) => b), ticktext: X.months, tickfont: { size: 10 } },
      yaxis: { title: "", tickvals: X.items.map((_, a) => a), ticktext: X.items, tickfont: { size: 11 } },
      zaxis: { title: "월 수익률 %", range: [-20, 22], tickfont: { size: 9 } }, aspectmode: "manual", aspectratio: { x: 2.1, y: 1.8, z: 0.55 },
      camera: { eye: { x: -1.25, y: -2.3, z: 1.75 } } } }), CFG);
  } else {
    const z = X.z, txt = z.map((r) => r.map((v) => v === null ? "" : (v > 0 ? "+" : "") + v.toFixed(1)));
    const sh = [];
    if (xsHl === "ts") sh.push({ type: "rect", x0: -0.5, x1: X.months.length - 0.5, y0: -0.5, y1: 0.5, line: { color: C.blue, width: 3 } });
    if (xsHl === "cs") sh.push({ type: "rect", x0: X.months.length - 1.5, x1: X.months.length - 0.5, y0: -0.5, y1: X.items.length - 0.5, line: { color: C.gold, width: 3 } });
    Plotly.react(el, [{ type: "heatmap", z, x: X.months, y: X.items, text: txt, texttemplate: "%{text}", colorscale: "RdYlGn", zmid: 0, zmin: -15, zmax: 15,
      hovertemplate: "%{y} · %{x}<br>%{z:+.1f}%<extra></extra>", colorbar: { title: "%", thickness: 12 } }],
      base({ yaxis: { autorange: "reversed" }, xaxis: { side: "top" }, margin: { l: 70, r: 10, t: 30, b: 10 }, shapes: sh }), CFG);
  }
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-xs],[data-xs-hl]"); if (!b) return;
  if (b.dataset.xs) { xsMode = b.dataset.xs; document.querySelectorAll("[data-xs]").forEach((x) => x.classList.toggle("on", x === b)); }
  if (b.dataset.xsHl) { xsHl = xsHl === b.dataset.xsHl ? null : b.dataset.xsHl; document.querySelectorAll("[data-xs-hl]").forEach((x) => x.classList.toggle("on", x.dataset.xsHl === xsHl)); }
  drawXs();
});

// ── 14장: 효율적 투자선 연도 슬라이더 · 재생
const FY = Object.keys(D.frontier).map(Number).sort();
let frYear = 2017, frTimer = null;
const frRange = (() => { const v = [], r = []; FY.forEach((y) => { const d = D.frontier[y]; v.push(...d.fv, d.tan[0], d.mix[0]); r.push(...d.fr, d.tan[1], d.mix[1]); });
  return { x: [Math.min(...v) - 0.4, Math.max(...v) + 0.4], y: [Math.min(...r, 0) - 0.5, Math.max(...r) + 1] }; })();
function drawFrontier() {
  const d = D.frontier[frYear], tr = [];
  FY.filter((y) => y < frYear).forEach((y) => tr.push({ x: D.frontier[y].fv, y: D.frontier[y].fr, mode: "lines", line: { color: "rgba(150,150,145,0.35)", width: 1 }, hoverinfo: "skip", showlegend: false }));
  tr.push({ x: d.cv, y: d.cr, mode: "markers", marker: { size: 3, color: "#D8DBD4" }, name: "가능한 조합", hoverinfo: "skip" },
    { x: d.fv, y: d.fr, mode: "lines", line: { color: C.g, width: 3.5 }, name: `${frYear} 효율적 투자선`, hovertemplate: "변동성 %{x:.1f}% · 수익 %{y:.1f}%<extra></extra>" },
    { x: [d.tan[0]], y: [d.tan[1]], mode: "markers", marker: { size: 14, color: C.red, line: { color: "#fff", width: 1.5 } }, name: "그해 사후 최적", hovertemplate: "사후 최적<br>변동성 %{x:.1f}% · 수익 %{y:.1f}%<extra></extra>" },
    { x: [d.mix[0]], y: [d.mix[1]], mode: "markers", marker: { size: 22, symbol: "star", color: C.gold, line: { color: "#333", width: 1 } }, name: "우리 혼합 50·25·25", hovertemplate: "우리 혼합<br>변동성 %{x:.1f}% · 수익 %{y:.1f}%<extra></extra>" });
  Plotly.react($("c-frontier"), tr, base({ xaxis: { title: "연 변동성 (%)", range: frRange.x, gridcolor: "#EEE" }, yaxis: { title: "연 수익 (%)", range: frRange.y, gridcolor: "#EEE" },
    legend: { orientation: "h", y: 1.06, x: 0 }, margin: { l: 55, r: 15, t: 30, b: 45 },
    annotations: [{ x: 0.98, y: 0.04, xref: "paper", yref: "paper", text: `<b>${frYear}</b>`, showarrow: false, font: { size: 44, color: "rgba(30,58,47,0.18)" } }] }), CFG);
  $("fr-ylab").textContent = frYear; $("fr-year").value = frYear;
  $("fr-title").textContent = `${frYear}년 기준 (직전 5년 ${frYear - 4}–${frYear})`;
  $("fr-mix").textContent = `변동성 ${d.mix[0].toFixed(1)}% · 수익 ${d.mix[1].toFixed(1)}%`;
  $("fr-gap").innerHTML = `<span style="color:${d.gap >= 2 ? C.red : C.g}">${d.gap.toFixed(1)}%p ${d.gap < 0.5 ? "(경계선 위)" : "아래"}</span>`;
  $("fr-tan").textContent = `변동성 ${d.tan[0].toFixed(1)}% · 수익 ${d.tan[1].toFixed(1)}%`;
  $("fr-tanw").textContent = `최소분산 ${d.tanw[0]} · 모멘텀 ${d.tanw[1]} · 가치 ${d.tanw[2]} (%)`;
}
function initFrontier() {
  const s = $("fr-year"); s.min = FY[0]; s.max = FY[FY.length - 1];
  s.oninput = () => { frYear = +s.value; drawFrontier(); };
  $("fr-play").onclick = () => {
    if (frTimer) { clearInterval(frTimer); frTimer = null; $("fr-play").textContent = "▶"; return; }
    if (frYear >= FY[FY.length - 1]) frYear = FY[0] - 1;
    $("fr-play").textContent = "❚❚";
    frTimer = setInterval(() => { frYear++; drawFrontier(); if (frYear >= FY[FY.length - 1]) { clearInterval(frTimer); frTimer = null; $("fr-play").textContent = "▶"; } }, 900);
  };
}

// ── 15장: 구매 판단 품목 선택
let buyItem = "옥수수";
function drawBuy() {
  const b = D.buy[buyItem], last = b.p[b.p.length - 1];
  Plotly.react($("c-buy"), [
    { x: b.d, y: b.p, mode: "lines", line: { color: C.blue, width: 2 }, name: buyItem, hovertemplate: "%{x}<br>%{y:.1f} " + b.unit + "<extra></extra>" },
    { x: [b.d[b.d.length - 1]], y: [last], mode: "markers+text", marker: { size: 13, color: b.col, line: { color: "#fff", width: 2 } }, text: [`  지금 ${last.toFixed(0)}`], textposition: "middle right", hoverinfo: "skip", showlegend: false }],
    base({ showlegend: false, xaxis: { gridcolor: "#EEE" }, yaxis: { title: b.unit, gridcolor: "#EEE" },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: b.lo, y1: b.lo, line: { color: C.g, dash: "dot", width: 1 } }, { type: "line", xref: "paper", x0: 0, x1: 1, y0: b.hi, y1: b.hi, line: { color: C.red, dash: "dot", width: 1 } }],
      annotations: [{ xref: "paper", x: 0.01, y: b.lo, text: `3년 최저 ${b.lo.toFixed(0)}`, showarrow: false, yshift: -10, font: { size: 11, color: C.g }, xanchor: "left" },
        { xref: "paper", x: 0.01, y: b.hi, text: `3년 최고 ${b.hi.toFixed(0)}`, showarrow: false, yshift: 10, font: { size: 11, color: C.red }, xanchor: "left" }] }), CFG);
  $("buy-asof").textContent = `이번 주 판단 (기준일 ${D.buy_asof})`; $("buy-name").textContent = buyItem;
  $("buy-dot").style.background = b.col; $("buy-act").textContent = b.act; $("buy-act").style.color = b.col; $("buy-score").textContent = b.score; $("buy-why").textContent = b.why;
  $("buy-pct").textContent = `${b.pct}% (0 = 최저, 100 = 최고)`; $("buy-evid").textContent = b.evid ? "예 — 과거 검증에서 근거 확인 품목" : "아니오 — 점수만 참고";
  document.querySelectorAll("#buy-btns button").forEach((x) => x.classList.toggle("on", x.textContent === buyItem));
}
function initBuy() {
  $("buy-btns").innerHTML = Object.keys(D.buy).map((k) => `<button>${k}</button>`).join("");
  $("buy-btns").onclick = (e) => { if (e.target.tagName === "BUTTON") { buyItem = e.target.textContent; drawBuy(); } };
  Plotly.newPlot($("c-buy-val"), [{ type: "bar", x: ["판단대로 미룸", "아무 때나 미룸"], y: [23, -14], marker: { color: [C.g, C.red] }, text: ["+23bp", "−14bp"], textposition: "outside", hoverinfo: "skip" }],
    base({ yaxis: { visible: false, range: [-22, 32] }, margin: { l: 10, r: 10, t: 5, b: 25 } }), Object.assign({}, CFG, { displayModeBar: false }));
}

// ── 상세: PCA · GARCH · HMM
function drawPca() {
  const P = D.pca, L = 4;
  $("pca-share").textContent = (P.share * 100).toFixed(1) + "%";
  const proj = P.x.map((_, i) => P.x[i] * P.pc1[0] + P.y[i] * P.pc1[1] + P.z[i] * P.pc1[2]);
  Plotly.react($("c-pca"), [
    { type: "scatter3d", mode: "markers", x: P.x, y: P.y, z: P.z, text: P.d, marker: { size: 2.4, color: proj, colorscale: "Viridis", opacity: 0.6 }, hovertemplate: "%{text}<br>옥수수 %{x:.1f} · 대두 %{y:.1f} · 밀 %{z:.1f}<extra></extra>", name: "주간" },
    { type: "scatter3d", mode: "lines+text", x: [-L * P.pc1[0], L * P.pc1[0]], y: [-L * P.pc1[1], L * P.pc1[1]], z: [-L * P.pc1[2], L * P.pc1[2]], text: ["", "공통 흐름 1"], line: { color: C.red, width: 8 }, textfont: { color: C.red, size: 14 }, hoverinfo: "skip", name: "PC1" }],
    base({ showlegend: false, margin: { l: 0, r: 0, t: 0, b: 0 }, scene: { xaxis: { title: "옥수수" }, yaxis: { title: "대두" }, zaxis: { title: "밀" }, camera: { eye: { x: 1.5, y: -1.5, z: 0.7 } } } }), CFG);
}
function drawGarch() {
  const G = D.garch;
  Plotly.react($("c-garch"), [{ type: "surface", x: G.t, y: G.items, z: G.z, colorscale: "YlOrRd", cmin: 10, cmax: 70, colorbar: { title: "연 %", thickness: 12, len: 0.7 },
    hovertemplate: "%{y} · %{x}<br>예측 변동성 %{z:.0f}%<extra></extra>" }],
    base({ margin: { l: 0, r: 0, t: 0, b: 0 }, scene: { xaxis: { title: "" }, yaxis: { title: "" }, zaxis: { title: "연 변동성 %" }, aspectmode: "manual", aspectratio: { x: 2.2, y: 1.1, z: 0.7 }, camera: { eye: { x: -1.2, y: -1.8, z: 1.0 } } } }), CFG);
}
function drawHmm() {
  const H = D.hmm;
  $("hmm-top").textContent = Object.entries(H.top).map(([y, v]) => `${y} (${v.toFixed(2)})`).join(" · ");
  Plotly.react($("c-hmm"), [
    { x: H.d, y: H.p, type: "scatter", mode: "lines", fill: "tozeroy", line: { color: C.red, width: 1 }, fillcolor: "rgba(217,88,43,0.35)", name: "위기 확률", hovertemplate: "%{x}<br>위기 확률 %{y:.2f}<extra></extra>" },
    { x: H.d, y: H.cum, type: "scatter", mode: "lines", line: { color: C.g, width: 1.6 }, name: "동일 비중 누적 수익 (%)", yaxis: "y2", hovertemplate: "누적 %{y:.0f}%<extra></extra>" }],
    base({ legend: { orientation: "h", y: 1.08 }, margin: { l: 55, r: 55, t: 30, b: 30 }, hovermode: "x unified", yaxis: { title: "위기 확률", range: [0, 1.02] },
      yaxis2: { title: "누적 수익 (%)", overlaying: "y", side: "right", showgrid: false }, xaxis: { rangeslider: { thickness: 0.08 } } }), CFG);
}

// ── Reveal + 장 전환 시 그리기
const DRAW = { xs: drawXs, frontier: drawFrontier, buy: drawBuy, pca: drawPca, garch: drawGarch, hmm: drawHmm };
const INIT = { frontier: initFrontier, buy: initBuy }, done = new Set();
function show(sl) {
  const k = sl && sl.dataset.chart; if (!k) return;
  if (!done.has(k)) { if (INIT[k]) INIT[k](); DRAW[k](); done.add(k); }
  sl.querySelectorAll(".js-plotly-plot").forEach((p) => Plotly.Plots.resize(p));
}
Reveal.initialize({ hash: true, width: "100%", height: "100%", margin: 0, minScale: 1, maxScale: 1, center: false, transition: "fade", transitionSpeed: "fast",
  slideNumber: "c/t", controls: true, progress: true, plugins: [RevealNotes] }).then(() => show(Reveal.getCurrentSlide()));
Reveal.on("slidechanged", (e) => show(e.currentSlide));
window.addEventListener("resize", () => show(Reveal.getCurrentSlide()));
