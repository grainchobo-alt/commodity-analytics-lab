// 발표 v4 동적 HTML: 화면 단위(--u) 맞춤 · 장이 보일 때 Plotly 그리기 · 3D 천천히 자동 회전 · 큰 숫자 세기 · 기준일 경고
const C = { g: "#1E3A2F", gold: "#C8A24A", red: "#D9582B", blue: "#2F6E8F", ink: "#222", muted: "#6B6B66", grey: "#B4B2A9" };
const $ = (s, el = document) => el.querySelector(s);
const unit = () => parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--u"));
const FONT = () => ({ family: "Malgun Gothic, sans-serif", size: Math.max(9, unit() * 0.11), color: C.ink });
const CFG = { displaylogo: false, responsive: true, displayModeBar: false };
const lay = (extra) => Object.assign({ font: FONT(), paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false }, extra);
const ax3 = (t, extra) => Object.assign({ title: { text: t, font: { size: Math.max(9, unit() * 0.1) } }, tickfont: { size: Math.max(8, unit() * 0.085) }, backgroundcolor: "#F7F7F4", showbackground: true, gridcolor: "#E2E2DC" }, extra);

function fit() {                                                                // 13.333 × 7.5인치 캔버스를 화면에 맞춤
  document.documentElement.style.setProperty("--u", Math.min(innerWidth / 13.333, innerHeight / 7.5) + "px");
}
fit();

(function stamp() {
  const days = Math.floor((Date.now() - new Date(D.asof + "T00:00:00")) / 864e5), el = $("#stamp");
  el.textContent = `자료 기준일 ${D.asof} · 구매 판단 ${D.buy_asof} · 생성 ${D.built}`;
  if (days > 7) { el.classList.add("stale"); el.textContent += ` · ⚠ 자료가 ${days}일 지남 — 15·16장·요약 숫자 갱신 필요`; }
})();

// ── 3D 도우미: 직육면체 막대 · 곡선 아래 커튼
function bars3d(B, dx = 0.6, dy = 0.6) {
  const x = [], y = [], z = [], i = [], j = [], k = [], fc = [], hv = [];
  const F = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]];
  B.forEach((b) => {
    const n = x.length, x0 = b.x - dx / 2, x1 = b.x + dx / 2, y0 = b.y - dy / 2, y1 = b.y + dy / 2, z0 = Math.min(b.z0 || 0, b.z), z1 = Math.max(b.z0 || 0, b.z);
    [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]].forEach((p) => { x.push(p[0]); y.push(p[1]); z.push(p[2]); hv.push(b.h); });
    F.forEach((f) => { i.push(n + f[0]); j.push(n + f[1]); k.push(n + f[2]); fc.push(b.c); });
  });
  return { type: "mesh3d", x, y, z, i, j, k, facecolor: fc, flatshading: true, hovertext: hv, hoverinfo: "text", lighting: { ambient: 0.7, diffuse: 0.7, specular: 0.1 } };
}
function curtain(xs, yv, zs, color, op = 0.75, name = "") {
  const x = [], y = [], z = [], i = [], j = [], k = [];
  xs.forEach((xv, q) => { x.push(xv, xv); y.push(yv, yv); z.push(0, zs[q]); });
  for (let q = 0; q < xs.length - 1; q++) { const a = 2 * q; i.push(a, a + 1); j.push(a + 1, a + 3); k.push(a + 2, a + 2); }
  return [{ type: "mesh3d", x, y, z, i, j, k, color, opacity: op, hoverinfo: "skip", flatshading: true },
          { type: "scatter3d", mode: "lines", x: xs, y: xs.map(() => yv), z: zs, line: { color, width: 4 }, name, hovertemplate: `${name} %{x}<br>%{z:.2f}<extra></extra>` }];
}
const seq = (n, f) => Array.from({ length: n }, (_, q) => f(q));

// ── 8장: 머신러닝 카드
const P = {};
P.lasso = (el) => {
  const L = D.lasso, B = [];
  L.vars.forEach((v, a) => L.lev.forEach((_, b) => { const z = L.z[a][b]; B.push({ x: a, y: b, z: Math.max(z, 0.012), c: z > 1e-6 ? "#2E7D52" : "#D5D5CE", h: `${v} · 강도 ${b + 1}<br>영향력 ${z.toFixed(2)}${z > 1e-6 ? "" : " (0으로 제거)"}` }); }));
  return [[bars3d(B)], { scene: { xaxis: ax3("", { tickvals: L.vars.map((_, a) => a), ticktext: L.vars }), yaxis: ax3("걸러내는 강도", { tickvals: [0, 4], ticktext: ["약", "강"] }),
    zaxis: ax3("영향력"), aspectmode: "manual", aspectratio: { x: 2, y: 1.1, z: 0.8 }, camera: { eye: { x: -1.2, y: -1.9, z: 1.0 } } } }];
};
P.pca = (el) => {
  const p = D.pca, L = 3.8, proj = p.x.map((_, q) => p.x[q] * p.pc1[0] + p.y[q] * p.pc1[1] + p.z[q] * p.pc1[2]);
  return [[{ type: "scatter3d", mode: "markers", x: p.x, y: p.y, z: p.z, text: p.d, marker: { size: 1.8, color: proj, colorscale: "Viridis", opacity: 0.6 }, hovertemplate: "%{text}<br>옥수수 %{x:.1f} · 대두 %{y:.1f} · 밀 %{z:.1f}<extra></extra>" },
    { type: "scatter3d", mode: "lines+text", x: [-L * p.pc1[0], L * p.pc1[0]], y: [-L * p.pc1[1], L * p.pc1[1]], z: [-L * p.pc1[2], L * p.pc1[2]], text: ["", `공통 흐름 ${Math.round(p.share * 100)}%`],
      line: { color: C.red, width: 7 }, textfont: { color: C.red, size: Math.max(9, unit() * 0.11) }, hoverinfo: "skip" }],
    { scene: { xaxis: ax3("옥수수", { range: [-4, 4] }), yaxis: ax3("대두", { range: [-4, 4] }), zaxis: ax3("밀", { range: [-4, 4] }), camera: { eye: { x: 1.45, y: -1.45, z: 0.7 } } } }];
};
P.rf = () => [[{ type: "surface", x: D.rf.g, y: D.rf.g, z: D.rf.z, colorscale: "Viridis", cmin: -1.5, cmax: 2.2, showscale: false, hovertemplate: "가격 위치 %{x:.2f} · 변동성 %{y:.2f}<br>예측 %{z:.2f}<extra></extra>" }],
  { scene: { xaxis: ax3("가격 위치"), yaxis: ax3("변동성"), zaxis: ax3("4주 뒤 수익률"), camera: { eye: { x: -1.5, y: 1.4, z: 0.9 } } } }];
P.lstm = () => [[{ type: "surface", x: seq(52, (q) => q + 1), y: seq(60, (q) => q), z: D.lstm.z, colorscale: "Cividis", showscale: false, hovertemplate: "%{x}주 전 정보<br>기억 비중 %{z:.2f}<extra></extra>" }],
  { scene: { xaxis: ax3("몇 주 전 정보", { tickvals: [1, 13, 26, 39, 52] }), yaxis: ax3("예측 시점 →", { showticklabels: false }), zaxis: ax3("기억 비중", { showticklabels: false }),
    annotations: [{ x: 52, y: 0, z: 0.62, text: "1년 전(계절)", showarrow: false, font: { color: C.red, size: 10 } }, { x: 1, y: 0, z: 1.1, text: "최근", showarrow: false, font: { color: C.red, size: 10 } }],
    camera: { eye: { x: -1.5, y: -1.4, z: 0.9 } } } }];
// ── 9장: AI·위험·국면 카드
P.chronos = () => {
  const c = D.chronos, tr = [], cols = ["#7F2704", "#C2410C", "#F07A2D", "#F9A55F", "#FDD0A2"];
  c.h.slice().reverse().forEach((h, q) => tr.push(...curtain(c.px, h, c.z[c.h.length - 1 - q], cols[c.h.length - 1 - q], 0.8, `${h}주 뒤`)));
  return [tr, { scene: { xaxis: ax3("가격 (오늘 = 100)"), yaxis: ax3("몇 주 뒤", { tickvals: c.h }), zaxis: ax3("가능성", { showticklabels: false }), camera: { eye: { x: -0.9, y: -2.0, z: 0.8 } } } }];
};
P.finbert = () => {
  const f = D.finbert, B = [];
  f.secs.forEach((s, b) => f.yrs.forEach((y, a) => { const v = f.z[b][a]; if (v === null) return; B.push({ x: y, y: b, z: v, c: v >= 0 ? C.g : C.red, h: `${s} · ${y}<br>어조 ${v > 0 ? "+" : ""}${v.toFixed(2)}` }); }));
  return [[bars3d(B, 0.7, 0.55)], { scene: { xaxis: ax3("", { tickvals: [2012, 2016, 2020, 2024] }), yaxis: ax3("", { tickvals: [0, 1, 2], ticktext: f.secs }), zaxis: ax3("어조 (+긍정/−부정)"),
    aspectmode: "manual", aspectratio: { x: 2, y: 1.1, z: 0.8 }, camera: { eye: { x: -1.1, y: -1.9, z: 0.9 } } } }];
};
P.garch = () => [[{ type: "surface", x: D.garch.t, y: D.garch.items, z: D.garch.z, colorscale: "YlOrRd", cmin: 10, cmax: 70, showscale: false, hovertemplate: "%{y} · %{x}<br>예측 위험 %{z:.0f}%<extra></extra>" }],
  { scene: { xaxis: ax3("", { nticks: 4 }), yaxis: ax3(""), zaxis: ax3("예측 위험 (연 %)"), aspectmode: "manual", aspectratio: { x: 2.1, y: 1.2, z: 0.8 }, camera: { eye: { x: -1.2, y: -1.8, z: 1.0 } } } }];
P.hmm = () => {
  const h = D.hmm, idx = h.t.map((_, q) => q);
  return [[...curtain(idx, 1, h.p.map((v) => 1 - v), C.blue, 0.5, "평시"), ...curtain(idx, 0, h.p, C.red, 0.65, "위기")].map((t) => (t.type === "scatter3d" ? Object.assign(t, { text: h.t, hovertemplate: `%{text}<br>${t.name} 확률 %{z:.2f}<extra></extra>` }) : t)),
    { scene: { xaxis: ax3("", { tickvals: [12, 84, 156, 228].filter((q) => q < h.t.length), ticktext: [12, 84, 156, 228].filter((q) => q < h.t.length).map((q) => h.t[q].slice(0, 4)) }),
      yaxis: ax3("", { tickvals: [0, 1], ticktext: ["위기", "평시"], range: [-0.5, 1.5] }), zaxis: ax3("국면 확률", { range: [0, 1] }), aspectmode: "manual", aspectratio: { x: 2.2, y: 1, z: 0.8 }, camera: { eye: { x: -1.0, y: -1.9, z: 0.9 } } } }];
};

// ── 14장: 대시보드 ⑤ Portfolio Customization 3D (위험 × 수익 × Sharpe)
let efKey = Object.keys(D.ef3d)[0];
P.ef3d = (el) => {
  const pick = $(".ef-pick", el.parentNode);
  if (!pick.options.length) { pick.innerHTML = Object.keys(D.ef3d).map((k) => `<option>${k}</option>`).join(""); pick.onchange = () => { efKey = pick.value; draw(el, true); }; }
  const E = D.ef3d[efKey], sh = E.v.map((v, q) => E.r[q] / v), srt = sh.slice().sort((a, b) => a - b), smin = srt[Math.floor(srt.length * 0.01)], smax = srt[srt.length - 1];
  const fz = E.fv.map((v, q) => E.fr[q] / v), pt = (p, nm, sym, col, sz = 8) => ({ type: "scatter3d", mode: "markers+text", x: [p[0]], y: [p[1]], z: [p[1] / p[0]], text: [nm], textposition: "top center",
    marker: { size: sz, symbol: sym, color: col, line: { color: "#111", width: 1 } }, textfont: { size: Math.max(9, unit() * 0.1) }, hovertemplate: `${nm}<br>변동성 %{x:.1f}% · 수익 %{y:.1f}% · Sharpe %{z:.2f}<extra></extra>` });
  return [[{ type: "mesh3d", x: E.v, y: E.r, z: sh, alphahull: 0, opacity: 0.06, color: C.blue, hoverinfo: "skip" },
    { type: "scatter3d", mode: "markers", x: E.v, y: E.r, z: sh, marker: { size: 1.8, color: sh, colorscale: "Viridis", cmin: smin, cmax: smax, opacity: 0.75, colorbar: { title: { text: "Sharpe" }, thickness: 10, len: 0.55, x: 0.98 } },
      hovertemplate: "변동성 %{x:.1f}% · 수익 %{y:.1f}% · Sharpe %{z:.2f}<extra></extra>" },
    { type: "scatter3d", mode: "lines", x: E.fv, y: E.fr, z: fz, line: { color: C.red, width: 7 }, hovertemplate: "효율적 투자선<br>변동성 %{x:.1f}% · 수익 %{y:.1f}%<extra></extra>" },
    { type: "scatter3d", mode: "lines", x: E.fv, y: E.fr, z: E.fv.map(() => smin), line: { color: C.red, width: 3, dash: "dot" }, opacity: 0.5, hoverinfo: "skip" },
    pt(E.tan, "최대 Sharpe", "diamond", C.gold), pt(E.gmv, "최소분산", "circle", "#7B5EA7"), pt(E.mix, "★ 우리 혼합 50·25·25", "diamond", C.g, 10),
    { type: "scatter3d", mode: "markers+text", x: E.single.v, y: E.single.r, z: E.single.v.map((v, q) => E.single.r[q] / v), text: E.names, textposition: "bottom center",
      marker: { size: 5, symbol: "square", color: C.grey, line: { color: "#111", width: 0.5 } }, textfont: { size: Math.max(9, unit() * 0.1), color: C.muted }, hovertemplate: "%{text} 단독<extra></extra>" }],
    { scene: { xaxis: ax3("연 변동성 (%)"), yaxis: ax3("연 수익률 (%)"), zaxis: ax3("Sharpe"), aspectmode: "manual", aspectratio: { x: 1.15, y: 1, z: 0.7 }, camera: { eye: { x: 1.55, y: -1.55, z: 0.85 } } } }];
};
// ── 14장 오른쪽 위: 연도별 경계선 재생
const FY = Object.keys(D.efyear).map(Number).sort();
let eyYear = 2017, eyTimer = null;
const eyRange = (() => { const v = [], r = []; FY.forEach((y) => { const d = D.efyear[y]; v.push(...d.fv, d.tan[0], d.mix[0]); r.push(...d.fr, d.tan[1], d.mix[1]); });
  return { x: [Math.min(...v) - 0.3, Math.max(...v) + 0.3], y: [Math.min(...r, 0) - 0.5, Math.max(...r) + 1] }; })();
P.efyear = (el) => {
  const box = el.parentNode, s = $(".ey-yr", box), b = $(".ey-play", box);
  if (!s.min) {
    s.min = FY[0]; s.max = FY[FY.length - 1]; s.oninput = () => { eyYear = +s.value; draw(el, true); };
    b.onclick = () => {
      if (eyTimer) { clearInterval(eyTimer); eyTimer = null; b.textContent = "▶"; return; }
      if (eyYear >= FY[FY.length - 1]) eyYear = FY[0] - 1;
      b.textContent = "❚❚"; eyTimer = setInterval(() => { eyYear++; draw(el, true); if (eyYear >= FY[FY.length - 1]) { clearInterval(eyTimer); eyTimer = null; b.textContent = "▶"; } }, 850);
    };
  }
  s.value = eyYear; $(".ey-lab", box).textContent = eyYear;
  const d = D.efyear[eyYear]; $(".ey-gap", box).textContent = d.gap < 0.5 ? "★ 경계선 위" : `★ 경계선보다 ${d.gap.toFixed(1)}%p 아래`;
  const tr = FY.filter((y) => y < eyYear).map((y) => ({ x: D.efyear[y].fv, y: D.efyear[y].fr, mode: "lines", line: { color: "rgba(150,150,145,0.3)", width: 1 }, hoverinfo: "skip" }));
  tr.push({ x: d.fv, y: d.fr, mode: "lines", line: { color: C.g, width: 3 }, hovertemplate: "변동성 %{x:.1f}% · 수익 %{y:.1f}%<extra></extra>" },
    { x: [d.tan[0]], y: [d.tan[1]], mode: "markers", marker: { size: 10, color: C.red, line: { color: "#fff", width: 1 } }, hovertemplate: "그해 사후 최적<extra></extra>" },
    { x: [d.mix[0]], y: [d.mix[1]], mode: "markers", marker: { size: 16, symbol: "star", color: C.gold, line: { color: "#333", width: 1 } }, hovertemplate: "우리 혼합<extra></extra>" });
  return [tr, { margin: { l: 34, r: 6, t: 6, b: Math.round(unit() * 0.62) }, xaxis: { title: { text: "연 변동성 (%)", standoff: 2 }, range: eyRange.x, gridcolor: "#EEE" }, yaxis: { title: { text: "연 수익 (%)", standoff: 2 }, range: eyRange.y, gridcolor: "#EEE" } }];
};

// ── pptx 차트 (13장 선 · 15장 막대)
function pptChart(id) {
  const c = D.charts[id], o = c.opts, cols = (o.chartColors || [C.g, C.red]).map((x) => "#" + x);
  const tr = c.data.map((s, q) => c.type === "bar"
    ? { type: "bar", x: s.labels, y: s.values, marker: { color: s.values.map((_, k) => cols[k % cols.length]) }, text: o.showValue ? s.values.map((v) => (v > 0 ? "+" : "") + v) : undefined, textposition: "outside", hoverinfo: "skip", cliponaxis: false }
    : { type: "scatter", mode: "lines+markers", x: s.labels, y: s.values, name: s.name, line: { color: cols[q], width: 2.5 }, marker: { size: 6 }, hovertemplate: `${s.name} %{y}<extra>%{x}</extra>` });
  return [tr, { showlegend: !!o.showLegend, legend: { orientation: "h", y: -0.18 }, margin: { l: o.valAxisHidden ? 6 : 40, r: 8, t: o.showTitle ? 30 : 6, b: 30 },
    title: o.showTitle ? { text: o.title, font: { size: Math.max(10, unit() * 0.13), color: C.g }, x: 0.02 } : undefined,
    yaxis: { visible: !o.valAxisHidden, gridcolor: "#E5E5E0", zeroline: true }, xaxis: { gridcolor: "rgba(0,0,0,0)" } }];
}

// ── 그리기 · 자동 회전 · 숫자 세기
function draw(el, again) {
  const key = el.dataset.plot, [tr, l] = P[key] ? P[key](el) : pptChart(key);
  if (l.scene && l.scene.camera) {                                             // 작은 카드는 카메라를 멀리 · 글자 작게
    const small = el.clientWidth < unit() * 4, f = small ? 1.6 : 1.2, e = l.scene.camera.eye;
    l.scene.camera.eye = { x: e.x * f, y: e.y * f, z: e.z * f };
    if (small) ["xaxis", "yaxis", "zaxis"].forEach((a) => { if (l.scene[a]) { l.scene[a].tickfont = { size: 8 }; l.scene[a].title.font = { size: 9 }; } });
  }
  (again ? Plotly.react : Plotly.newPlot)(el, tr, lay(l), CFG);
  if (!again && tr.some((t) => t.type === "mesh3d" || t.type === "surface" || t.type === "scatter3d")) {
    el.dataset.spin = "1"; ["mousedown", "wheel", "touchstart"].forEach((ev) => el.addEventListener(ev, () => { el.dataset.spin = "0"; }, { passive: true }));
  }
}
let spinT = 0;
function spin(ts) {                                                            // 보이는 장의 3D를 천천히 회전 (마우스를 대면 멈춤)
  if (ts - spinT > 70) {
    spinT = ts;
    const cur = Reveal.getCurrentSlide();
    cur && cur.querySelectorAll('.plot[data-spin="1"]').forEach((el) => {
      const cam = el._fullLayout && el._fullLayout.scene && el._fullLayout.scene.camera; if (!cam) return;
      const e = cam.eye, a = Math.atan2(e.y, e.x) + 0.006, rr = Math.hypot(e.x, e.y);
      Plotly.relayout(el, { "scene.camera.eye": { x: rr * Math.cos(a), y: rr * Math.sin(a), z: e.z } });
    });
  }
  requestAnimationFrame(spin);
}
function countUp(sl) {
  sl.querySelectorAll("[data-count]").forEach((el) => {
    const raw = el.dataset.count, m = raw.match(/-?[\d,]*\.?\d+/); if (!m) return;
    const tgt = parseFloat(m[0].replace(/,/g, "")), dec = (m[0].split(".")[1] || "").length, span = el.querySelector("p span") || el.querySelector("p");
    const t0 = performance.now(), dur = 1100;
    const step = (t) => { const k = Math.min(1, (t - t0) / dur), v = tgt * (1 - Math.pow(1 - k, 3));
      span.textContent = raw.replace(m[0], v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec })); if (k < 1) requestAnimationFrame(step); };
    setTimeout(() => requestAnimationFrame(step), 300);
  });
}
function show(sl) {
  if (!sl) return;
  sl.querySelectorAll(".plot").forEach((el) => { if (!el.dataset.done) { el.dataset.done = 1; setTimeout(() => draw(el), 250); } else Plotly.Plots.resize(el); });
  countUp(sl);
}
Reveal.initialize({ hash: true, width: "100%", height: "100%", margin: 0, minScale: 1, maxScale: 1, center: false, transition: "slide", transitionSpeed: "default",
  backgroundTransition: "fade", slideNumber: false, controls: true, progress: true, plugins: [RevealNotes] }).then(() => { show(Reveal.getCurrentSlide()); requestAnimationFrame(spin); });
Reveal.on("slidechanged", (e) => show(e.currentSlide));
addEventListener("resize", () => { fit(); const cur = Reveal.getCurrentSlide(); cur && cur.querySelectorAll(".plot[data-done]").forEach((el) => draw(el, true)); });
