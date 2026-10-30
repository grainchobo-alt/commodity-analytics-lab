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

// ── 6·7·14장 선 그래프 (PPT 그림과 같은 자료, 선이 그려지는 효과)
const title2 = (t) => ({ text: `<b>${t}</b>`, font: { size: Math.max(10, unit() * 0.13), color: C.g }, x: 0.01, xanchor: "left", y: 0.98 });
const ax2 = (extra) => Object.assign({ gridcolor: "#EEEEEA", zeroline: false, tickfont: { size: Math.max(8, unit() * 0.1) } }, extra);
const lay2 = (t, extra) => Object.assign({ title: title2(t), margin: { l: Math.round(unit() * 0.5), r: Math.round(unit() * 0.35), t: Math.round(unit() * 0.38), b: Math.round(unit() * 0.32) }, hovermode: "x unified" }, extra);
P.rw = () => {
  const R = D.rw, cols = [C.g, C.blue, C.gold, C.red, "#8A8A84", "#6E5A9B"], L = "ABCDEF", last = R.d[R.d.length - 1];
  return [R.y.map((y, k) => ({ x: R.d, y, mode: "lines", line: { color: cols[k], width: 1.6 }, name: L[k], hovertemplate: `${L[k]} %{y:.0f}<extra></extra>` })),
    lay2("6개 중 하나만 실제 옥수수 가격 — 나머지는 동전 던지기", { xaxis: ax2({ range: [R.d[0], "2027-04-01"] }), yaxis: ax2({ title: { text: "2016년 10월 = 100", font: { size: 10 } } }),
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 100, y1: 100, line: { color: "#bbbbb5", width: 0.8 } }],
      annotations: R.y.map((y, k) => ({ x: last, y: y[y.length - 1], text: `<b>${L[k]}</b>`, showarrow: false, xanchor: "left", xshift: 6, font: { color: cols[k], size: 13 } })) })];
};
P.mc = () => {
  const M = D.mc, wk = M.lo.map((_, i) => i + 1), x = [], y = [];
  M.paths.forEach((p) => { x.push(0, ...wk, null); y.push(100, ...p, null); });
  const pct = (v) => `${v - 100 > 0 ? "+" : ""}${Math.round(v - 100)}%`;
  return [[{ x, y, mode: "lines", line: { color: "rgba(138,138,132,0.35)", width: 0.6 }, hoverinfo: "skip" },
    { x: wk, y: M.lo, mode: "lines", line: { color: C.gold, width: 2 }, hovertemplate: "%{x}주 뒤 하단 %{y:.0f}<extra></extra>" },
    { x: wk, y: M.hi, mode: "lines", line: { color: C.gold, width: 2 }, fill: "tonexty", fillcolor: "rgba(200,162,74,0.22)", hovertemplate: "%{x}주 뒤 상단 %{y:.0f}<extra></extra>" }],
    lay2("몬테카를로 시뮬레이션 400회 (옥수수와 같은 출렁임)", { hovermode: "closest", xaxis: ax2({ title: { text: "오늘부터 몇 주 뒤", font: { size: 10 } }, range: [0, 54] }), yaxis: ax2({ title: { text: "오늘 가격 = 100", font: { size: 10 } }, range: [45, 205] }),
      shapes: [{ type: "line", x0: 0, x1: 54, y0: 100, y1: 100, line: { color: C.g, width: 1.2 } }],
      annotations: [[4, 6, 172], [52, 33, 192]].map(([w, ax, ay]) => ({ x: w, y: M.hi[w - 1], ax, ay, axref: "x", ayref: "y", text: `<b>${w}주 뒤 90% 범위<br>${pct(M.lo[w - 1])} ~ ${pct(M.hi[w - 1])}</b>`,
        font: { color: "#7a5d10", size: 11 }, bgcolor: "rgba(255,255,255,0.9)", arrowcolor: "#7a5d10", arrowwidth: 1 })) })];
};
P.vol = () => {
  const V = D.vol, ev = [["2008-06-01", "2009-03-31", "금융위기"], ["2012-06-01", "2012-10-31", "미국 가뭄"], ["2020-03-01", "2020-06-30", "코로나"], ["2022-02-01", "2022-07-31", "우크라이나"]];
  return [[{ type: "bar", x: V.d, y: V.w, marker: { color: "rgba(138,138,132,0.55)" }, name: "주간 수익률 (%)", hovertemplate: "주간 %{y:.1f}%<extra></extra>" },
    { x: V.d, y: V.rv, mode: "lines", line: { color: C.red, width: 2 }, name: "연 변동성 (26주, %)", yaxis: "y2", hovertemplate: "연 변동성 %{y:.0f}%<extra></extra>" }],
    lay2("옥수수: 큰 움직임은 몰려서 옴 (변동성 군집)", { showlegend: true, legend: { orientation: "h", x: 1, xanchor: "right", y: 1.06, font: { size: 9 } }, bargap: 0,
      xaxis: ax2({}), yaxis: ax2({ title: { text: "주간 수익률 %", font: { size: 10 } } }), yaxis2: ax2({ overlaying: "y", side: "right", showgrid: false, title: { text: "연 변동성 %", font: { size: 10, color: C.red } } }),
      shapes: ev.map(([a, b]) => ({ type: "rect", xref: "x", yref: "paper", x0: a, x1: b, y0: 0, y1: 1, fillcolor: "rgba(200,162,74,0.18)", line: { width: 0 }, layer: "below" })),
      annotations: ev.map(([a, , t]) => ({ x: a, y: 0.97, xref: "x", yref: "paper", text: t, showarrow: false, xanchor: "left", font: { size: 9, color: "#7a5d10" } })) })];
};
P.strat = () => {
  const S = D.strat, cols = [C.g, C.blue, C.gold];
  return [[...S.y.map((y, k) => ({ x: S.d, y, mode: "lines", line: { color: cols[k], width: 1.6 }, name: S.names[k], hovertemplate: `${S.names[k]} %{y:.0f}%<extra></extra>` })),
    { x: S.d, y: S.mix, mode: "lines", line: { color: C.red, width: 2.8 }, name: "가치 혼합 (목표 변동성 10%)", hovertemplate: "가치 혼합 %{y:.0f}%<extra></extra>" }],
    lay2("2007–2026 누적 수익 (각 목표 변동성 10%, 거래비용 차감)", { showlegend: true, legend: { orientation: "h", x: 0.01, y: 0.93, font: { size: 9 }, bgcolor: "rgba(255,255,255,0.6)" },
      xaxis: ax2({}), yaxis: ax2({ title: { text: "누적 수익 (%, 합산)", font: { size: 10 } } }) })];
};
// ── 7장 학습 vs 시험 성적 3D 막대 (개념 예시)
P.ovf = () => {
  const O = [["통계·계량(선형)", 60, 56], ["머신러닝 단독", 90, 40], ["통계 + 머신러닝", 72, 63]], B = [], tx = [], ty = [], tz = [], tt = [], tc = [];
  O.forEach(([nm, tr, te], i) => {
    B.push({ x: i, y: 1, z: tr, c: "#8DB7CF", h: `${nm}<br>학습 성적 ${tr}` }, { x: i, y: 0, z: te, c: i === 1 ? C.red : C.g, h: `${nm}<br>시험 성적 ${te}` });
    tx.push(i, i); ty.push(1, 0); tz.push(tr + 7, te + 7); tt.push(`<b>${tr}</b>`, `<b>${te}</b>`); tc.push(C.blue, C.ink);
  });
  return [[bars3d(B, 0.55, 0.55), { type: "scatter3d", mode: "text", x: tx, y: ty, z: tz, text: tt, textfont: { size: 13, color: tc }, hoverinfo: "skip" }],
    { scene: { xaxis: ax3("", { tickvals: [0, 1, 2], ticktext: O.map((o) => o[0]) }), yaxis: ax3("", { tickvals: [0, 1], ticktext: ["시험 성적", "학습 성적"] }),
      zaxis: ax3("맞힌 정도 (예시)", { range: [0, 105] }), aspectmode: "manual", aspectratio: { x: 1.7, y: 1, z: 0.9 }, near: true, camera: { eye: { x: -1.45, y: -2.15, z: 1.05 } } } }];
};

// ── 7장: 섞기 vs 시간 분리 3D 블록 · 선형 평면 vs 머신러닝 곡면
P.split = () => {
  const mix = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], tm = Array(10).fill(0).concat([1, 1, 1, 1]), B = [], TR = "#3B8EA5", TE = "#F2A541";
  [[1, tm, "시간 순 분리"], [0, mix, "무작위로 섞기"]].forEach(([row, seq, nm]) => seq.forEach((m, i) => B.push({ x: i, y: row, z: 0.5, c: m ? TE : TR, h: `${nm}<br>${m ? "시험" : "학습"} 구간` })));
  B.push({ x: 15.2, y: 1, z: 2.0, c: "#2E7D52", h: "시간 분리 AUC 0.50 (실제 실력)" }, { x: 15.2, y: 0, z: 2.76, c: C.red, h: "섞기 AUC 0.69 (부풀려짐)" });
  return [[bars3d(B, 0.85, 0.62), { type: "scatter3d", mode: "text", x: [15.2, 15.2], y: [1.05, -0.05], z: [2.45, 3.25], text: ["<b>AUC 0.50</b>", "<b>AUC 0.69</b>"], textfont: { size: 13, color: ["#2E7D52", C.red] }, hoverinfo: "skip" }],
    { scene: { xaxis: ax3("시간 →", { showticklabels: false }), yaxis: ax3("", { tickvals: [0, 1], ticktext: ["무작위로 섞기", "시간 순 분리"] }), zaxis: ax3("", { showticklabels: false }),
      aspectmode: "manual", aspectratio: { x: 2.2, y: 1.1, z: 0.9 }, near: true, camera: { eye: { x: -1.35, y: -2.4, z: 1.25 } } } }];
};
P.linfit = () => {
  const L = D.linfit;
  return [[{ type: "surface", x: L.g, y: L.g, z: L.plane, colorscale: [[0, "#9CC3D5"], [1, "#2F6E8F"]], opacity: 0.8, showscale: false, hovertemplate: "통계 평면 %{z:.2f}<extra></extra>" },
    { type: "scatter3d", mode: "markers", x: L.x, y: L.y, z: L.z, marker: { size: 0.55, color: C.g, opacity: 0.9 }, hovertemplate: "수익률 %{z:.2f}<extra></extra>" }],
    { scene: { xaxis: ax3("설명변수 1"), yaxis: ax3("설명변수 2"), zaxis: ax3("수익률"), camera: { eye: { x: 1.5, y: -1.5, z: 0.6 } } } }];
};
P.mlfit = () => {
  const M = D.mlfit;
  return [[{ type: "surface", x: M.g, y: M.g, z: M.surf, colorscale: "Viridis", opacity: 0.92, showscale: false, hovertemplate: "머신러닝 곡면 %{z:.2f}<extra></extra>" },
    { type: "surface", x: M.g, y: M.g, z: M.plane, colorscale: [[0, "#9a9a9a"], [1, "#9a9a9a"]], opacity: 0.4, showscale: false, hoverinfo: "skip" },
    { type: "scatter3d", mode: "markers", x: M.x, y: M.y, z: M.z, marker: { size: 0.5, color: C.red, opacity: 0.9 }, hovertemplate: "수익률 %{z:.2f}<extra></extra>" }],
    { scene: { xaxis: ax3("설명변수 1"), yaxis: ax3("설명변수 2"), zaxis: ax3("수익률"), camera: { eye: { x: 1.5, y: -1.5, z: 0.8 } } } }];
};

// ── 3대 전략 직관 (charts_v4 c55·c56·c57과 같은 식)
P.mom = () => {                                                                 // 추세는 이어진다: 지난 1년(면) 방향이 다음 반년(점선)에도 지속
  const t = seq(79, (q) => q), S = [1.0, 0.55, -0.5, -0.95], tr = [];
  S.forEach((s_, k) => {
    const z = t.map((q) => s_ * q * 0.4 + 3 * Math.sin(q / 6 + k)), col = s_ > 0 ? "#2E7D52" : C.red, nm = s_ > 0 ? "오른 품목 → 사기" : "내린 품목 → 팔기";
    tr.push(...curtain(t.slice(0, 53), k, z.slice(0, 53), col, 0.55, nm));
    tr.push({ type: "scatter3d", mode: "lines", x: t.slice(52), y: t.slice(52).map(() => k), z: z.slice(52), line: { color: col, width: 5, dash: "dash" }, opacity: 0.7, hovertemplate: `${nm}<br>추세 지속 %{z:.0f}<extra></extra>` });
  });
  tr.push({ type: "mesh3d", x: [52, 52, 52, 52], y: [-0.5, 3.5, 3.5, -0.5], z: [-42, -42, 42, 42], i: [0, 0], j: [1, 2], k: [2, 3], color: C.gold, opacity: 0.18, hoverinfo: "skip" },
    { type: "scatter3d", mode: "text", x: [52, 72, 72], y: [3.9, 0, 3], z: [44, 36, -36], text: ["<b>오늘</b>", "<b>계속 ↑</b>", "<b>계속 ↓</b>"], textfont: { size: 11, color: [C.ink, "#2E7D52", C.red] }, hoverinfo: "skip" });
  return [tr, { scene: { xaxis: ax3("", { tickvals: [0, 52, 78], ticktext: ["1년 전", "오늘", "반년 뒤"] }), yaxis: ax3("", { showticklabels: false }), zaxis: ax3("수익 (%)", { showticklabels: false }),
    aspectmode: "manual", aspectratio: { x: 2.3, y: 1, z: 0.8 }, near: true, camera: { eye: { x: -0.7, y: -1.45, z: 0.55 } } } }];
};
P.minvar = () => {                                                              // 세 품목 비중 × 전체 출렁임 = 그릇, 달걀이 자리 잡는 바닥 = 최소분산
  const vol = [25, 20, 30], cor = [[1, 0.2, 0.1], [0.2, 1, 0.3], [0.1, 0.3, 1]], g = seq(81, (q) => q / 80);
  const V = g.map((b) => g.map((a) => { const w = [a, b, 1 - a - b]; if (w[2] < -1e-9) return null; let v = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) v += w[i] * w[j] * vol[i] * vol[j] * cor[i][j]; return Math.sqrt(v); }));
  let m = [0, 0, 99]; V.forEach((row, bi) => row.forEach((v, ai) => { if (v !== null && v < m[2]) m = [g[ai], g[bi], v]; }));
  const ex = [], ey = [], ez = [], N = 18;                                       // 달걀: 아래가 조금 통통한 회전체
  for (let p = 0; p <= N; p++) { const th = Math.PI * p / N, rr = Math.sin(th) * (1 + 0.18 * Math.cos(th)); ex.push([]); ey.push([]); ez.push([]);
    for (let q = 0; q <= N; q++) { const ph = 2 * Math.PI * q / N; ex[p].push(m[0] + 0.08 * rr * Math.cos(ph)); ey[p].push(m[1] + 0.08 * rr * Math.sin(ph)); ez[p].push(m[2] + 2.4 - 2.4 * Math.cos(th)); } }
  const corner = [[1, 0, 25, "A만"], [0, 1, 20, "B만"], [0, 0, 30, "C만"]];
  return [[{ type: "surface", x: g, y: g, z: V, colorscale: [[0, "#2F6E8F"], [0.5, "#9CC3D5"], [1, "#F3EEDC"]], opacity: 0.9, showscale: false, hovertemplate: "출렁임 %{z:.1f}%<extra></extra>" },
    { type: "surface", x: ex, y: ey, z: ez, colorscale: [[0, "#E9D3A6"], [1, "#FFF7E6"]], surfacecolor: ez, showscale: false, hovertemplate: `최소분산: 출렁임 ${m[2].toFixed(1)}%<br>세 품목을 고르게 섞은 바닥<extra></extra>`, lighting: { ambient: 0.65, diffuse: 0.8, specular: 0.4, roughness: 0.4 } },
    { type: "scatter3d", mode: "markers", x: corner.map((c) => c[0]), y: corner.map((c) => c[1]), z: corner.map((c) => c[2]), text: corner.map((c) => `한 바구니에 몰빵(${c[3]})`),
      marker: { size: 4, color: C.red }, textfont: { size: 9, color: C.red }, hovertemplate: "%{text}<br>출렁임 %{z:.0f}%<extra></extra>" },
    { type: "scatter3d", mode: "text", x: [m[0]], y: [m[1]], z: [m[2] + 7.5], text: ["<b>달걀 = 섞으면 바닥 (최소분산)</b>"], textfont: { size: 11, color: C.g }, hoverinfo: "skip" }],
    { scene: { xaxis: ax3("A 비중", { showticklabels: false }), yaxis: ax3("B 비중", { showticklabels: false }), zaxis: ax3("출렁임", { showticklabels: false }),
      aspectmode: "manual", aspectratio: { x: 1.3, y: 1.3, z: 0.75 }, near: true, camera: { eye: { x: -1.0, y: -0.95, z: 0.6 } } } }];
};
P.value = () => {                                                               // 평균에서 멀어질수록 당겨지는 고무줄: 싸면 사고(초록) 비싸면 팔고(빨강)
  const t = seq(121, (q) => q), dv = t.map((q) => 18 * Math.sin(q / 9) + 6 * Math.sin(q / 3.7)), x = [], y = [], z = [], c = [], i = [], j = [], k = [];
  t.forEach((q) => { x.push(q, q); y.push(0, 0); z.push(0, dv[q]); c.push(0, dv[q]); });
  for (let q = 0; q < t.length - 1; q++) { const a = 2 * q; i.push(a, a + 1); j.push(a + 1, a + 3); k.push(a + 2, a + 2); }
  const RG = [[0, "#2E7D52"], [0.5, "#F7F6F2"], [1, C.red]], band = t.filter((q) => q % 3 === 0 && Math.abs(dv[q]) > 9);
  return [[{ type: "mesh3d", x, y, z, i, j, k, intensity: c, colorscale: RG, cmin: -24, cmax: 24, showscale: false, opacity: 0.85, flatshading: true, hoverinfo: "skip" },
    { type: "scatter3d", mode: "lines", x: t, y: t.map(() => 0), z: dv, line: { color: C.ink, width: 4 }, hovertemplate: "평균 대비 %{z:+.0f}%<extra></extra>" },
    { type: "surface", x: [-2, 122], y: [-1.2, 1.2], z: [[0, 0], [0, 0]], colorscale: [[0, C.gold], [1, C.gold]], opacity: 0.25, showscale: false, hoverinfo: "skip" },
    { type: "scatter3d", mode: "markers", x: band, y: band.map(() => 0), z: band.map((q) => dv[q]), marker: { size: 3.5, color: band.map((q) => (dv[q] > 0 ? C.red : "#2E7D52")) },
      hovertemplate: "%{z:+.0f}% 벗어남 → 평균으로 당겨짐<extra></extra>" },
    { type: "scatter3d", mode: "text", x: [0, 30, 60], y: [1.3, 0, 0], z: [3, 30, -32], text: ["5년 평균", "<b>비싸면 팔기</b>", "<b>싸면 사기</b>"], textfont: { size: 10, color: [C.gold, C.red, "#2E7D52"] }, hoverinfo: "skip" }],
    { scene: { xaxis: ax3("시간 →", { showticklabels: false }), yaxis: ax3("", { showticklabels: false }), zaxis: ax3("평균 대비", { showticklabels: false }),
      aspectmode: "manual", aspectratio: { x: 2.4, y: 0.7, z: 0.9 }, near: true, camera: { eye: { x: -0.6, y: -1.45, z: 0.5 } } } }];
};

// ── pptx 차트 (14장 선 · 16장 막대)
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
    const small = el.clientWidth < unit() * 4, f = l.scene.near ? 1 : small ? 1.6 : 1.2, e = l.scene.camera.eye; delete l.scene.near;   // near = 카메라를 직접 맞춘 그림
    l.scene.camera.eye = { x: e.x * f, y: e.y * f, z: e.z * f };
    if (small) ["xaxis", "yaxis", "zaxis"].forEach((a) => { if (l.scene[a]) { l.scene[a].tickfont = { size: 8 }; l.scene[a].title.font = { size: 9 }; } });
  }
  if (l.scene) { l.hovermode = el.dataset.hov === "1" ? "closest" : false; l.scene.hovermode = l.hovermode; }   // 3D 말풍선은 마우스를 움직였을 때만
  (again ? Plotly.react : Plotly.newPlot)(el, tr, lay(l), CFG);
  if (!again && tr.some((t) => t.type === "mesh3d" || t.type === "surface" || t.type === "scatter3d")) spinCtl(el);
  if (!again && isSweep(key)) sweep(el);
}
// 3D 회전 제어: 끌기·휠 = 멈추고 조작, 끌지 않고 한 번 클릭 = 멈춤 ↔ 다시 회전
function spinCtl(el) {
  const box = el.parentNode, tip = document.createElement("div"); tip.className = "spin-tip"; box.appendChild(tip);
  const set = (on) => { el.dataset.spin = on ? "1" : "0"; tip.textContent = on ? "⟳ 회전 중 · 클릭하면 멈춤" : "❚❚ 멈춤 · 클릭하면 다시 회전"; box.classList.toggle("paused", !on); };
  let sx = 0, sy = 0, was = true;
  set(true);
  el.addEventListener("mousedown", (e) => { sx = e.clientX; sy = e.clientY; was = el.dataset.spin === "1"; el.dataset.spin = "0"; }, true);
  el.addEventListener("mouseup", (e) => { const click = Math.hypot(e.clientX - sx, e.clientY - sy) < 5; set(click ? !was : false); }, true);
  el.addEventListener("wheel", () => set(false), { capture: true, passive: true });
  el.addEventListener("mousemove", () => { if (el.dataset.hov !== "1") { el.dataset.hov = "1"; Plotly.relayout(el, { hovermode: "closest", "scene.hovermode": "closest" }); } });
  el.addEventListener("mouseleave", () => hovOff(el));
  el.addEventListener("touchstart", (e) => { const t = e.touches[0]; sx = t.clientX; sy = t.clientY; was = el.dataset.spin === "1"; el.dataset.spin = "0"; }, { capture: true, passive: true });
  el.addEventListener("touchend", (e) => { const t = e.changedTouches[0]; set(Math.hypot(t.clientX - sx, t.clientY - sy) < 8 ? !was : false); }, { capture: true, passive: true });
}
// 3D 말풍선 끄기: 회전으로 그래프가 멈춰 있는 마우스 밑을 지나가거나 장을 옮길 때 저절로 뜨지 않게
function hovOff(el) {
  if (el.dataset.hov === "0") return;
  el.dataset.hov = "0"; Plotly.Fx.unhover(el); if (el._fullLayout && el._fullLayout.scene) Plotly.relayout(el, { hovermode: false, "scene.hovermode": false });
}
// 2D 선 그래프: 왼쪽에서 오른쪽으로 선이 그려지는 효과 (장에 올 때마다 다시)
const SWEEP = new Set(["rw", "mc", "vol", "strat"]);
const isSweep = (key) => SWEEP.has(key) || (D.charts && D.charts[key] && D.charts[key].type === "line");
function sweep(el) { el.classList.remove("sweep"); void el.offsetWidth; el.classList.add("sweep"); }
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
let doneT;
function show(sl) {
  if (!sl) return;
  document.querySelectorAll("section.done").forEach((x) => x.classList.remove("done"));      // 다시 오면 등장 효과 재생
  clearTimeout(doneT); doneT = setTimeout(() => sl.classList.add("done"), 3000);
  sl.querySelectorAll(".plot").forEach((el) => {
    if (!el.dataset.done) { el.dataset.done = 1; setTimeout(() => draw(el), 250); }
    else { Plotly.Plots.resize(el); if (isSweep(el.dataset.plot)) setTimeout(() => sweep(el), 250); }
  });
  document.querySelectorAll(".plot[data-spin]").forEach((el) => hovOff(el));             // 장을 옮기면 3D 말풍선 끔
  document.documentElement.classList.toggle("on-pilot", sl.classList.contains("pilot"));   // 테마 시범 장에서만 바깥 여백색도 바꿈
  countUp(sl);
  document.querySelectorAll("video").forEach((v) => { if (!sl.contains(v)) v.pause(); });   // 다른 장의 영상은 멈춤
  sl.querySelectorAll("video").forEach((v) => { v.currentTime = 0; v.play().catch(() => {}); });   // 시연 장에 오면 처음부터 재생
}
Reveal.initialize({ hash: true, width: "100%", height: "100%", margin: 0, minScale: 1, maxScale: 1, center: false, transition: "slide", transitionSpeed: "default",
  backgroundTransition: "fade", slideNumber: false, controls: true, progress: true, plugins: [RevealNotes] }).then(() => {
  document.documentElement.classList.add("anim");                             // 준비된 뒤에만 등장 효과 → 첫 장도 바로 들어옴
  const b = document.getElementById("boot"); b.classList.add("gone"); setTimeout(() => b.remove(), 400);
  show(Reveal.getCurrentSlide()); requestAnimationFrame(spin);
});
Reveal.on("slidechanged", (e) => show(e.currentSlide));
// 테마: T 키로 기본 → 유리·밝게 → 유리·어둡게 순환 (회의실 조명에 맞춰). 주소 뒤 ?theme=glass / glass-dark 로도 지정, 마지막 선택은 이 브라우저에 기억
const THEMES = [["", "기본 디자인"], ["th-glass", "유리 · 밝게"], ["th-glass th-dark", "유리 · 어둡게"]];
let thIdx = 0, thTimer;
function setTheme(k, toast) {
  thIdx = (k + THEMES.length) % THEMES.length;
  const cl = document.documentElement.classList; cl.remove("th-glass", "th-dark"); THEMES[thIdx][0].split(" ").filter(Boolean).forEach((c) => cl.add(c));
  try { localStorage.setItem("mi-theme", thIdx); } catch (e) {}
  if (toast) { const t = $("#thToast"); t.textContent = `테마: ${THEMES[thIdx][1]}  (T 키로 바꾸기 · 지금은 표지·3대 전략·리스크 장에 시범 적용)`; t.classList.add("on"); clearTimeout(thTimer); thTimer = setTimeout(() => t.classList.remove("on"), 2200); }
}
(() => { const q = new URLSearchParams(location.search).get("theme"); let k = 0;
  if (q) k = { glass: 1, "glass-dark": 2 }[q] || 0; else { try { k = +localStorage.getItem("mi-theme") || 0; } catch (e) {} }
  setTheme(k, false); })();
Reveal.addKeyBinding({ keyCode: 84, key: "T", description: "테마 바꾸기 (기본 · 유리 밝게 · 유리 어둡게)" }, () => setTheme(thIdx + 1, true));
// 마우스 휠 = 장 넘김 (3D 그래프 위에서는 휠이 확대·축소). 터치패드 관성으로 여러 장 넘어가지 않게 한 번 멈췄다가 다음 장
let wheelLast = 0, wheelFire = 0;
addEventListener("wheel", (e) => {
  const now = Date.now(), gap = now - wheelLast; wheelLast = now;
  if (e.ctrlKey || (e.target.closest && e.target.closest(".plot[data-spin]")) || Math.abs(e.deltaY) < 4) return;
  if (gap < 200 || now - wheelFire < 650) return;
  wheelFire = now; e.deltaY > 0 ? Reveal.next() : Reveal.prev();
}, { passive: true });
// 시연 영상: 장면 카드를 누르면 그 시각으로 이동, 재생 중인 장면 카드 강조
document.querySelectorAll("section").forEach((sec) => {
  const v = sec.querySelector("video"); if (!v) return;
  const cards = [];
  sec.querySelectorAll(".el.tx").forEach((t) => {
    const m = t.textContent.match(/▶\s*(\d+):(\d\d)/); if (!m) return;
    const at = +m[1] * 60 + +m[2]; t.classList.add("seek"); t.title = "누르면 이 장면으로 이동";
    t.addEventListener("click", () => { v.currentTime = at; v.play().catch(() => {}); });
    cards.push([at, t]);
  });
  v.addEventListener("timeupdate", () => { let cur = null; cards.forEach(([at, t]) => { if (v.currentTime >= at - 0.25) cur = t; }); cards.forEach(([, t]) => t.classList.toggle("on", t === cur)); });
});
// 히트맵: 마우스를 댄 칸의 줄·열만 강조 + 말풍선
document.querySelectorAll(".hx").forEach((box) => {
  const g = box.querySelector(".hx-g"), tip = document.createElement("div"), cx = box.classList.contains("cx"); tip.className = "hx-tip"; box.appendChild(tip);
  const all = [...g.querySelectorAll("[data-r],[data-c]")];
  g.addEventListener("mouseover", (e) => {
    const c = e.target.closest(".hx-v"); if (!c) return;
    const r = c.dataset.r, k = c.dataset.c; g.classList.add("hl");
    all.forEach((x) => x.classList.toggle("on", x.dataset.r === r || x.dataset.c === k || (cx && !x.classList.contains("hx-v") && (x.dataset.r === k || x.dataset.c === r))));
    tip.innerHTML = c.dataset.tip || ""; tip.style.display = c.dataset.tip ? "block" : "none";
    const bb = box.getBoundingClientRect(), cb = c.getBoundingClientRect(), left = cb.left - bb.left + cb.width / 2;
    tip.style.left = Math.min(Math.max(left, 80), bb.width - 80) + "px"; tip.style.top = (cb.top - bb.top - 6) + "px";
  });
  g.addEventListener("mouseleave", () => { g.classList.remove("hl"); all.forEach((x) => x.classList.remove("on")); tip.style.display = "none"; });
});
addEventListener("resize", () => { fit(); const cur = Reveal.getCurrentSlide(); cur && cur.querySelectorAll(".plot[data-done]").forEach((el) => draw(el, true)); });
