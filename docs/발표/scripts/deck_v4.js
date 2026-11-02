// 발표 자료 v4: v3 + 퀀트 학문 지도 정밀판 · AI 카드별 3D 그림 8개 · 연도별 효율적 투자선(겹침 제거) · 품목군 기준 확장 비전 (본문 20장 + 별첨 2장)
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const IMG = (n) => path.join(__dirname, "img", n + ".png");
const OUT = process.argv[2] || path.join(__dirname, "deck_v4.pptx");
const NOTES = require("./notes_v4.js");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "퀀트 기법 기반 곡물·원자재 Market Intelligence (10분 보고 v4)";
const F = "Malgun Gothic";
const G = "1E3A2F", GOLD = "C8A24A", RED = "D9582B", BLUE = "2F6E8F", INK = "222222", MUTED = "6B6B66", TINT = "F1F3EF", LINE = "D6D9D2", WHITE = "FFFFFF";
const W = 13.333;
let page = 0;

// ── 공통 요소 (v1과 같은 디자인)
function base(dark = false) {
  const s = pres.addSlide();
  s.background = { color: dark ? G : WHITE };
  page += 1;
  if (NOTES[page - 1]) s.addNotes(NOTES[page - 1]);
  if (!dark) s.addText(String(page), { x: W - 0.9, y: 7.02, w: 0.5, h: 0.3, fontFace: F, fontSize: 9, color: MUTED, align: "right", margin: 0 });
  return s;
}
function header(s, sec, title, lead) {
  s.addShape(pres.shapes.OVAL, { x: 0.5, y: 0.42, w: 0.56, h: 0.56, fill: { color: GOLD }, line: { color: GOLD } });
  s.addText(sec, { x: 0.5, y: 0.42, w: 0.56, h: 0.56, fontFace: F, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(title, { x: 1.25, y: 0.36, w: 11.4, h: 0.68, fontFace: F, fontSize: 26, bold: true, color: G, valign: "middle", margin: 0 });
  if (lead) s.addText(lead, { x: 1.25, y: 1.04, w: 11.4, h: 0.45, fontFace: F, fontSize: 14, color: MUTED, valign: "middle", margin: 0 });
}
function foot(s, t) {
  s.addText(t, { x: 0.5, y: 6.98, w: 11.6, h: 0.36, fontFace: F, fontSize: 9, color: MUTED, valign: "middle", margin: 0 });
}
function bullets(s, items, o) {
  const arr = items.map((it, i) => {
    const last = i === items.length - 1;
    if (typeof it === "string") return { text: it, options: { bullet: true, breakLine: !last, paraSpaceAfter: 6 } };
    return { text: it.t, options: { bullet: it.b ? false : true, bold: !!it.b, color: it.c || INK, breakLine: !last, paraSpaceAfter: 6 } };
  });
  s.addText(arr, Object.assign({ fontFace: F, fontSize: 14, color: INK, valign: "top", margin: 2 }, o));
}
function card(s, x, y, w, h, head, body, col = G, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: o.fill || TINT }, line: { color: o.fill || TINT }, rectRadius: 0.08 });
  s.addText(head, { x: x + 0.2, y: y + 0.14, w: w - 0.4, h: 0.42, fontFace: F, fontSize: o.hs || 15, bold: true, color: col, valign: "middle", margin: 0 });
  if (body) s.addText(body, { x: x + 0.2, y: y + 0.6, w: w - 0.4, h: h - 0.72, fontFace: F, fontSize: o.bs || 12, color: INK, valign: "top", margin: 0, paraSpaceAfter: 3 });
}
function stat(s, x, y, w, big, label, col = G) {
  s.addText(big, { x, y, w, h: 0.75, fontFace: F, fontSize: 34, bold: true, color: col, margin: 0, valign: "bottom" });
  s.addText(label, { x, y: y + 0.78, w, h: 0.5, fontFace: F, fontSize: 11, color: MUTED, margin: 0, valign: "top" });
}
function dot(s, x, y, col, d = 0.22) { s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: col }, line: { color: col } }); }
function arrow(s, x1, y1, x2, y2, col = MUTED) { s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: col, width: 1.75, endArrowType: "triangle" } }); }
function img(s, n, x, y, w, h) { s.addImage({ path: IMG(n), x, y, w, h, sizing: { type: "contain", w, h } }); }
function table(s, rows, o) {
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || (o.boldFirst && j === 0), color: i === 0 ? WHITE : INK, fill: { color: i === 0 ? G : (i % 2 ? WHITE : TINT) } } }))),
    Object.assign({ fontFace: F, fontSize: 11.5, border: { type: "solid", color: LINE, pt: 0.5 } }, o));
}
const chartBase = { catAxisLabelFontFace: F, valAxisLabelFontFace: F, legendFontFace: F, titleFontFace: F, dataLabelFontFace: F,
  catAxisLabelColor: "55554F", valAxisLabelColor: "55554F", valGridLine: { color: "E5E5E0", size: 0.5 }, catGridLine: { style: "none" },
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, legendFontSize: 11, titleFontSize: 13, titleColor: G, dataLabelFontSize: 10 };
const C = (o) => Object.assign({}, chartBase, o);
const SH = D.sharpe, VT = D.voltgt;

// ════════════════ 1 표지
{
  const s = base(true);
  s.addText("Market Intelligence", { x: 0.8, y: 1.5, w: 11, h: 0.5, fontFace: F, fontSize: 18, color: GOLD, bold: true, margin: 0 });
  s.addText("퀀트 기법 기반 곡물·원자재 시황 분석", { x: 0.8, y: 2.1, w: 11.8, h: 1.0, fontFace: F, fontSize: 40, bold: true, color: WHITE, margin: 0 });
  s.addText("가격 위험을 숫자로 · 구매 시점의 근거 · 장기 포트폴리오 운영 기반", { x: 0.8, y: 3.15, w: 11.8, h: 0.6, fontFace: F, fontSize: 20, color: "DCE3DE", margin: 0 });
  s.addText("식량사업개발실 · 팜사업운영그룹   |   2026. 10.", { x: 0.8, y: 6.3, w: 11, h: 0.4, fontFace: F, fontSize: 13, color: "AFC0B5", margin: 0 });
}

// ════════════════ 2 요약
{
  const s = base();
  header(s, "요약", "한 장 요약", "원자재 약 20개 · 약 20년 자료에 같은 규칙을 적용해 검증한 것만 의사결정에 연결함");
  const cards = [
    ["가격 위험을 금액으로", "$3.2M", "옥수수·대두·대두박 각 3만 톤 매입 예정 시\n20일 동안 100번 중 99번은 이보다 덜 잃음\n→ 헤지·선매입 판단의 기준선", G],
    ["구매 시점의 근거", "+23bp", "'지금 살까, 미룰까' 판단당 평균 절감\n(아무 때나 미루면 −14bp)\n→ 작은 우위도 매주 반복하면 누적 절감", RED],
    ["장기 강건한 포트폴리오", "0.88", "3대 전략(모멘텀·최소분산·가치) 혼합의\n위험 1단위당 수익(Sharpe) — 약 20년 기준 우수\n→ 목표 위험을 높이면 수익도 비례", GOLD],
    ["확장성", "전사", "품목군별로 흩어진 원자재 매매를 묶어 관리하는\n리스크 컨트롤 타워로 확장 가능\n→ 주식·채권 등 모든 가격 자산에 같은 엔진", BLUE],
  ];
  cards.forEach((c, i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.75 + Math.floor(i / 2) * 2.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.95, h: 2.35, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addText(c[1], { x: x + 0.25, y: y + 0.25, w: 2.1, h: 0.9, fontFace: F, fontSize: 30, bold: true, color: c[3], margin: 0, valign: "middle" });
    s.addText(c[0], { x: x + 2.4, y: y + 0.25, w: 3.4, h: 0.9, fontFace: F, fontSize: 17, bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText(c[2], { x: x + 0.25, y: y + 1.2, w: 5.5, h: 1.05, fontFace: F, fontSize: 12.5, color: INK, margin: 0, valign: "top" });
  });
  foot(s, "bp = 0.01%. 위험 금액은 기준일 2026-10-26 가격, 구매 절감은 근거가 확인된 9개 품목의 시험 기간(2017~) 평균, Sharpe는 2007–2026 거래비용 차감 기준.");
}

// ════════════════ 3 배경·목표
{
  const s = base();
  header(s, "Ⅰ", "배경 — 시세를 읽는 역량이 곧 경쟁력", "Back-to-back 거래는 정보 우위가 없어 마진이 Handling charge 수준에 머묾");
  const box = (x, y, t, sub, col) => { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.2, h: 1.0, fill: { color: col }, line: { color: col }, rectRadius: 0.1 });
    s.addText([{ text: t, options: { bold: true, fontSize: 14, breakLine: true } }, { text: sub, options: { fontSize: 10 } }], { x, y, w: 2.2, h: 1.0, fontFace: F, color: WHITE, align: "center", valign: "middle", margin: 2 }); };
  box(0.6, 1.75, "공급사·선사", "FOB 공급가 · 운임 견적", "6E7F74"); box(3.4, 1.75, "당사", "공급가 + 운임 + 마진", G); box(6.2, 1.75, "고객사", "복수 오퍼 중 최저가 선택", BLUE);
  arrow(s, 2.85, 2.1, 3.35, 2.1, G); arrow(s, 5.65, 2.1, 6.15, 2.1, G);
  s.addText("가격 →", { x: 2.8, y: 1.78, w: 0.65, h: 0.28, fontFace: F, fontSize: 9, bold: true, color: G, align: "center", margin: 0 });
  s.addText("오퍼 →", { x: 5.6, y: 1.78, w: 0.65, h: 0.28, fontFace: F, fontSize: 9, bold: true, color: G, align: "center", margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 3.4, y: 2.9, w: 4.9, h: 0, line: { color: MUTED, width: 1, dashType: "dash", beginArrowType: "triangle" } });
  s.addText("← Inquiry (물량·시기) · 고객 → 당사 → 공급사·선사", { x: 1.2, y: 2.95, w: 7.2, h: 0.28, fontFace: F, fontSize: 9.5, color: MUTED, align: "right", margin: 0 });
  bullets(s, [{ t: "지금의 한계", b: true, c: RED }, "고객도 시세를 실시간으로 앎 → 매번 최저가 공급선 선택", "시세 변동 위험은 그대로 노출, 우리만의 판단 근거 부족"], { x: 0.6, y: 3.4, w: 8.0, h: 1.45, fontSize: 13.5 });
  card(s, 0.6, 4.95, 3.95, 1.85, "단기 · 원가 절감", "주간 '지금 살까, 미룰까' 판단 근거\n시세 위험을 물량 금액으로 환산", BLUE, { bs: 12.5 });
  card(s, 4.7, 4.95, 3.95, 1.85, "중장기 · 독립 수익", "여러 원자재를 묶어 운영하는\n포트폴리오 사업의 실험 기반", GOLD, { bs: 12.5 });
  card(s, 8.95, 1.75, 3.85, 5.05, "발표자 배경", "• 식량사업실 곡물 트레이딩 약 2년\n  — '언제·얼마나 살까'를 현장에서 고민\n\n• 퀀트 석사\n  — 시계열·리스크·포트폴리오 이론\n\n• 팜사업운영그룹\n  — 농장 운영·신규 사업·시황 분석\n\n→ 현장 질문에 정량 답을 주는 도구로 구축", G, { bs: 12 });
}

// ════════════════ 퀀트란 — 네 학문의 결합 (정밀판: 원마다 기법 · 겹치는 영역 · 바탕 수학)
{
  const s = base();
  header(s, "Ⅰ", "퀀트(Quant)란 — 네 분야를 금융 문제에 결합", "시장을 '감'이 아닌 숫자로 읽고, 규칙을 검증한 뒤 자동으로 반복하는 방법");
  img(s, "c46_quant_venn", 0.35, 1.5, 6.3, 5.4);
  s.addText("이 대시보드에서 쓴 기법 — 쉬운 말로", { x: 6.85, y: 1.5, w: 6.0, h: 0.35, fontFace: F, fontSize: 13, bold: true, color: G, margin: 0 });
  const rows = [["통계학", BLUE, [["VaR · ES", "'100번 중 99번은 이보다 덜 잃는다'는 손실 기준선"], ["가설 검정", "좋은 결과가 우연(동전 던지기)인지 확인"], ["부트스트랩", "자료를 다시 뽑아 결과가 얼마나 흔들리는지 측정"]]],
    ["계량경제학", G, [["GARCH", "큰 출렁임 뒤엔 큰 출렁임 → 다음 기간 위험 크기 예측"], ["HMM", "평시·위기 국면을 확률로 판별"], ["칼만 필터", "시간에 따라 변하는 관계를 매주 조금씩 보정"]]],
    ["머신러닝", "B8901C", [["Lasso · PCA", "쓸모없는 지표 지우기 · 여러 품목을 큰 흐름으로 요약"], ["랜덤 포레스트", "판단 나무 수백 개의 예측을 평균"], ["LSTM · Chronos · FinBERT", "순서 기억 신경망 · 사전학습 AI · 문서 감성 점수"]]],
    ["프로그래밍", RED, [["Python 수집", "36개 원자재·거시·기상·보고서를 매일 자동 수집"], ["백테스트 엔진", "규칙을 과거에 그대로 적용해 성과 재현"], ["대시보드", "결과를 화면으로 — 누구나 같은 숫자를 봄"]]],
    ["수학 (바탕)", MUTED, [["선형대수", "여러 품목의 위험을 행렬로 한 번에 계산"], ["최적화", "같은 위험에서 수익이 가장 큰 비중 찾기"], ["확률과정", "랜덤워크·브라운 운동으로 가격 경로 모의"]]]];
  rows.forEach(([t, col, items], i) => { const y = 1.92 + i * 1.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y, w: 6.0, h: 0.9, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.06 });
    s.addShape(pres.shapes.RECTANGLE, { x: 6.85, y: y + 0.1, w: 0.07, h: 0.7, fill: { color: col }, line: { color: col } });
    s.addText(t, { x: 7.0, y, w: 1.15, h: 0.9, fontFace: F, fontSize: 12, bold: true, color: col, valign: "middle", margin: 0 });
    s.addText(items.flatMap(([k, v], j) => [{ text: k + "  ", options: { bold: true, color: INK } }, { text: v, options: { color: "4A4A45", breakLine: j < items.length - 1 } }]),
      { x: 8.15, y: y + 0.03, w: 4.65, h: 0.84, fontFace: F, fontSize: 9.5, valign: "middle", margin: 0, paraSpaceAfter: 1 }); });
  foot(s, "어느 한 분야만으로는 부족 — 통계로 의심하고, 계량으로 구조를 보고, 머신러닝으로 패턴을 찾고, 프로그래밍으로 매일 반복. 겹치는 영역 = 두 분야가 만나 생긴 실무 기법.");
}

// ════════════════ 퀀트 전략 16개 — 시험해 보고 통과한 것만 (4행 × 4열, 카드마다 회전 3D · quant16.py · charts_q16.py)
{
  const s = base();
  const Q = D.q16 || [];
  const cnt = (k) => Q.filter((q) => q.판단.startsWith(k)).length;
  header(s, "Ⅰ", "퀀트 전략 16개 — 우리 자료로 모두 시험, 통과한 것만 대시보드에",
    `채택 ${cnt("●")} · 부분 적용 ${cnt("◐")} (위험 관리용) · 시험 후 미채택 ${cnt("✕")} · 해당 없음 ${cnt("—")}  (옵션 전략 2개 제외)`);
  const BADGE = { "●": ["2E7D52", "채택"], "◐": ["B8901C", "부분 적용"], "✕": [RED, "미채택"], "—": ["9A9A94", "해당 없음"] };
  const ROWS = [["추세", 1, 5, 14, 13], ["되돌림", 2, 4, 3, 9], ["비중\n위험", 15, 16, 8, 12], ["데이터\n실행", 17, 7, 18, 6]];
  const x0 = 0.97, y0 = 1.58, cw = 2.91, ch = 1.265, gx = 0.08, gy = 0.07;
  ROWS.forEach(([lab, ...nums], r) => {
    const y = y0 + r * (ch + gy);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y, w: 0.4, h: ch, fill: { color: G }, line: { color: G }, rectRadius: 0.05 });
    s.addText(lab.split("\n").map((t, j, a) => ({ text: t, options: { breakLine: j < a.length - 1 } })),
      { x: 0.5, y, w: 0.4, h: ch, fontFace: F, fontSize: 8, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
    nums.forEach((n, c) => {
      const q = Q.find((o) => o.번호 === n); if (!q) return;
      const x = x0 + c * (cw + gx), [bc, bt] = BADGE[q.판단[0]] || BADGE["—"];
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.05 });
      s.addText(`${n}  ${q.전략}`, { x: x + 0.08, y: y + 0.04, w: cw - 0.95, h: 0.24, fontFace: F, fontSize: 9.5, bold: true, color: G, valign: "middle", margin: 0 });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + cw - 0.84, y: y + 0.06, w: 0.78, h: 0.2, fill: { color: bc }, line: { color: bc }, rectRadius: 0.04 });
      s.addText(bt, { x: x + cw - 0.84, y: y + 0.06, w: 0.78, h: 0.2, fontFace: F, fontSize: 7, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
      s.addImage({ path: path.join(__dirname, "img", `q16_${String(n).padStart(2, "0")}.gif`), x: x + 0.04, y: y + 0.31, w: 1.12, h: 0.91 });
      s.addText([{ text: q["한 줄 설명"], options: { color: INK, breakLine: true } },
                 { text: q["결과 1"], options: { color: G, bold: true, breakLine: true, fontSize: 6.8 } },
                 { text: q["결과 2"], options: { color: MUTED, fontSize: 6.8 } }],
        { x: x + 1.2, y: y + 0.3, w: cw - 1.26, h: ch - 0.34, fontFace: F, fontSize: 7.3, valign: "top", margin: 0, paraSpaceAfter: 2 });
    });
  });
  foot(s, "롱숏 = 19품목 주간 · 비용 5bp · 목표 변동성 10% · 2007–2026, '혼합에 10%' = 기준 혼합(MinVar 50·모멘텀 25·가치 25, Sharpe 0.88)에 추가했을 때. 채택 조건 = 혼합 개선 t ≥ 2 & 두 기간 모두 개선. 그림은 회전 3D(HTML은 클릭하면 직접 돌려 볼 수 있음). 출처: quant16.py · 구매 판단 검증 백로그");
}

// ════════════════ 4 데이터 유형
{
  const s = base();
  header(s, "Ⅱ", "데이터는 두 방향으로 읽음 — 시계열과 횡단면", "같은 표라도 '한 품목의 시간 흐름'으로 보느냐, '같은 날 품목끼리 비교'로 보느냐에 따라 질문이 달라짐");
  img(s, "c50_xs_ts3d", 0.3, 1.5, 8.2, 5.4);
  card(s, 8.6, 1.75, 4.2, 2.35, "시계열 (가로로 읽기)", "한 품목의 가격이 시간에 따라 어떻게 변하나\n• 질문: 다음 달 옥수수는 오를까?\n• 쓰임: 위험 크기 예측, 추세 판단", BLUE, { bs: 12.5 });
  card(s, 8.6, 4.25, 4.2, 2.35, "횡단면 (세로로 읽기)", "같은 시점에 어느 품목이 상대적으로 싼가\n• 질문: 이번 달 무엇부터 살까?\n• 쓰임: 품목 순위, 비슷한 품목 묶기", "9A7A1A", { bs: 12.5 });
  foot(s, "자료: 원자재 12개 월간 수익률(%, 롤 보정), 2025-11 ~ 2026-10(10월은 26일까지). 막대 높이는 ±20%에서 자름, 숫자는 실제 값.");
}

// ════════════════ 5 시계열은 왜 예측이 어려운가
{
  const s = base();
  header(s, "Ⅱ", "시계열 가격은 왜 완벽히 예측할 수 없나", "가격 변화의 대부분은 아직 오지 않은 뉴스(기상·정책·수급)가 만듦 — 랜덤워크(무작위 걸음)에 가까움");
  img(s, "c41_randomwalk", 0.4, 1.6, 6.2, 4.0);
  img(s, "c42_brownian", 6.75, 1.6, 6.2, 4.0);
  s.addText(`정답: ${D.rw_answer} — 진짜와 동전 던지기를 눈으로 구별하기 어려움`, { x: 0.5, y: 5.6, w: 6.1, h: 0.35, fontFace: F, fontSize: 11.5, bold: true, color: RED, margin: 0 });
  s.addText("오늘 가격에서 출발해 무작위 충격을 400번 이어 붙인 결과(브라운 운동)", { x: 6.85, y: 5.6, w: 6.1, h: 0.35, fontFace: F, fontSize: 11.5, bold: true, color: "7A5D10", margin: 0 });
  bullets(s, [`지난주 등락으로 이번 주 등락을 설명하는 힘 ≈ 0 (12개 품목 평균 상관 ${D.ac1_mean_abs.toFixed(2)}) → 아무리 정교한 모델도 '정답'을 맞히기는 어려움`,
    `불확실성은 시간의 제곱근으로 커짐: 옥수수 4주 뒤 ${D.band4[0]}% ~ +${D.band4[1]}%, 1년 뒤 ${D.band52[0]}% ~ +${D.band52[1]}% (90% 범위)`,
    "그래서 목표는 '맞히기'가 아니라 ① 위험의 크기 관리 ② 오래 반복된 작은 경향의 확률적 우위 ③ 섞어서 출렁임 줄이기"], { x: 0.5, y: 6.0, w: 12.3, h: 0.95, fontSize: 11.5 });
}

// ════════════════ 5-2 머신러닝의 강점(왼쪽) → 그럼에도 단독으로 쓰면 안 되는 두 가지 함정(오른쪽)
{
  const s = base();
  header(s, "Ⅱ", "머신러닝은 곡선에 강함 — 그러나 단독으로 쓰면 함정", "곡선 자료는 머신러닝이 유리 — 단, 미래 정보가 새면 성적이 부풀고, 단독으론 잡음까지 외움");
  // 왼쪽: 강점 — 선형 평면(통계·계량) vs 머신러닝 곡면
  s.addText("① 강점: 직선이면 통계·계량, 넓게 퍼지고 방향이 바뀌면 머신러닝", { x: 0.5, y: 1.62, w: 6.0, h: 0.35, fontFace: F, fontSize: 12.5, bold: true, color: G, margin: 0 });
  img(s, "c53_linfit3d", 0.5, 2.02, 2.95, 2.7);
  img(s, "c54_mlfit3d", 3.55, 2.02, 2.95, 2.7);
  s.addText("통계·계량: 점들을 관통하는 평면(직선)으로 설명 — 관계가 일정할 때 안정적", { x: 0.5, y: 4.74, w: 2.95, h: 0.62, fontFace: F, fontSize: 9.5, color: BLUE, bold: true, margin: 0, valign: "top" });
  s.addText("머신러닝: 범위가 넓고 방향이 계속 바뀌면 곡면으로 근사 (회색 평면 = 직선은 못 따라감)", { x: 3.55, y: 4.74, w: 2.95, h: 0.62, fontFace: F, fontSize: 9.5, color: RED, bold: true, margin: 0, valign: "top" });
  bullets(s, ["직선 관계면 통계·계량으로 충분하고 '왜'를 설명하기 쉬움",
    "곡선·방향 전환이 잦은 자료는 머신러닝이 훨씬 잘 따라감 → 설명력이 높음",
    { t: "그런데 잘 따라가는 힘이 곧 약점 — 오른쪽 두 가지 함정", b: true, c: RED }],
    { x: 0.5, y: 5.42, w: 6.0, h: 1.5, fontSize: 10.5 });
  s.addShape(pres.shapes.LINE, { x: 6.67, y: 1.62, w: 0, h: 5.25, line: { color: LINE, width: 1 } });
  // 오른쪽: 그럼에도 단독 사용은 위험 — 함정 두 가지
  s.addText("② 그럼에도 머신러닝만 쓰면 안 되는 이유", { x: 6.85, y: 1.62, w: 6.0, h: 0.35, fontFace: F, fontSize: 12.5, bold: true, color: G, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y: 2.0, w: 6.0, h: 0.45, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
  s.addText([{ text: "비유  ", options: { bold: true, color: G } }, { text: "시험 문제를 미리 보고 공부하면 점수가 높게 나옴 → 실력이 아니라 '미리 본 답'", options: { color: INK } }],
    { x: 7.0, y: 2.0, w: 5.75, h: 0.45, fontFace: F, fontSize: 10, valign: "middle", margin: 0 });
  [["함정 ① 섞어서 시험 (실제 실험)", "c52_split3d", 6.85,
    [{ text: "✕ 섞기: 미래 자료가 학습에 섞여 AUC 0.69로 부풀려짐", options: { color: RED, bold: true, breakLine: true } },
     { text: "○ 시간 분리: 처음 보는 미래로만 채점 → 0.50 (동전 던지기 = 실제 실력)", options: { color: "2E7D52", bold: true } }]],
   ["함정 ② 단독 사용 (개념 예시)", "c51_ml_overfit3d", 9.9,
    [{ text: "✕ 머신러닝 단독: 학습 90 → 시험 40", options: { color: RED, bold: true, breakLine: true } },
     { text: "○ 통계를 뼈대로, 머신러닝은 보조로 접목: 학습 72 · 시험 63으로 안정", options: { color: G, bold: true } }]]].forEach(([t, im, x, tx]) => {
    s.addText(t, { x, y: 2.55, w: 2.95, h: 0.3, fontFace: F, fontSize: 10.5, bold: true, color: G, margin: 0 });
    img(s, im, x, 2.88, 2.95, 2.62);
    s.addText(tx, { x, y: 5.55, w: 2.95, h: 1.0, fontFace: F, fontSize: 9.5, margin: 0, valign: "top", paraSpaceAfter: 3 });
  });
  foot(s, "AUC = 맞힌 정도(0.5 = 동전 던지기). 0.69 · 0.50 = 대시보드 실험(LightGBM 구매 분류기, 2001–2011 학습 표본을 무작위 K-fold vs 기간 분리 Purged K-fold로 평가). 그 밖의 그림은 개념 예시(가상 자료).");
}

// ════════════════ 6 그래도 예측 가능한 것
{
  const s = base();
  header(s, "Ⅱ", "방향은 어려워도, 이것들은 숫자로 다룰 수 있음", "학계·헤지펀드가 수십 년간 확인한 세 가지 규칙성을 원자재 약 20개에 적용");
  img(s, "c09_volatility", 0.4, 1.6, 7.2, 3.7);
  card(s, 7.9, 1.7, 4.9, 1.6, "① 위험의 크기는 몰려옴", "큰 출렁임 뒤엔 큰 출렁임 → 다음 주·다음 달 위험 폭을 미리 계산 (왼쪽: 2008·2012·2020·2022)", G, { bs: 12 });
  card(s, 7.9, 3.45, 4.9, 1.6, "② 오래 반복된 작은 경향", "오르던 것은 조금 더 오르고(추세), 너무 싸진 것은 제자리로 돌아오려 함(가치)", BLUE, { bs: 12 });
  card(s, 7.9, 5.2, 4.9, 1.6, "③ 함께 덜 움직이는 것끼리 섞기", "곡물·에너지·금속은 서로 다르게 움직임 → 섞으면 같은 수익에 출렁임 감소", GOLD, { bs: 12 });
  bullets(s, ["①은 리스크 관리(위험 금액·구매 물량)에, ②·③은 구매 판단과 포트폴리오 전략에 사용", "모든 결과는 '과거로 배우고 그 뒤 기간으로 시험'해 통과한 것만 채택 (미래 정보 차단)"], { x: 0.5, y: 5.5, w: 7.2, h: 1.4, fontSize: 12.5 });
  foot(s, "왼쪽 그림: 옥수수 주간 수익률과 26주 변동성(연율), 음영 = 금융위기·미국 가뭄·코로나·우크라이나.");
}

// ════════════════ AI 모델 ① 숫자 자료로 배우는 머신러닝 — 카드마다 대응하는 3D 그림
function mcards(s, ms) {
  ms.forEach((m, i) => {
    const x = 0.5 + i * 3.1, y = 1.6, w = 3.0, h = 5.25;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addShape(pres.shapes.RECTANGLE, { x, y: y + 0.12, w: 0.08, h: 0.45, fill: { color: m[3] }, line: { color: m[3] } });
    s.addText(m[0], { x: x + 0.2, y: y + 0.08, w: w - 0.3, h: 0.5, fontFace: F, fontSize: 13.5, bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText(m[1], { x: x + 0.2, y: y + 0.62, w: w - 0.35, h: 0.72, fontFace: F, fontSize: 10.5, color: INK, margin: 0, valign: "top" });
    s.addText(m[2], { x: x + 0.2, y: y + 1.34, w: w - 0.35, h: 0.3, fontFace: F, fontSize: 9, color: MUTED, margin: 0, valign: "top" });
    s.addShape(pres.shapes.RECTANGLE, { x: x + 0.08, y: y + 1.7, w: w - 0.16, h: 2.65, fill: { color: WHITE }, line: { color: WHITE } });
    img(s, m[4], x + 0.1, y + 1.72, w - 0.2, 2.6);
    s.addText(m[5], { x: x + 0.2, y: y + 4.4, w: w - 0.35, h: 0.8, fontFace: F, fontSize: 9.5, color: G, margin: 0, valign: "top" });
  });
}
{
  const s = base();
  header(s, "Ⅱ", "활용한 모델 ① — 숫자 자료에서 패턴을 배우는 머신러닝", "가격·거시 지표 같은 숫자 자료로 '어떤 조건에서 어떻게 움직였나'를 학습");
  mcards(s, [
    ["쓸모 있는 변수 고르기", "후보 지표가 많으면 우연히 맞은 지표까지 섞임 → 영향이 작은 지표를 0으로 지워 꼭 필요한 것만 남김", "Lasso · Ridge · Elastic Net", G, "c47a_lasso3d",
      "그림: 걸러내는 강도를 높일수록 회색(지워진 지표)이 늘고 원유·재고·포지션만 남음 (예시)"],
    ["공통 흐름으로 요약", "함께 움직이는 품목들의 공통 움직임을 몇 개의 '큰 흐름'으로 압축", "PCA · PLS · 요인 모형", G, "c47b_pca3d",
      "그림: 실제 옥수수·대두·밀 주간 수익률 — 점들이 한 방향(빨간 축)으로 늘어서 공통 흐름 하나가 71%를 설명"],
    ["판단 나무 숲 (랜덤 포레스트)", "조건('가격이 1년 범위 하위 25%인가?')으로 자료를 나누는 판단 나무 수백 개의 예측을 평균", "Random Forest · XGBoost · LightGBM", BLUE, "c47c_rf3d",
      "그림: 조건 구간마다 높이가 다른 계단 지형 — '아주 싸고 변동성 클 때만 반등' 같은 관계 포착 (예시)"],
    ["순서를 기억하는 신경망", "과거 흐름을 순서대로 읽으며 무엇을 오래 기억할지 스스로 학습", "LSTM · TFT · PatchTST · iTransformer", BLUE, "c47d_lstm3d",
      "그림: 최근 몇 주와 1년 전(계절)을 많이 기억 — 출렁임이 큰 시기엔 최근에 더 집중 (예시)"]]);
  foot(s, "예시 = 개념 설명용 합성 자료, 실제 = 2007–2026 주간 자료. 복잡한 모델일수록 과거에 과하게 맞출 위험 → 모든 모델은 학습에 안 쓴 이후 기간으로 시험.");
}

// ════════════════ AI 모델 ② AI · 위험 · 국면 — 카드마다 대응하는 3D 그림
{
  const s = base();
  header(s, "Ⅱ", "활용한 모델 ② — 사전학습 AI · 문서 AI · 위험과 국면", "최신 AI는 '읽고 요약'하는 데, 통계 모형은 '위험의 크기와 국면'을 재는 데 강점");
  mcards(s, [
    ["사전학습 AI", "수많은 시계열을 미리 학습한 '시계열판 ChatGPT' — 추가 학습 없이 바로 예측", "Chronos (아마존) · TimesFM (구글)", RED, "c48a_chronos3d",
      "그림: 답을 한 점이 아니라 '가능성 분포'로 냄 — 먼 미래일수록 넓게 퍼짐 (옥수수 출렁임 기준 예시)"],
    ["문서를 읽는 AI", "USDA·EIA 보고서, 연준 경기 보고서 문장을 긍정·부정 점수로 바꿈", "FinBERT (금융 특화 언어 모델)", RED, "c48b_finbert3d",
      "그림: 실제 USDA 수급 보고서(WASDE) 품목별 연평균 어조 — 초록 = 긍정, 주황 = 부정"],
    ["위험 크기 예측", "큰 출렁임이 몰려오는 성질로 다음 기간의 위험 폭을 계산", "GARCH 계열 · MIDAS", GOLD, "c48c_garch3d",
      "그림: 실제 6개 품목의 GARCH 예측 위험 — 2008·2012 가뭄·2020·2022에 동시에 솟음"],
    ["시장 국면 판별", "보이지 않는 평시·위기 국면을 수익률 움직임으로 확률 추정", "국면 전환 모형(HMM) · 칼만 필터", GOLD, "c48d_hmm3d",
      "그림: 실제 원자재 평균 수익률로 추정한 위기 확률 — 2008–09·2011·2022에 높음"]]);
  foot(s, "최신 AI도 가격 '방향'을 꾸준히 맞히지는 못함 → 위험 계산·국면 판별·보조 신호로 활용, 시험 기간에 개선이 없는 모델은 채택하지 않음. 위험 = 주간 GARCH(1,1) 연율, 월평균.");
}

// ════════════════ 멀티모달 흐름도
{
  const s = base();
  header(s, "Ⅱ", "멀티모달 — 숫자·기상·문서를 하나의 신호로", "성격이 다른 자료를 모두 '주간 숫자'로 바꿔 모델에 넣고, 신호 → 전략 → 비중까지 연결");
  const ins = [["가격 시계열", "원자재 약 20개 선물 (일·주간)", G], ["거시·금융", "금리·달러·VIX·신용 스프레드", G], ["기상", "엘니뇨 · 주산지 강수·기온", BLUE], ["보고서 수치", "USDA 재고·작황 · EIA 재고", BLUE], ["시장 포지션", "CFTC 투기·헤지 포지션(COT)", BLUE], ["문서(텍스트)", "WASDE · EIA 전망 · 연준 베이지북", RED]];
  ins.forEach(([t, sub, col], i) => { const y = 1.7 + i * 0.83;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y, w: 2.75, h: 0.7, fill: { color: WHITE }, line: { color: col, width: 1.25 }, rectRadius: 0.08 });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: y + 0.1, w: 0.07, h: 0.5, fill: { color: col }, line: { color: col } });
    s.addText([{ text: t, options: { bold: true, fontSize: 11.5, color: INK, breakLine: true } }, { text: sub, options: { fontSize: 9, color: MUTED } }], { x: 0.68, y, w: 2.55, h: 0.7, fontFace: F, valign: "middle", margin: 0 });
    if (i < 5) arrow(s, 3.3, y + 0.35, 5.15, y + 0.35, LINE); });
  s.addText("입력 6갈래", { x: 0.5, y: 6.68, w: 2.75, h: 0.28, fontFace: F, fontSize: 10, bold: true, color: MUTED, align: "center", margin: 0 });
  // 문서 → LLM 필터
  const ty = 1.7 + 5 * 0.83;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.5, y: ty - 0.05, w: 1.5, h: 0.8, fill: { color: "FBEDE7" }, line: { color: RED, width: 1.25 }, rectRadius: 0.08 });
  s.addText([{ text: "LLM 필터", options: { bold: true, fontSize: 11, color: RED, breakLine: true } }, { text: "FinBERT: 문장 →\n긍정·부정 점수", options: { fontSize: 8.5, color: INK } }], { x: 3.5, y: ty - 0.05, w: 1.5, h: 0.8, fontFace: F, align: "center", valign: "middle", margin: 0 });
  arrow(s, 3.27, ty + 0.35, 3.47, ty + 0.35, RED); arrow(s, 5.02, ty + 0.35, 5.15, ty + 0.35, RED);
  // 숫자로 정리
  const col = (x, w, head, body, c) => { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.7, w, h: 4.9, fill: { color: TINT }, line: { color: c, width: 1.25 }, rectRadius: 0.08 });
    s.addText(head, { x: x + 0.1, y: 1.85, w: w - 0.2, h: 0.55, fontFace: F, fontSize: 13, bold: true, color: c, align: "center", margin: 0 });
    s.addText(body, { x: x + 0.15, y: 2.5, w: w - 0.3, h: 4.0, fontFace: F, fontSize: 10.5, color: INK, margin: 0, valign: "top", paraSpaceAfter: 6 }); };
  col(5.2, 1.95, "① 숫자로 정리", "• 발표된 뒤의 날짜에만 사용\n  (미래 정보 차단)\n\n• 모두 주간으로 맞춤\n\n• 품목별로 '평소 대비 얼마나 높은가'로 표준화\n\n• 품목마다 해당 자료만 연결\n  (곡물 ↔ 작황·주산지 기상)", G);
  arrow(s, 7.17, 4.15, 7.45, 4.15, G);
  col(7.5, 1.85, "② 모델 학습", "• 머신러닝·AI 시계열\n  (앞 두 장의 모델)\n\n• 과거로 학습 →\n  이후 기간으로 시험\n\n• 갈래별 효과를\n  따로·함께 비교", BLUE);
  arrow(s, 9.37, 4.15, 9.65, 4.15, BLUE);
  // 출력 3단
  const outs = [["③ 품목별 신호", "이번 주 강세·약세 점수와 순위", BLUE], ["④ 포트폴리오 전략", "3대 전략(모멘텀·최소분산·가치)에\n신호를 섞어 볼지 판단", GOLD], ["⑤ 목표 위험에 맞춘 비중", "연 변동성 목표(예: 10%)로\n품목별 매수·매도 비중 결정", RED]];
  outs.forEach(([t, sub, c], i) => { const y = 1.7 + i * 1.7;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.7, y, w: 3.1, h: 1.4, fill: { color: c }, line: { color: c }, rectRadius: 0.08 });
    s.addText([{ text: t, options: { bold: true, fontSize: 12.5, color: WHITE, breakLine: true } }, { text: sub, options: { fontSize: 10, color: WHITE } }], { x: 9.8, y, w: 2.9, h: 1.4, fontFace: F, valign: "middle", margin: 0 });
    if (i < 2) arrow(s, 11.25, y + 1.42, 11.25, y + 1.68, MUTED); });
  s.addText([{ text: "검증 결과  ", options: { bold: true, color: RED } }, { text: "멀티모달을 더해도 가격 정보만 쓴 3대 전략 혼합(Sharpe 0.88)을 넘지 못함 → 화면에 '보조 신호'로 표시 (갈래 중 시장 포지션 COT가 가장 유용)" }],
    { x: 0.5, y: 6.98, w: 12.3, h: 0.36, fontFace: F, fontSize: 10.5, color: INK, margin: 0, valign: "middle" });
}

// ════════════════ 8 포트폴리오 3대 전략
{
  const s = base();
  header(s, "Ⅲ", "포트폴리오 3대 전략 — 모멘텀 · 최소분산(평균-분산) · 가치", "성격이 다른 세 가지 규칙 — 잘 되는 시기가 서로 달라 섞을수록 안정적");
  const st = [
    ["모멘텀", "추세는 이어진다", "지난 1년(52주) 수익률이\n+인 품목은 사고, −인 품목은 팖", "'달리는 말에 올라탄다'\n공급 부족·과잉은 몇 달~몇 년 지속", SH["52주 모멘텀"], BLUE],
    ["최소분산 (평균-분산)", "섞으면 출렁임이 준다", "19개 품목을 모두 사되, 함께 덜\n움직이는 조합으로 비중을 정해\n전체 출렁임을 최소화", "'달걀을 여러 바구니에'\n노벨경제학상 포트폴리오 이론(1990)", SH["최소분산"], G],
    ["가치", "너무 싸면 돌아온다", "5년 평균 가격 대비 가장 싼 1/3은\n사고, 가장 비싼 1/3은 팖", "'고무줄은 당긴 만큼 돌아온다'\n생산·소비 조정으로 가격 회귀", SH["가치"], GOLD],
  ];
  const VIS = ["c55_mom3d", "c56_minvar3d", "c57_value3d"];
  st.forEach((c, i) => {
    const x = 0.6 + i * 4.15, w = 3.95, [q, why] = c[3].split("\n");
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.7, w, h: 4.35, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.7, w, h: 0.12, fill: { color: c[5] }, line: { color: c[5] } });
    s.addText(c[0], { x: x + 0.2, y: 1.92, w: w - 0.4, h: 0.42, fontFace: F, fontSize: 18, bold: true, color: c[5], margin: 0 });
    s.addText(c[1], { x: x + 0.2, y: 2.34, w: w - 0.4, h: 0.36, fontFace: F, fontSize: 13.5, bold: true, color: INK, margin: 0 });
    s.addText([{ text: "규칙", options: { bold: true, color: GOLD, breakLine: true } }, { text: c[2] }],
      { x: x + 0.2, y: 2.76, w: w - 0.35, h: 0.86, fontFace: F, fontSize: 10.5, color: INK, margin: 0, valign: "top" });
    s.addText([{ text: "직관  ", options: { bold: true, color: GOLD } }, { text: q, options: { bold: true, breakLine: true } }, { text: why, options: { color: MUTED, fontSize: 9.5 } }],
      { x: x + 0.2, y: 3.64, w: w - 0.35, h: 0.44, fontFace: F, fontSize: 10.5, color: INK, margin: 0, valign: "top" });
    img(s, VIS[i], x + 0.12, 4.12, w - 0.24, 1.5);
    s.addText([{ text: `Sharpe ${c[4]}`, options: { bold: true, fontSize: 16, color: c[5] } }, { text: "  (단독, 2007–2026)", options: { fontSize: 10, color: MUTED } }], { x: x + 0.2, y: 5.62, w: w - 0.4, h: 0.4, fontFace: F, margin: 0 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 6.15, w: 12.2, h: 0.72, fill: { color: "FBF4E1" }, line: { color: GOLD, width: 1 }, rectRadius: 0.08 });
  s.addText([{ text: `세 전략을 섞으면 Sharpe ${SH["가치 혼합"]}   `, options: { bold: true, color: RED, fontSize: 15 } }, { text: "최소분산 50% + 모멘텀 25% + 가치 25% — 각 전략 단독(0.4~0.6)보다 위험 대비 성과가 크게 개선 (한 전략이 부진할 때 다른 전략이 보완)", options: { fontSize: 11.5, color: INK } }],
    { x: 0.85, y: 6.15, w: 11.8, h: 0.72, fontFace: F, valign: "middle", margin: 0 });
  foot(s, "Sharpe = 연 수익 ÷ 연 변동성(위험 1단위당 수익, 1에 가까우면 우수). 각 목표 변동성 10%, 편도 거래비용 5bp 차감, 연속월물 기준.");
}

// ════════════════ 혼합 전략은 매주 이렇게 계산 (3대 전략 다음 장)
{
  const s = base();
  header(s, "Ⅲ", "혼합 전략은 매주 이렇게 계산 — 최소분산 50 · 모멘텀 25 · 가치 25", "세 전략이 각자 정한 품목별 비중을 50:25:25로 합쳐 '순 비중' 하나로 주문 — 매주 반복");
  const steps = [["① 전략별 비중", "세 전략이 따로 품목별 비중을 정함\n각 전략 단독으로 연 10% 위험이 되게 크기 맞춤", BLUE],
    ["② 50:25:25 가중합", "품목마다 세 비중을 섞는 비율로 더함\n반대 방향 포지션은 서로 상쇄", G],
    ["③ 전체 위험 다시 맞춤", "섞은 전체를 연 10% 위험으로 조정\n(직전 26주 기준, 배율 상한 4배)", GOLD],
    ["④ 다음 주 반복", "매주 다시 50:25:25로 되돌림\n(잘된 전략이 커지지 않게)", RED]];
  steps.forEach(([t, b, col], i) => { const x = 0.6 + i * 3.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.65, w: 2.8, h: 1.5, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.65, w: 2.8, h: 0.08, fill: { color: col }, line: { color: col } });
    s.addText(t, { x: x + 0.18, y: 1.8, w: 2.5, h: 0.4, fontFace: F, fontSize: 14, bold: true, color: col, margin: 0, valign: "middle" });
    s.addText(b, { x: x + 0.18, y: 2.22, w: 2.5, h: 0.85, fontFace: F, fontSize: 10.5, color: INK, margin: 0, valign: "top" });
    if (i < 3) arrow(s, x + 2.83, 2.4, x + 3.07, 2.4, MUTED); });

  // 옥수수 한 품목 예시: 전략별 비중 × 섞는 비율 = 기여 → 순 비중
  s.addText("예시 — 이번 주 옥수수 한 품목", { x: 0.6, y: 3.4, w: 7.4, h: 0.38, fontFace: F, fontSize: 13.5, bold: true, color: G, margin: 0 });
  const hdr = [["전략", 0.6, 1.7], ["전략이 정한 옥수수 비중", 2.3, 3.2], ["섞는 비율", 5.55, 0.9], ["기여", 6.5, 1.5]];
  hdr.forEach(([t, x, w]) => s.addText(t, { x, y: 3.8, w, h: 0.3, fontFace: F, fontSize: 9.5, color: MUTED, margin: 0, align: x > 5 ? "center" : "left" }));
  const Z = 3.65, K = 0.032;                                   // 0% 기준선 위치, 1%p당 길이
  const rows = [["최소분산", "함께 덜 움직여 롱", 20, 0.5, BLUE], ["모멘텀", "1년간 내려 숏", -30, 0.25, G], ["가치", "5년 평균보다 싸서 롱", 40, 0.25, GOLD]];
  rows.forEach(([nm, why, v, w, col], i) => { const y = 4.15 + i * 0.62;
    s.addText([{ text: nm, options: { bold: true, fontSize: 12, color: INK, breakLine: true } }, { text: why, options: { fontSize: 8.5, color: MUTED } }], { x: 0.6, y, w: 1.7, h: 0.52, fontFace: F, margin: 0, valign: "middle" });
    const bx = v >= 0 ? Z : Z + v * K;
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: y + 0.11, w: Math.abs(v) * K, h: 0.3, fill: { color: col }, line: { color: col } });
    s.addText((v > 0 ? "+" : "−") + Math.abs(v) + "%", { x: v >= 0 ? Z + v * K + 0.05 : Z + v * K - 0.65, y, w: 0.6, h: 0.52, fontFace: F, fontSize: 11, bold: true, color: col, margin: 0, valign: "middle", align: v >= 0 ? "left" : "right" });
    s.addText("× " + w.toFixed(2), { x: 5.55, y, w: 0.9, h: 0.52, fontFace: F, fontSize: 12, color: INK, margin: 0, valign: "middle", align: "center" });
    const c = v * w;
    s.addText("= " + (c > 0 ? "+" : "−") + Math.abs(c).toFixed(1) + "%p", { x: 6.5, y, w: 1.5, h: 0.52, fontFace: F, fontSize: 12.5, bold: true, color: c >= 0 ? G : RED, margin: 0, valign: "middle", align: "center" }); });
  s.addShape(pres.shapes.LINE, { x: Z, y: 4.15, w: 0, h: 1.86, line: { color: "999999", width: 1, dashType: "dash" } });
  s.addShape(pres.shapes.LINE, { x: 0.6, y: 6.08, w: 7.4, h: 0, line: { color: G, width: 1.25 } });
  s.addText([{ text: "순 비중 (실제 주문)", options: { bold: true, fontSize: 12, color: G, breakLine: true } }, { text: "세 기여를 더함", options: { fontSize: 8.5, color: MUTED } }], { x: 0.6, y: 6.13, w: 1.7, h: 0.55, fontFace: F, margin: 0, valign: "middle" });
  s.addShape(pres.shapes.RECTANGLE, { x: Z, y: 6.25, w: 12.5 * K, h: 0.3, fill: { color: RED }, line: { color: RED } });
  s.addText("+12.5% 롱", { x: Z + 12.5 * K + 0.08, y: 6.13, w: 1.4, h: 0.55, fontFace: F, fontSize: 13, bold: true, color: RED, margin: 0, valign: "middle" });
  s.addText("10.0 − 7.5 + 10.0", { x: 5.55, y: 6.13, w: 2.45, h: 0.55, fontFace: F, fontSize: 11, color: INK, margin: 0, valign: "middle", align: "center" });

  // 왜 효과가 있나: 세 전략의 상관
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.45, y: 3.4, w: 4.4, h: 3.35, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
  s.addText("왜 섞으면 좋아지나", { x: 8.65, y: 3.5, w: 4.0, h: 0.4, fontFace: F, fontSize: 13.5, bold: true, color: G, margin: 0, valign: "middle" });
  table(s, [["함께 움직이는 정도", "최소분산", "모멘텀"], ["모멘텀", "0.20", ""], ["가치", "−0.06", "−0.44"]],
    { x: 8.65, y: 3.95, w: 4.0, colW: [1.6, 1.2, 1.2], rowH: 0.3, fontSize: 10.5, align: "center", boldFirst: true });
  s.addText([{ text: "• 서로 거의 따로 움직임 (1 = 똑같이, 0 = 무관, − = 반대)", options: { breakLine: true } },
    { text: "• 추세가 꺾일 때 모멘텀은 잃고 가치는 버는 경향 → 손실이 서로 메워짐", options: { breakLine: true } },
    { text: "• 그래서 같은 10% 위험에서 수익이 커짐: 단독 0.4–0.6 → 혼합 0.88", options: { bold: true, color: G } }],
    { x: 8.65, y: 4.95, w: 4.05, h: 1.75, fontFace: F, fontSize: 10.5, color: INK, margin: 0, valign: "top", paraSpaceAfter: 4 });
  foot(s, "이번 주 수익 = (0.5 × 최소분산 + 0.25 × 모멘텀 + 0.25 × 가치) × 배율. 옥수수 비중은 설명용 예시 수치. 상관 = 각 전략 주간 수익률, 2007–2026.");
}

// ════════════════ 9 섞은 결과 · 목표 위험
{
  const s = base();
  header(s, "Ⅲ", "섞은 결과 — 약 20년간 꾸준, 목표 위험을 높이면 수익도 비례", "위험 1단위당 수익은 유지된 채 규모만 커짐 → 감내 가능한 최대 낙폭에서 목표를 정하는 것이 핵심");
  img(s, "c16_strategies", 0.4, 1.6, 6.4, 3.6);
  s.addChart(pres.charts.LINE, [{ name: "연수익 (%)", labels: VT.tgt.map((t) => t + "%"), values: VT.arith }, { name: "최대 낙폭 (%)", labels: VT.tgt.map((t) => t + "%"), values: VT.mdd }],
    C({ x: 6.9, y: 1.6, w: 5.9, h: 3.6, chartColors: [G, RED], lineSize: 2.5, lineDataSymbolSize: 6, showLegend: true, legendPos: "b", showTitle: true, title: "목표 위험(연 변동성)별 결과 — 3대 전략 혼합", catAxisTitle: "목표 변동성", showCatAxisTitle: false }));
  const i10 = VT.tgt.indexOf(10), i20 = VT.tgt.indexOf(20), i30 = VT.tgt.indexOf(30);
  stat(s, 0.7, 5.45, 2.9, `${VT.arith[i10]}%`, `목표 10% 연수익 (최대 낙폭 ${VT.mdd[i10]}%)`, G);
  stat(s, 3.7, 5.45, 2.9, `${VT.arith[i20]}%`, `목표 20% 연수익 (최대 낙폭 ${VT.mdd[i20]}%)`, GOLD);
  stat(s, 6.7, 5.45, 2.9, `${VT.arith[i30]}%`, `목표 30% 연수익 (최대 낙폭 ${VT.mdd[i30]}%)`, RED);
  bullets(s, ["목표를 여유 있게 잡을수록 수익은 거의 비례해 커지고, 낙폭도 함께 커짐", "시작은 보수적(10%)으로, 실적 확인 후 단계적 상향"], { x: 9.7, y: 5.45, w: 3.1, h: 1.45, fontSize: 11 });
  foot(s, "2007–2026, 편도 거래비용 5bp 차감, 배율 상한 4배, 차입 이자 미반영, 연속월물 기준(실제 ETF 기준보다 연 약 3%p 높게 나올 수 있음). 출처: ⑤ 목표변동성 백테스트.");
}

// ════════════════ 효율적 투자선 — 연도별 작은 그림(겹침 없음) + 3D 리본(3년 간격)
{
  const s = base();
  header(s, "Ⅲ", "효율적 투자선 — 우리 혼합은 대부분 '최선의 경계' 가까이", "효율적 투자선 = 같은 위험에서 가장 높은 수익을 내는 조합들의 경계선");
  img(s, "c49a_frontier_grid", 0.35, 1.5, 7.6, 5.45);
  img(s, "c49b_frontier_ribbon", 8.1, 1.45, 4.8, 2.75);
  s.addText("3D: 3년 간격 경계선을 연도 축으로 세움", { x: 8.15, y: 4.12, w: 4.7, h: 0.25, fontFace: F, fontSize: 9, color: MUTED, align: "center", margin: 0 });
  bullets(s, [{ t: "읽는 법 · 시사점", b: true, c: G },
    "작은 그림 하나 = 그해 기준 직전 5년 자료로 그린 경계선(초록), ★ = 우리 혼합(고정), ● = 그해 사후 최적",
    "2011–14 · 2024–26: ★가 경계선 위 (차이 0–0.4%p)",
    "2015–20: 한 전략만 독주(모멘텀 → 가치)한 시기 → 사후 최적은 '그 전략에 몰빵', ★는 같은 위험에서 연 3–4%p 낮음",
    "● 위치는 해마다 크게 이동 → 미리 알 수 없음. 고정 혼합은 최선은 아니어도 꾸준히 경계선 근처"], { x: 8.1, y: 4.45, w: 4.8, h: 2.5, fontSize: 10.5 });
  foot(s, "3대 전략 각 목표 변동성 10%, 주간 수익률, 연도별 직전 5년 평균·공분산으로 조합 6,000개 계산. 차이 = 우리 혼합과 같은 위험 이하 조합 중 최고 수익 − 혼합 수익.");
}

// ════════════════ 10 구매 의사결정
{
  const s = base();
  header(s, "Ⅳ", "지금 바로 쓰는 기능 — '지금 살까, 미룰까' 주간 판단", "두 가지를 매주 계산: 지금 가격이 매력적인가(점수) · 기다리면 더 싸질 가능성이 높은가(확률)");
  card(s, 0.6, 1.7, 4.0, 2.55, "① 가격 매력 점수 (0~100)", "• 최근 범위 중 지금 얼마나 낮은가\n• 바닥을 찍고 반등을 시작했나\n• 대체재·원료 대비 싼가 (옥수수/밀 등)\n• 해마다 이 시기에 쌌나 (계절)", G, { bs: 12 });
  card(s, 0.6, 4.4, 4.0, 2.4, "② 기다리면 더 쌀 확률", "• 1~4주 안에 지금보다 쌀 확률\n• 과거에 '동전보다 확실히 잘 맞힌'\n  9개 품목에만 사용 (곡물 중 옥수수)\n• 나머지 품목은 점수만 참고", BLUE, { bs: 12 });
  const cs = [["옥수수", "적극 매수", "2E7D3E", "60점", "저점권 + 대체재 대비 쌈"], ["밀", "판단 유보", "B8901C", "41점", "최근 상승, 급하지 않으면 대기"], ["대두", "판단 유보", "B8901C", "43점", "반등 약함, 다음 주 재확인"], ["대두박", "판단 유보", "B8901C", "35점", "1년 추세 상승 중"]];
  s.addText("이번 주 판단 (기준일 2026-10-26)", { x: 4.9, y: 1.7, w: 4.5, h: 0.4, fontFace: F, fontSize: 14, bold: true, color: G, margin: 0 });
  cs.forEach((c, i) => { const y = 2.2 + i * 0.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.9, y, w: 4.5, h: 0.82, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addText(c[0], { x: 5.05, y, w: 1.1, h: 0.82, fontFace: F, fontSize: 14, bold: true, color: INK, valign: "middle", margin: 0 });
    dot(s, 6.15, y + 0.3, c[2], 0.22);
    s.addText([{ text: c[1] + "  ", options: { bold: true, color: c[2], fontSize: 12.5 } }, { text: c[3], options: { bold: true, color: G, fontSize: 12.5, breakLine: true } }, { text: c[4], options: { fontSize: 10.5, color: MUTED } }],
      { x: 6.45, y, w: 2.9, h: 0.82, fontFace: F, valign: "middle", margin: 0 }); });
  s.addText("근거가 확인된 품목의 검증 결과 (2017년 이후)", { x: 9.7, y: 1.7, w: 3.2, h: 0.4, fontFace: F, fontSize: 12.5, bold: true, color: G, margin: 0 });
  s.addChart(pres.charts.BAR, [{ name: "판단당 평균 절감 (bp)", labels: ["판단대로 미룸", "아무 때나 미룸"], values: [23, -14] }],
    C({ x: 9.6, y: 2.05, w: 3.3, h: 2.9, barDir: "col", chartColors: [G, RED], showValue: true, dataLabelPosition: "outEnd", showLegend: false, valAxisHidden: true, valGridLine: { style: "none" } }));
  bullets(s, ["판단 하나하나는 틀릴 수 있음 → 매주 반복할수록 평균 우위가 누적", "옥수수 카고(3만 톤) 매주 판단 시 연 약 $35만 절감 기대(예시)"], { x: 9.6, y: 5.0, w: 3.2, h: 1.9, fontSize: 11 });
  foot(s, "검증 = 2008–2016으로 정한 규칙을 2017년 이후에 그대로 적용. 원자재 25개 품목에 같은 규칙 → 근거 확인 9개. 예시 절감 = 옥수수 판단당 12.6bp × $5.34M × 52회.");
}

// ════════════════ 11 리스크 관리
{
  const s = base();
  header(s, "Ⅳ", "리스크 관리 — 물량의 가격 위험을 금액으로", "'100번 중 99번은 이보다 덜 잃는다'는 손실 기준선(VaR)을 실제 물량 단위로 계산");
  const rows = [["매입 예정 물량 (각 3만 톤)", "20일 손실 기준선 (99%)"], ["옥수수 + 대두 + 대두박", "약 $3.2M"], ["과거 실제 충격을 재현한 방식", "약 $3.4M"], ["품목별로 따로 계산해 더하면", "약 $3.9M (섞여서 $0.6M 상쇄)"]];
  table(s, rows, { x: 0.6, y: 1.75, w: 6.0, colW: [3.6, 2.4], rowH: 0.5, fontSize: 12.5 });
  bullets(s, [{ t: "실무 활용", b: true, c: G }, "기준선이 감내 범위를 넘으면 → 선매입·헤지 비율 검토", "어느 품목이 위험을 가장 많이 만드는지 기여도로 표시", { t: "주의", b: true, c: RED }, "'평범한 종 모양' 가정은 위기를 과소평가 → 과거 실제 충격을 다시 쓰는 방식을 함께 제시", "평시가 아닌 위기 때 기준으로 한도 설정 (위기엔 품목들이 함께 움직임)"], { x: 0.6, y: 3.95, w: 6.0, h: 2.95, fontSize: 12.5 });
  img(s, "c26_corr", 6.9, 1.6, 5.9, 5.2);
  foot(s, "기준일 2026-10-26 가격, 매입 예정(가격 상승이 손실). 오른쪽: 원자재 19개 주간 수익률 상관(2007–2026) — 같은 색 묶음은 함께 움직임. 출처: ◆ 리스크 › 구매·보유 물량 위험.");
}

// ════════════════ 11-2 대시보드 시연 영상
{
  const s = base();
  header(s, "Ⅳ", "대시보드 시연 — 실제 화면으로 보는 4가지 기능", "구매 판단 · 머신러닝 예측 · 포트폴리오 구성 · 레짐(칼만 필터)을 실제 앱에서 조작한 화면 (약 75초)");
  const cover = "data:image/png;base64," + fs.readFileSync(path.join(__dirname, "img", "demo_v4_cover.png")).toString("base64");
  s.addMedia({ type: "video", path: path.join(__dirname, "img", "demo_v4.mp4"), cover, x: 0.5, y: 1.6, w: 8.9, h: 5.0 });
  const sc = [["① 구매 의사결정", "0:00", "품목별 이번 주 매수 · 보류 판단과 물량 위험 금액", G],
    ["② 머신러닝 예측", "0:15", "OLS · Ridge · Lasso · 나무 모델 등 7종을 같은 조건으로 비교", BLUE],
    ["③ Portfolio Customization", "0:51", "전략을 골라 섞고 효율적 투자선 3D에서 내 위치 확인", GOLD],
    ["④ 레짐 · 칼만 필터", "1:02", "위기 때 커지는 민감도(β)를 시간 × 변수 지형으로 추적", RED]];
  sc.forEach(([t, tm, d, col], i) => { const y = 1.6 + i * 1.27;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.6, y, w: 3.25, h: 1.15, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addShape(pres.shapes.RECTANGLE, { x: 9.6, y: y + 0.12, w: 0.07, h: 0.9, fill: { color: col }, line: { color: col } });
    s.addText([{ text: t, options: { bold: true, color: INK, fontSize: 12 } }, { text: "   ▶ " + tm, options: { color: MUTED, fontSize: 9.5, breakLine: true } },
      { text: d, options: { color: INK, fontSize: 9.5 } }], { x: 9.8, y, w: 2.95, h: 1.15, fontFace: F, valign: "middle", margin: 0, paraSpaceAfter: 2 }); });
  foot(s, "영상 = 대시보드 실제 화면 녹화(계산 대기 구간 생략). 화면 숫자는 자료 기준일 2026-10-26 기준.");
}

// ════════════════ 12 확장 비전 — 품목군 기준
{
  const s = base();
  header(s, "Ⅴ", "확장 비전 — 전사 원자재 리스크 컨트롤 타워", "금융 트레이딩을 하지 않더라도, 품목군별로 흩어진 원자재 매매를 통합해 회사 전체의 가격 변동 위험을 줄일 수 있음");
  const OK = "2E7D3E", MID = "B8901C", NO = RED, MK = { o: ["● ", OK], m: ["◐ ", MID], x: ["✕ ", NO] };
  const grp = [["식량", "6E7F74", [["옥수수", "o"], ["밀", "o"], ["대두", "o"], ["대두박", "o"], ["대두유", "o"], ["팜유", "m"], ["수수", "x"]], 3],
    ["철강·소재", BLUE, [["철광석", "o"], ["구리(동)", "o"], ["알루미늄", "o"], ["아연", "m"], ["리튬", "m"], ["코발트", "x"], ["텅스텐", "x"]], 2],
    ["에너지", GOLD, [["원유", "o"], ["석탄", "m"], ["LNG", "m"], ["PNG", "x"], ["암모니아", "x"], ["수소", "x"]], 3]];
  grp.forEach(([nm, col, items, br], i) => { const y = 1.68 + i * 1.42;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y, w: 3.0, h: 1.3, fill: { color: TINT }, line: { color: LINE }, rectRadius: 0.08 });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: y + 0.15, w: 0.08, h: 1.0, fill: { color: col }, line: { color: col } });
    const runs = [{ text: nm, options: { bold: true, fontSize: 14, color: INK, breakLine: true } }];
    items.forEach(([t, k], j) => { runs.push({ text: MK[k][0], options: { fontSize: 9.5, color: MK[k][1], bold: true } });
      runs.push({ text: t + (j === br ? "" : "   "), options: { fontSize: 10, color: k === "x" ? "9A9A94" : INK, breakLine: j === br } }); });
    s.addText(runs, { x: 0.7, y, w: 2.78, h: 1.3, fontFace: F, valign: "middle", margin: 0, paraSpaceAfter: 2 });
    arrow(s, 3.55, y + 0.65, 3.95, 3.85, MUTED); });
  s.addText([{ text: "● ", options: { color: OK, bold: true } }, { text: "지금 분석 가능   ", options: {} }, { text: "◐ ", options: { color: MID, bold: true } }, { text: "시장은 있음 · 자료 연결 필요   ", options: {} },
    { text: "✕ ", options: { color: NO, bold: true } }, { text: "시장 시세 없음 → 대체 지표로 위험만", options: {} }], { x: 0.5, y: 5.92, w: 9.2, h: 0.3, fontFace: F, fontSize: 9.5, color: INK, margin: 0 });
  s.addShape(pres.shapes.OVAL, { x: 4.0, y: 2.6, w: 2.5, h: 2.5, fill: { color: G }, line: { color: G } });
  s.addText("리스크\n컨트롤 타워", { x: 4.0, y: 2.6, w: 2.5, h: 2.5, fontFace: F, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  const outs = ["전사 포지션 합산 · 순 위험 금액", "품목끼리 상쇄되는 자연 헤지 식별", "목표 위험 기준 한도 배분", "구매 시점·물량의 통합 의사결정"];
  outs.forEach((o, i) => { const y = 1.95 + i * 0.95; arrow(s, 6.55, 3.85, 7.0, y + 0.32, GOLD);
    s.addText(o, { x: 7.05, y, w: 2.5, h: 0.65, fontFace: F, fontSize: 11.5, bold: true, color: G, valign: "middle", margin: 0 }); });
  card(s, 9.7, 1.75, 3.1, 2.45, "학계·헤지펀드의 표준 방법", "• 포트폴리오 이론 — 노벨경제학상(1990)\n• 위험 크기 예측 — 노벨경제학상(2003)\n• 추세·가치 전략 — 대형 헤지펀드의 핵심 운용 방식", BLUE, { hs: 13, bs: 11 });
  card(s, 9.7, 4.35, 3.1, 2.4, "모든 가격 자산으로 확장", "원자재뿐 아니라 주식·채권·환율 등 가격이 있는 모든 자산에 같은 엔진(위험 계산·전략·검증)을 그대로 적용 가능", GOLD, { hs: 13, bs: 11.5 });
  bullets(s, ["품목군별로는 작아 보이는 위험도, 합치면 함께 움직여 커지거나 서로 상쇄됨 — 통합해서 봐야 실제 위험이 보임"], { x: 0.5, y: 6.3, w: 9.0, h: 0.6, fontSize: 12 });
  foot(s, "판정 기준 = 유동성 있는 선물 시세가 장기간(5년+) 있는가. ✕ 품목은 모멘텀·가치 등 전략 적용 불가, 계약가·대체 지표(천연가스·유가·옥수수 등)로 위험만 근사. 품목별 근거는 별첨 2.");
}

// ════════════════ 13 활용 제안 · 한계
{
  const s = base();
  header(s, "Ⅴ", "활용 제안 — 작게 시작해 실적으로 확장", "검증된 기능부터 3단계로 적용, 한계를 알고 판단을 보조하는 도구로 사용");
  const st = [["1단계 · 즉시", "주간 구매 판단 공유", "• 식량사업실 주간 회의에 판단 카드 공유\n• 옥수수 우선, 밀·대두는 점수만 참고\n• 실제 구매 결과 기록 → 사후 검증", BLUE], ["2단계 · 6개월", "리스크 관리 연계", "• 물량 단위 위험 금액으로 선매입·헤지 참고\n• 위기 기준 위험 한도 설정", G], ["3단계 · 1년+", "포트폴리오 사업 검토", "• 소규모 모의 포트폴리오(목표 위험 10%)\n• 실제 시간으로 실적·낙폭 검증 후\n  사업화 여부 판단", GOLD]];
  st.forEach((c, i) => { const x = 0.6 + i * 4.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: 3.95, h: 2.95, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addText(c[0], { x: x + 0.2, y: 1.9, w: 3.5, h: 0.35, fontFace: F, fontSize: 12, bold: true, color: c[3], margin: 0 });
    s.addText(c[1], { x: x + 0.2, y: 2.25, w: 3.5, h: 0.5, fontFace: F, fontSize: 18, bold: true, color: INK, margin: 0 });
    s.addText(c[2], { x: x + 0.2, y: 2.85, w: 3.6, h: 1.8, fontFace: F, fontSize: 12, color: INK, margin: 0, valign: "top", paraSpaceAfter: 4 });
    if (i < 2) arrow(s, x + 3.97, 3.2, x + 4.13, 3.2, MUTED); });
  s.addText([{ text: "한계  ", options: { bold: true, color: RED } }, { text: "과거 성과가 미래를 보장하지 않음 · 개별 판단은 틀릴 수 있는 확률적 우위 · 물량·계약 조건 등 현장 판단과 병행" }],
    { x: 0.6, y: 4.85, w: 12.2, h: 0.5, fontFace: F, fontSize: 12.5, color: INK, margin: 0, valign: "middle" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.5, w: 12.2, h: 1.3, fill: { color: "FBF4E1" }, line: { color: GOLD, width: 1 }, rectRadius: 0.08 });
  s.addText([{ text: "의사결정 요청   ", options: { bold: true, color: RED } }, { text: "① 1단계(식량사업실 주간 판단 공유) 시범 운영 승인  ② 실제 구매 기록(일자·물량·단가) 공유 협조 — 사후 검증용  ③ 3단계 검토를 위한 모의 운용 기준(위험 한도) 협의" }],
    { x: 0.85, y: 5.5, w: 11.8, h: 1.3, fontFace: F, fontSize: 13, color: INK, valign: "middle", margin: 0 });
}

// ════════════════ 별첨: 쉬운 이름 ↔ 기법 이름 (질의응답용)
{
  const s = base();
  header(s, "별첨", "질의응답용 — 쉬운 표현과 실제 기법 이름", "본문은 비전공 청중 기준으로 풀어 씀. 대시보드·방법론 탭의 실제 명칭은 아래와 같음");
  const rows = [["본문 표현", "실제 기법 · 지표", "대시보드 위치"],
    ["위험 크기 예측", "GARCH(1,1)·GJR·MIDAS-RV, 다변량 DCC·BEKK", "③ 변동성 · ◆ 리스크"],
    ["손실 기준선 / 과거 실제 충격 재현", "VaR·ES, 필터링 역사적 시뮬레이션(FHS)·EVT", "◆ 리스크"],
    ["시장 국면 판별 / 변하는 관계 추적", "Hidden Markov Model, Markov-switching, Kalman 필터", "② 레짐"],
    ["동전보다 잘 맞힌 품목", "AUC > 0.55 (학습 2008–16), 시험 2017~ 평균 0.570", "★ 구매 · ⑥ 실험"],
    ["기다리면 더 쌀 확률", "로지스틱 회귀, 매년 1월 재학습(walk-forward)", "★ 구매"],
    ["무작위 걸음 / 시뮬레이션", "랜덤워크·기하 브라운 운동, 몬테카를로 400회", "본문 6장"],
    ["위험 1단위당 수익", "Sharpe = 연 수익 ÷ 연 변동성", "⑤ 포트폴리오"],
    ["과거로 배우고 이후로 시험", "학습/시험 분리, look-ahead bias 차단, 위약 비교", "⑧ 방법론"]];
  table(s, rows, { x: 0.6, y: 1.75, w: 12.2, colW: [3.6, 5.8, 2.8], rowH: 0.5, fontSize: 11.5, boldFirst: true });
}

// ════════════════ 별첨 2: 품목별 퀀트 분석 가능 여부 (확장 비전 근거)
{
  const s = base();
  header(s, "별첨", "별첨 2 — 품목별 퀀트 분석 가능 여부", "기준: 유동성 있는 선물·시장 시세가 5년 이상 매일 있는가 (모멘텀 1년 · 가치 5년 평균 · 위험 추정에 필요)");
  const OK = "● 가능", MID = "◐ 연결 필요", NO = "✕ 어려움";
  const rows = [["품목군", "품목", "시장 (거래소)", "판정", "대시보드 적용 방법"],
    ["식량", "옥수수 · 밀 · 대두 · 대두박 · 대두유", "CBOT 선물", OK, "이미 수집 중 — 전체 분석 적용"],
    ["", "팜유", "Bursa Malaysia (FCPO) · DCE", MID, "선물 활발, 자료 소스 연결만 필요"],
    ["", "수수", "유동성 있는 선물 없음", NO, "옥수수 가격으로 대체 (사료용 대체재, 함께 움직임)"],
    ["철강·소재", "철광석", "SGX 62% · DCE", OK, "이미 수집 중"],
    ["", "구리 · 알루미늄", "COMEX · LME", OK, "이미 수집 중 (알루미늄은 LME가 주 시장)"],
    ["", "아연", "LME · SHFE", MID, "선물 활발, 무료 일별 자료 제한 → 소스 연결 필요"],
    ["", "리튬", "GFEX 탄산리튬(2023~) · CME 수산화리튬", MID, "역사 짧음 → 위험 계산만, 5년 가치 전략은 아직 불가"],
    ["", "코발트", "CME 선물 상장, 거래 거의 없음", NO, "가격 평가지(유료) 기준 → 위험만 근사"],
    ["", "텅스텐", "선물 없음 (APT 가격 평가만)", NO, "계약가 기록 기반 → 퀀트 전략 불가"],
    ["에너지", "원유", "WTI · Brent", OK, "이미 수집 중"],
    ["", "석탄", "ICE Newcastle · Rotterdam", MID, "선물 활발, 무료 자료 제한 (IMF 월간 호주탄은 무료)"],
    ["", "LNG", "JKM 선물 (CME · ICE)", MID, "자료 연결 필요. 미국 천연가스(HH)는 수집 중이나 아시아 LNG와 다름"],
    ["", "PNG (배관 가스)", "자체 시세 없음 — 장기 계약", NO, "계약 공식(유가·허브 가스 연동)대로 유가·가스로 위험 환산"],
    ["", "암모니아", "CME 선물 상장, 거래 거의 없음", NO, "원가 대부분인 천연가스로 위험 근사"],
    ["", "수소", "거래 시장 없음", NO, "정책·계약가 → 현재는 정량 분석 대상 아님"]];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: {
      bold: i === 0 || j === 0 || j === 3, fontSize: 9.5,
      color: i === 0 ? WHITE : j === 3 ? (c.startsWith("●") ? "2E7D3E" : c.startsWith("◐") ? "B8901C" : RED) : j === 4 && r[3].startsWith("✕") ? MUTED : INK,
      fill: { color: i === 0 ? G : (i % 2 ? WHITE : TINT) } } }))),
    { x: 0.5, y: 1.6, w: 12.3, colW: [1.1, 2.9, 2.9, 1.2, 4.2], rowH: 0.3, fontFace: F, border: { type: "solid", color: LINE, pt: 0.5 }, valign: "middle" });
  foot(s, "판정은 2026년 10월 기준 거래소 상장·거래 현황에 대한 일반 지식 — 자료 연결 전 거래량·자료 구입 비용 재확인 필요. 대시보드 수집 여부는 코드 기준 확인.");
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("written", f, "slides", page));
if (NOTES.length !== page) console.warn(`발표자 노트 ${NOTES.length}개 ≠ 장 수 ${page}`);
