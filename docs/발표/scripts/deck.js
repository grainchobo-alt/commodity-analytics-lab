// 발표 자료: 퀀트 기법 기반 곡물·원자재 Market Intelligence
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const IMG = (n) => path.join(__dirname, "img", n + ".png");
const OUT = process.argv[2] || path.join(__dirname, "deck.pptx");
const NOTES = require("./notes.js");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "퀀트 기법 기반 곡물·원자재 Market Intelligence";
const F = "Malgun Gothic";
const G = "1E3A2F", GOLD = "C8A24A", RED = "D9582B", BLUE = "2F6E8F", INK = "222222", MUTED = "6B6B66", TINT = "F1F3EF", LINE = "D6D9D2", WHITE = "FFFFFF";
const W = 13.333;
let page = 0;

// ── 공통 요소
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
    return { text: it.t, options: { bullet: it.b ? false : (it.sub ? { indent: 18 } : true), indentLevel: it.sub ? 1 : 0, bold: !!it.b, color: it.c || INK, breakLine: !last, paraSpaceAfter: 6, fontSize: it.sub ? (o.fontSize || 14) - 1.5 : undefined } };
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
function concept(s, rows, x = 0.6, y = 1.75, w = 4.9) {
  // 한 줄 정의 / 직관 / 원자재 예시 / 대시보드 위치
  let yy = y;
  rows.forEach(([lab, txt, h]) => {
    s.addText(lab, { x, y: yy, w, h: 0.3, fontFace: F, fontSize: 11, bold: true, color: GOLD, margin: 0 });
    s.addText(txt, { x, y: yy + 0.3, w, h: h, fontFace: F, fontSize: 13, color: INK, margin: 0, valign: "top" });
    yy += 0.3 + h + 0.18;
  });
}
function divider(num, title, sub) {
  const s = base(true);
  s.addText(num, { x: 0.8, y: 2.3, w: 2, h: 1.2, fontFace: F, fontSize: 60, bold: true, color: GOLD, margin: 0 });
  s.addText(title, { x: 0.8, y: 3.5, w: 11.5, h: 0.9, fontFace: F, fontSize: 34, bold: true, color: WHITE, margin: 0 });
  if (sub) s.addText(sub, { x: 0.8, y: 4.4, w: 11.5, h: 0.6, fontFace: F, fontSize: 16, color: "C9D3CC", margin: 0 });
  return s;
}
const chartBase = { catAxisLabelFontFace: F, valAxisLabelFontFace: F, legendFontFace: F, titleFontFace: F, dataLabelFontFace: F,
  catAxisLabelColor: "55554F", valAxisLabelColor: "55554F", valGridLine: { color: "E5E5E0", size: 0.5 }, catGridLine: { style: "none" },
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, legendFontSize: 11, titleFontSize: 13, titleColor: G, dataLabelFontSize: 10 };
const C = (o) => Object.assign({}, chartBase, o);

// ════════════════ Ⅰ. 배경
{ // 1 표지
  const s = base(true);
  s.addText("Market Intelligence", { x: 0.8, y: 1.5, w: 11, h: 0.5, fontFace: F, fontSize: 18, color: GOLD, bold: true, margin: 0 });
  s.addText("퀀트 기법 기반 곡물·원자재 시황 분석", { x: 0.8, y: 2.1, w: 11.8, h: 1.0, fontFace: F, fontSize: 40, bold: true, color: WHITE, margin: 0 });
  s.addText("리스크 정량화 · 변동성 구조화 · 구매 의사결정 지원 대시보드", { x: 0.8, y: 3.15, w: 11.8, h: 0.6, fontFace: F, fontSize: 20, color: "DCE3DE", margin: 0 });
  s.addText("식량사업개발실 · 팜사업운영그룹   |   2026. 10.", { x: 0.8, y: 6.3, w: 11, h: 0.4, fontFace: F, fontSize: 13, color: "AFC0B5", margin: 0 });
}
{ // 2 Executive Summary
  const s = base();
  header(s, "요약", "Executive Summary", "원자재 시황을 리스크·변동성·전략·구매 판단의 숫자로 바꾸고, 장기 검증으로 걸러낸 것만 의사결정에 연결함");
  const cards = [
    ["리스크·변동성 정량화", "GARCH", "1일 변동성 예측 12개 품목 중 12개에서 1위\n→ VaR·물량 위험·헤지 판단의 정량 기준으로 활용", G],
    ["위험 전이·국면 구조화", "금·원유", "BEKK로 위험 전이 중심 품목 식별(세 기간 공통 금, 2017년 이후 원유)\nHMM으로 2008·2020·2022 위기 국면 포착 → 위기 기준 리스크 관리", BLUE],
    ["구매 타이밍 근거 확보", "AUC 0.57", "근거 품목 9개 시험 기간(2017~) 평균, 판단당 +23bp\n(무작위 미루기 −14bp) → 옥수수 등 우선 적용", RED],
    ["장기 강건한 멀티 전략", "0.88", "가치 혼합 Sharpe — 약 20년 백테스트에서 1에 가까운 우수한 수준\n목표 변동성을 높이면 수익도 비례 상승(30%: 연 27.5%)", GOLD],
  ];
  cards.forEach((c, i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.75 + Math.floor(i / 2) * 2.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.95, h: 2.35, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addText(c[1], { x: x + 0.25, y: y + 0.25, w: 2.1, h: 0.9, fontFace: F, fontSize: 30, bold: true, color: c[3], margin: 0, valign: "middle" });
    s.addText(c[0], { x: x + 2.4, y: y + 0.25, w: 3.4, h: 0.9, fontFace: F, fontSize: 16, bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText(c[2], { x: x + 0.25, y: y + 1.25, w: 5.5, h: 1.0, fontFace: F, fontSize: 12.5, color: INK, margin: 0, valign: "top" });
  });
  foot(s, "모든 결과는 곡물 일부가 아닌 원자재 19–25개 품목 전체에 같은 규칙을 적용한 검증 기준. 결과 고정 기준 2026-10-26. bp = 0.01%.");
}
{ // 3 발표자·조직 맥락
  const s = base();
  header(s, "Ⅰ", "왜 이 분석을 하는가 — 현장 경험에서 출발", "트레이딩 현장의 고민 → 퀀트 학위 → 식량사업개발실 Market Intelligence 기여");
  const steps = [["식량사업실", "곡물 트레이딩 약 2년\n시황 움직임을 어떻게 분석할지 고민"], ["대학원", "퀀트 석사 취득\n시계열·리스크·포트폴리오 이론"], ["팜사업운영그룹", "팜 농장 운영 관리(주 업무)\n신규 사업개발 · 원자재 시황 분석"], ["대시보드 구축", "원자재 시황에 퀀트 기법 적용\n옆 사업실의 고민 해결 지원"]];
  steps.forEach((st, i) => {
    const x = 0.6 + i * 3.1;
    dot(s, x + 1.3, 2.05, i === 3 ? GOLD : G, 0.32);
    if (i < 3) s.addShape(pres.shapes.LINE, { x: x + 1.65, y: 2.21, w: 2.75, h: 0, line: { color: LINE, width: 2 } });
    s.addText(st[0], { x, y: 2.5, w: 2.9, h: 0.4, fontFace: F, fontSize: 15, bold: true, color: G, align: "center", margin: 0 });
    s.addText(st[1], { x, y: 2.92, w: 2.9, h: 0.9, fontFace: F, fontSize: 12, color: INK, align: "center", margin: 0, valign: "top" });
  });
  // 조직 관계
  card(s, 0.6, 4.3, 5.4, 2.4, "식량사업개발실 (팜사업운영그룹)", "• 인니 팜 농장 운영 관리 · 신규 사업개발\n• 팜유·곡물 등 원자재 시황 분석·전망\n• Market Intelligence: 리스크·구매 타이밍의 정량 근거 제공", G);
  card(s, 7.3, 4.3, 5.4, 2.4, "식량사업실", "• 곡물 원자재 실물 트레이딩(Back-to-back)\n• 매 Inquiry마다 공급가 + 운임 + 마진 견적\n• 고민: 언제·얼마나 살까, 시세 리스크 관리", BLUE);
  arrow(s, 6.1, 5.25, 7.2, 5.25, GOLD); arrow(s, 7.2, 5.75, 6.1, 5.75, MUTED);
  s.addText("정량 근거", { x: 6.0, y: 4.85, w: 1.3, h: 0.35, fontFace: F, fontSize: 10, color: GOLD, bold: true, align: "center", margin: 0 });
  s.addText("현장 질문", { x: 6.0, y: 5.85, w: 1.3, h: 0.35, fontFace: F, fontSize: 10, color: MUTED, align: "center", margin: 0 });
}
{ // 4 현재 사업 구조의 한계
  const s = base();
  header(s, "Ⅰ", "현재 사업 구조의 한계 — Back-to-back 거래", "고객도 시세를 알고 최저가 공급선을 고르는 구조 → 마진은 Handling charge 수준");
  const box = (x, y, t, sub, col) => { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.3, h: 1.1, fill: { color: col }, line: { color: col }, rectRadius: 0.1 });
    s.addText([{ text: t, options: { bold: true, fontSize: 15, breakLine: true } }, { text: sub, options: { fontSize: 10.5 } }], { x, y, w: 2.3, h: 1.1, fontFace: F, color: WHITE, align: "center", valign: "middle", margin: 2 }); };
  box(0.6, 2.0, "고객사", "Inquiry (물량·시기)", BLUE); box(3.6, 2.0, "당사", "동시간 견적 산정", G); box(6.6, 1.4, "공급사", "FOB 공급가", "6E7F74"); box(6.6, 2.65, "선사", "운임 견적", "6E7F74");
  arrow(s, 2.95, 2.55, 3.55, 2.55); arrow(s, 5.95, 2.45, 6.55, 1.95); arrow(s, 5.95, 2.65, 6.55, 3.2);
  s.addText("공급가 + 운임 + 마진 → 즉시 오퍼", { x: 0.6, y: 3.35, w: 5.6, h: 0.4, fontFace: F, fontSize: 12, color: MUTED, margin: 0 });
  s.addChart(pres.charts.BAR, [{ name: "공급가", labels: ["가격 구성 (예시)"], values: [92] }, { name: "운임", labels: ["가격 구성 (예시)"], values: [6.5] }, { name: "마진", labels: ["가격 구성 (예시)"], values: [1.5] }],
    C({ x: 9.3, y: 1.3, w: 3.5, h: 4.6, barDir: "col", barGrouping: "stacked", chartColors: ["9AA79F", BLUE, GOLD], showLegend: true, legendPos: "b", showValue: true, dataLabelPosition: "ctr", dataLabelColor: WHITE, valAxisHidden: true, valGridLine: { style: "none" }, showTitle: true, title: "판매가 100 기준 (예시)" }));
  bullets(s, [{ t: "구조적 한계", b: true, c: G }, "고객사도 시세를 실시간으로 앎 → 정보 우위 없음", "복수 공급선 사이에서 매번 최저가 선택 → Buyer loyalty 기대 어려움", "마진 = Handling charge 수준, 시세 변동 리스크는 고객·공급사에 전가",
    { t: "전환 방향", b: true, c: G }, "시황을 정량적으로 읽는 역량 → 구매 시점·물량 판단에 활용", "장기적으로 다품목 포트폴리오를 운영하는 독립 수익 사업으로 발전"], { x: 0.6, y: 3.95, w: 8.3, h: 2.9, fontSize: 13.5 });
  foot(s, "가격 구성 비율은 구조 설명용 예시이며 실제 거래 수치가 아님.");
}
{ // 5 문제의식 → 목표
  const s = base();
  header(s, "Ⅰ", "목표 — 단기 원가 절감, 중장기 독립 수익 사업", "같은 분석 엔진을 '지금의 구매 판단'과 '미래의 포트폴리오 사업'에 함께 씀");
  card(s, 0.6, 1.8, 5.8, 4.7, "단기 · 지금 바로", "", BLUE);
  bullets(s, ["옥수수·밀·대두·대두박 '지금 살까, 미룰까' 주간 판단", "판단마다 과거 검증(AUC·절감 bp)을 함께 제시", "작은 우위라도 100번·1,000번 반복하면 누적 원가 절감", "시세 리스크(VaR·4주 가격 범위)를 물량 단위 금액으로 환산"], { x: 0.8, y: 2.45, w: 5.4, h: 3.9, fontSize: 14 });
  arrow(s, 6.55, 4.15, 6.85, 4.15, GOLD);
  card(s, 6.95, 1.8, 5.8, 4.7, "중장기 · 사업 확장", "", GOLD);
  bullets(s, ["부서별 원자재 매매를 통합 관리하는 리스크 컨트롤 타워", "다품목 원자재 포트폴리오 운영 → 거래 마진 외 수익원", "목표 변동성·효율적 투자선으로 위험 한도 내 비중 결정", "모멘텀·가치 등 멀티 전략의 장기 강건성 사전 검증", "페이퍼 운용 → 실적 검증 → 사업화 단계적 접근"], { x: 7.15, y: 2.45, w: 5.4, h: 3.9, fontSize: 14 });
}
{ // 6 대시보드가 하는 일
  const s = base();
  header(s, "Ⅰ", "대시보드가 하는 일 — 한 장 요약", "가격 데이터 + 거시 + 텍스트를 하나로 묶어 리스크·전략·구매 판단까지 연결");
  s.addShape(pres.shapes.OVAL, { x: 5.17, y: 3.0, w: 3.0, h: 3.0, fill: { color: G }, line: { color: G } });
  s.addText("Commodity\nAnalytics Lab", { x: 5.17, y: 3.0, w: 3.0, h: 3.0, fontFace: F, fontSize: 17, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  const items = [["리스크 정량화", "동조 방향·크기 → VaR"], ["포트폴리오", "목표 변동성 · 효율적 투자선"], ["멀티 전략", "모멘텀 · 가치 · 최소분산"], ["AI 시계열", "Chronos · TFT · iTransformer"],
    ["텍스트·멀티모달", "FinBERT 감성 · 기상 · COT"], ["거시 레짐", "금리·인플레·VIX → 숨은 국면"], ["변동성 전이", "GARCH · DCC · BEKK"], ["장기 백테스트", "학습/시험 분리 · 강건성"]];
  items.forEach((it, i) => {
    const ang = -Math.PI / 2 + i * 2 * Math.PI / items.length, cx = 6.67 + 4.6 * Math.cos(ang), cy = 4.5 + 2.15 * Math.sin(ang);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx - 1.45, y: cy - 0.42, w: 2.9, h: 0.84, fill: { color: TINT }, line: { color: LINE }, rectRadius: 0.1 });
    s.addText([{ text: it[0], options: { bold: true, color: G, fontSize: 13, breakLine: true } }, { text: it[1], options: { color: MUTED, fontSize: 10 } }], { x: cx - 1.45, y: cy - 0.42, w: 2.9, h: 0.84, fontFace: F, align: "center", valign: "middle", margin: 1 });
  });
}

// ════════════════ Ⅱ. 개념 설명
divider("Ⅱ", "개념 설명 — 대시보드의 주요 모델", "수식 없이 직관으로: 무엇을 보고, 왜 쓰며, 대시보드 어디에 있나");
{ // 시계열
  const s = base();
  header(s, "Ⅱ", "시계열 데이터란", "시간 순서가 있는 관측치 — 순서 자체가 정보이므로 섞으면 안 됨");
  concept(s, [["한 줄 정의", "같은 대상을 일정 간격(일·주·월)으로 기록한 값의 흐름", 0.6], ["직관", "오늘 가격은 어제 가격에서 출발함. 그래서 '가격 수준'보다 '얼마나 변했나(수익률)'를 분석 단위로 씀", 0.9],
    ["원자재 예시", "옥수수 근월물 가격 → 주간 수익률. 2012년 가뭄·2022년 우크라이나 전쟁 때 큰 막대가 몰려 있음", 0.9], ["대시보드 위치", "모든 탭의 입력. 36개 원자재 + 거시 변수를 매일 자동 수집·정제", 0.6]]);
  img(s, "c07_timeseries", 5.8, 1.6, 7.0, 5.2);
  foot(s, "자료: Yahoo Finance 옥수수 근월물(롤 보정), 2006–2026.");
}
{ // Train/Test & look-ahead
  const s = base();
  header(s, "Ⅱ", "학습(Training)과 시험(Test)을 나누는 이유", "미래 정보가 조금만 섞여도 백테스트는 실제보다 훨씬 좋아 보임(Look-ahead bias)");
  // 타임라인
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.9, w: 6.6, h: 0.7, fill: { color: G }, line: { color: G } });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.2, y: 1.9, w: 5.5, h: 0.7, fill: { color: GOLD }, line: { color: GOLD } });
  s.addText("학습 2008–2016: 규칙·모델을 정함", { x: 0.6, y: 1.9, w: 6.6, h: 0.7, fontFace: F, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText("시험 2017–: 정한 규칙 그대로 성적 확인", { x: 7.2, y: 1.9, w: 5.5, h: 0.7, fontFace: F, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  bullets(s, [{ t: "지키는 원칙", b: true, c: G }, "매주 판단은 그 시점까지의 데이터만 사용, 체결은 다음 거래일 가격", "모델은 매년 1월, 그 이전 자료로만 다시 학습(walk-forward)", "계절성은 올해 이전 연도만, HMM은 '그때까지' 확률만 사용", "학습·시험 두 기간 모두에서 통한 설정만 채택(표에서 최고값 고르기 금지)",
    { t: "왜 중요한가", b: true, c: G }, "시간을 무시하고 무작위로 섞어 검증하면 같은 모델의 AUC가 0.69로 부풀려짐 — 시간 순서를 지키면 0.50(동전 던지기)"], { x: 0.6, y: 2.9, w: 6.8, h: 3.9, fontSize: 13 });
  s.addChart(pres.charts.BAR, [{ name: "AUC", labels: ["무작위 섞어 검증\n(미래 정보 누출)", "시간 순서 지켜 검증\n(실제 가능한 성적)"], values: [0.69, 0.50] }],
    C({ x: 7.7, y: 2.9, w: 5.0, h: 3.8, barDir: "col", chartColors: [RED, G], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0.4, valAxisMaxVal: 0.75, showLegend: false, showTitle: true, title: "같은 모델, 검증 방식만 다를 때 AUC", varyColors: true }));
  foot(s, "AUC 0.5 = 동전 던지기 수준. 출처: ④ 예측 최신 시계열 모델 검증(9품목, purged 교차검증 vs 무작위 K-fold).");
}
{ // 변동성
  const s = base();
  header(s, "Ⅱ", "변동성이란", "가격 방향이 아니라 '출렁임의 폭' — 그리고 큰 출렁임은 몰려서 옴");
  concept(s, [["한 줄 정의", "수익률이 평균에서 얼마나 흩어지는지(표준편차)를 연 단위로 표시한 값", 0.6], ["직관", "변동성 24% = 1년 뒤 가격이 대략 ±24% 범위에서 움직일 가능성이 큼(약 2/3 확률)", 0.9],
    ["원자재 예시", `옥수수 26주 변동성: 현재 ${D.corn_vol_now.toFixed(0)}% · 과거 최대 ${D.corn_vol_max.toFixed(0)}%. 위기·가뭄 때 급등 후 서서히 진정(군집)`, 0.9], ["대시보드 위치", "◆ 리스크, ③ 변동성, ★ 구매 카드의 '4주 가격 범위'", 0.6]]);
  img(s, "c09_volatility", 5.7, 1.8, 7.1, 4.6);
  foot(s, "자료: 옥수수 근월물 주간 수익률, 26주 실현 변동성(연율). 음영 = 주요 사건 구간.");
}
{ // GARCH
  const s = base();
  header(s, "Ⅱ", "GARCH 계열 — 변동성이 변동성을 부른다", "어제의 큰 충격과 높은 변동성이 오늘의 변동성을 키움 → 다음 기간 위험을 미리 계산");
  img(s, "c10_newsimpact", 0.5, 1.7, 5.4, 3.75);
  s.addText("하락 충격(왼쪽)이 같은 크기의 상승 충격보다 변동성을 더 키우는지를 비대칭 GARCH로 측정", { x: 0.6, y: 5.5, w: 5.2, h: 0.7, fontFace: F, fontSize: 11.5, color: MUTED, margin: 0 });
  const cs = [["GARCH", "단일 품목. 충격(α)·지속성(β)으로 다음 기간 변동성 예측. 1일 예측 12개 품목 중 12개에서 1위"], ["GJR · EGARCH", "하락·상승 충격의 비대칭 반영. 원자재는 상승 충격이 더 큰 '역레버리지'도 흔함"],
    ["DCC · ADCC", "품목 간 상관이 시점마다 변함. 위기 때 상관이 함께 오르는 '동조화' 포착"], ["BEKK", "한 품목의 충격이 다른 품목의 변동성으로 전이되는 방향·크기 추정(변동성 전이망)"]];
  cs.forEach((c, i) => card(s, 6.3, 1.7 + i * 1.28, 6.45, 1.15, c[0], c[1], i < 2 ? G : BLUE, { hs: 14, bs: 12 }));
  foot(s, "출처: ◆ 리스크 변동성 예측 경주(12품목, 2010~, QLIKE·MCS), ③ 변동성·공분산 탭.");
}
{ // 레짐
  const s = base();
  header(s, "Ⅱ", "레짐(국면)이란", "시장은 늘 같은 규칙으로 움직이지 않음 — 평시와 위기는 다른 세계");
  const cols = [["평시 국면", G, ["변동성 낮음, 품목별로 따로 움직임", "분산투자 효과 큼", "추세·가치 신호가 비교적 잘 통함"]], ["위기 국면", RED, ["변동성 급등(약 2.3배), 동반 하락", "상관 급등 → 분산투자 효과 소멸", "유동성·달러·위험회피가 가격 지배"]]];
  cols.forEach((c, i) => { card(s, 0.6 + i * 3.25, 1.8, 3.05, 3.2, c[0], "", c[1]); bullets(s, c[2], { x: 0.75 + i * 3.25, y: 2.4, w: 2.8, h: 2.5, fontSize: 12.5 }); });
  bullets(s, [{ t: "왜 중요한가", b: true, c: G }, "같은 전략·같은 리스크 모델도 국면에 따라 성과가 달라짐", "국면은 직접 관측되지 않는 '숨은 변수' → 통계 모형으로 추정", { t: "대시보드의 레짐 모형", b: true, c: G }, "HMM · Markov-switching 회귀 · TAR/STAR(문턱 모형) · Kalman TVP"], { x: 0.6, y: 5.2, w: 6.3, h: 1.7, fontSize: 12.5 });
  img(s, "c12_hmm", 7.1, 1.65, 5.7, 5.1);
  foot(s, "자료: 19개 원자재 동일 비중 바스켓 주간 수익률, 2국면 Gaussian HMM(설명용). 변동성 비율은 두 국면 표준편차 비. 음영 = 2008·2020·2022.");
}
{ // HMM
  const s = base();
  header(s, "Ⅱ", "Hidden Markov Model — 보이지 않는 국면을 확률로 추정", "관측되는 것(수익률·VIX·신용 스프레드)으로 '지금 위기일 확률'을 매주 갱신");
  // 상태 전이도
  s.addShape(pres.shapes.OVAL, { x: 1.0, y: 2.3, w: 2.2, h: 2.2, fill: { color: G }, line: { color: G } });
  s.addShape(pres.shapes.OVAL, { x: 4.6, y: 2.3, w: 2.2, h: 2.2, fill: { color: RED }, line: { color: RED } });
  s.addText("평시\n낮은 변동성", { x: 1.0, y: 2.3, w: 2.2, h: 2.2, fontFace: F, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText("위기\n높은 변동성", { x: 4.6, y: 2.3, w: 2.2, h: 2.2, fontFace: F, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  arrow(s, 3.25, 3.05, 4.55, 3.05, MUTED); arrow(s, 4.55, 3.75, 3.25, 3.75, MUTED);
  s.addText("전환 확률", { x: 3.2, y: 2.6, w: 1.4, h: 0.35, fontFace: F, fontSize: 10.5, color: MUTED, align: "center", margin: 0 });
  s.addText("유지 확률 높음 →\n국면은 한동안 지속", { x: 0.6, y: 4.65, w: 6.4, h: 0.6, fontFace: F, fontSize: 11.5, color: MUTED, align: "center", margin: 0 });
  bullets(s, [{ t: "작동 방식(직관)", b: true, c: G }, "국면마다 수익률 분포(평균·변동성)가 다르다고 가정", "관측값이 어느 국면 분포에 더 가까운지로 국면 확률 계산", "국면은 마르코프 연쇄로 전환(오늘 국면은 어제 국면에 좌우)",
    { t: "대시보드 결과", b: true, c: G }, "원자재 + VIX·신용 스프레드 다변량 HMM의 스트레스 국면 = 2008·2020·2022", "'지금 상태 설명'에는 유용, 다음 기간 방향 예측력은 우연 수준 → 구매 점수에는 미반영, 맥락 표시만"], { x: 7.3, y: 1.75, w: 5.5, h: 5.0, fontSize: 13 });
  foot(s, "출처: ② 레짐 탭(원자재 + 거시 다변량 HMM, 필터 확률만 사용).");
}
{ // Kalman
  const s = base();
  header(s, "Ⅱ", "Kalman 필터 — 변하는 관계를 실시간 추적", "새 관측이 들어올 때마다 '예측 → 오차만큼 보정'을 반복해 시변 계수를 추정");
  img(s, "c13_kalman", 0.5, 1.7, 7.6, 3.4);
  concept(s, [["한 줄 정의", "관계(예: 원유 1% 변화에 대한 옥수수 반응 β)가 시간에 따라 바뀐다고 보고 매 시점 갱신", 0.9], ["롤링 회귀와 차이", "롤링은 창 길이만큼 늦고 창 밖 정보를 갑자기 버림. 칼만은 불확실성을 고려해 빠르고 매끄럽게 추적", 0.9],
    ["대시보드 위치", "② 레짐(Kalman TVP), ◆ 리스크(시변 λ), ⑤ 롱숏(칼만 추세 전략)", 0.6]], 8.4, 1.75, 4.4);
  bullets(s, ["위기 때 헤지 비율·품목 간 민감도가 급변하는 구간을 빠르게 포착", "단, 1일 변동성 예측에서는 GARCH가 더 안정적 → 칼만은 진단·관계 추적용"], { x: 0.6, y: 5.3, w: 7.5, h: 1.5, fontSize: 13 });
  foot(s, "그림: 합성 데이터(참 관계를 아는 상태에서 비교한 설명용 예시).");
}
{ // VaR
  const s = base();
  header(s, "Ⅱ", "VaR(Value at Risk) — '최악의 하루'의 기준선", "100일 중 99일은 이 손실보다 덜 잃는다는 선 — 정규분포 가정은 꼬리를 과소평가함");
  img(s, "c14_var", 0.5, 1.65, 6.4, 3.5);
  const v = D.var_exceed;
  const labs = ["정규분포", "역사적 시뮬레이션", "Student-t", "FHS", "EVT"], keys = ["정규", "역사적 시뮬레이션 (500일)", "Student-t", "FHS (필터링 역사적)", "EVT-GPD (조건부 POT)"];
  s.addChart(pres.charts.BAR, [{ name: "실제 초과율 (%)", labels: labs, values: keys.map((k) => v[k]) }],
    C({ x: 7.2, y: 1.65, w: 5.6, h: 3.6, barDir: "col", chartColors: [RED, "C98A6F", GOLD, G, G], varyColors: true, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1.5, showLegend: false, showTitle: true, title: "1일 99% VaR 실제 초과율 (목표 1.00%)" }));
  bullets(s, ["정규분포 VaR는 실제 손실이 기준선을 더 자주 넘음 → 위험 과소평가", "FHS(변동성 반영 과거 시뮬레이션)·EVT(극단값 이론)가 목표 1%에 가장 근접", "★ 구매 카드의 '4주 범위 ±$M'은 같은 원리로 물량 금액 위험을 표시"], { x: 0.6, y: 5.35, w: 12.2, h: 1.5, fontSize: 13 });
  foot(s, "출처: ◆ 리스크 VaR 백테스트(risk_models_compare.csv, 12품목 평균, 보유 포지션, σ = GARCH(1,1)). 왼쪽 그림: 옥수수 일간 수익률 2007–2026.");
}
{ // 평균-분산
  const s = base();
  header(s, "Ⅱ", "포트폴리오 이론 — 섞으면 위험이 줄어든다", "상관이 낮은 자산을 섞으면 같은 수익에 더 낮은 위험 → 효율적 투자선 = 가장 좋은 조합의 경계");
  img(s, "c15_frontier", 6.2, 1.6, 6.6, 4.3);
  concept(s, [["평균-분산(Markowitz)", "기대수익은 높이고 변동성(위험)은 낮추는 비중을 찾음. 핵심 재료는 수익률·변동성·상관", 0.9], ["효율적 투자선", "같은 위험에서 수익이 가장 높은 조합들의 경계선. 그 아래 점은 개선 여지가 있는 조합", 0.9],
    ["목표 변동성", "위험 수준(예: 연 10%)을 먼저 정하고 직전 변동성에 맞춰 비중(레버리지)을 매주 조정. 목표를 높이면 수익도 비례해 커짐", 0.9], ["대시보드 위치", "⑤ Portfolio Customization(3D·2D 투자선), 목표변동성 백테스트", 0.6]], 0.6, 1.75, 5.3);
  foot(s, "자료: 최소분산·52주 모멘텀·가치 전략 주간 수익률(각 목표 변동성 10%), 2007–2026, 무작위 비중 5,000개.");
}
{ // 멀티 전략
  const s = base();
  header(s, "Ⅱ", "멀티 전략 — 모멘텀 · 가치 · 최소분산", "성격이 다른 전략을 섞으면 각 전략보다 위험 대비 성과가 좋아짐");
  const cs = [["52주 모멘텀", `품목마다 지난 52주 수익률이 +면 매수, −면 매도\n추세 지속 현상 활용 · Sharpe ${D.sharpe["52주 모멘텀"]}`, BLUE], ["가치", `5년 평균 대비 가장 싼 1/3 매수, 가장 비싼 1/3 매도\n평균 회귀 활용 · Sharpe ${D.sharpe["가치"]}`, GOLD], ["최소분산", `19개 품목을 롱으로 보유하되 변동성이 가장 작아지는 비중\n안정적 기초 수익 · Sharpe ${D.sharpe["최소분산"]}`, G]];
  cs.forEach((c, i) => card(s, 0.6, 1.7 + i * 1.3, 4.6, 1.2, c[0], c[1], c[2], { hs: 14, bs: 11 }));
  stat(s, 0.7, 5.62, 4.5, `Sharpe ${D.sharpe["가치 혼합"]}`, "가치 혼합(최소분산 50 + 모멘텀 25 + 가치 25)", RED);
  img(s, "c16_strategies", 5.5, 1.7, 7.3, 4.6);
  foot(s, "Sharpe = 연 수익 ÷ 연 변동성(위험 1단위당 수익). 2007–2026, 각 목표 변동성 10%, 편도 거래비용 5bp 차감, 연속월물 기준.");
}
function famTable(s, rows, y, colW, fs = 11.5, rh = 0.48) {
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, color: i === 0 ? WHITE : INK, fill: { color: i === 0 ? G : (i % 2 ? WHITE : TINT) } } }))),
    { x: 0.6, y, w: colW.reduce((a, b) => a + b, 0), colW, fontFace: F, fontSize: fs, border: { type: "solid", color: LINE, pt: 0.5 }, rowH: rh });
}
{ // ML 지도
  const s = base();
  header(s, "Ⅱ", "머신러닝 모델 지도 — 5개 계열", "변수가 많고 관계가 복잡한 원자재 자료를 다루기 위해 계열별 대표 모델을 같은 학습/시험 틀로 비교");
  const fam = [["변수 선택·규제", "LASSO · Ridge · Elastic Net\nAdaptive LASSO · SCAD · MCP", "많은 설명변수 중 핵심만 남김", G], ["차원 축소·요인", "PCA · PCR · PLS · 3PRF\nFAVAR · Ledoit-Wolf 수축", "공통 흐름으로 압축", BLUE], ["비선형·트리 앙상블", "Random Forest · XGBoost\nLightGBM · Gaussian Process", "상호작용·비선형 포착", GOLD],
    ["딥러닝·AI 시계열", "LSTM · TFT · iTransformer\nPatchTST · Chronos · TimesFM", "순서·패턴 학습", RED], ["베이지안·앙상블·불확실성", "BMA · Horseshoe · Minnesota BVAR\nStacking · 분위 회귀 · Conformal", "모델 불확실성·예측 범위", "6E7F74"]];
  fam.forEach((c, i) => { const x = 0.6 + i * 2.48;
    s.addShape(pres.shapes.OVAL, { x: x + 0.72, y: 1.8, w: 0.8, h: 0.8, fill: { color: c[3] }, line: { color: c[3] } });
    s.addText(String(i + 1), { x: x + 0.72, y: 1.8, w: 0.8, h: 0.8, fontFace: F, fontSize: 20, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.8, w: 2.3, h: 2.9, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addText(c[0], { x: x + 0.12, y: 2.9, w: 2.06, h: 0.6, fontFace: F, fontSize: 13.5, bold: true, color: c[3], align: "center", valign: "middle", margin: 0 });
    s.addText(c[1], { x: x + 0.12, y: 3.55, w: 2.06, h: 1.3, fontFace: F, fontSize: 11, color: INK, align: "center", valign: "top", margin: 0 });
    s.addText(c[2], { x: x + 0.12, y: 4.95, w: 2.06, h: 0.6, fontFace: F, fontSize: 10.5, color: MUTED, align: "center", valign: "middle", margin: 0 }); });
  bullets(s, ["모든 모델은 매년 그 이전 자료로만 재학습하고, 기존 전략(모멘텀·가치) 대비 추가 효과가 있는지 같은 기준으로 비교", "원리는 대시보드 ⑦ 모델 도감에서 슬라이더·3D 그림으로 확인 가능"], { x: 0.6, y: 5.95, w: 12.2, h: 0.95, fontSize: 12.5 });
}
{ // 변수 선택·규제
  const s = base();
  header(s, "Ⅱ", "① 변수 선택·규제 (Variable Selection) 계열", "설명변수가 많을 때 계수에 '벌점'을 줘서 쓸모없는 변수를 줄이거나 0으로 만듦 — 과적합 방지");
  famTable(s, [["모델", "하는 일", "특징"], ["Ridge", "모든 계수를 0 쪽으로 고르게 줄임", "변수를 없애지는 않음, 상관 높은 변수에 안정적"], ["LASSO", "중요하지 않은 계수를 정확히 0으로", "변수 선택 효과, 상관 높은 변수 중 하나만 남기는 경향"],
    ["Elastic Net", "Ridge + LASSO 혼합", "상관 높은 변수 묶음을 함께 남김"], ["Adaptive LASSO", "1차 추정이 큰 변수는 벌점을 작게", "큰 계수의 과도한 축소(편향) 완화"], ["SCAD · MCP", "작은 계수는 0, 큰 계수는 거의 깎지 않음", "비볼록 벌점 — '오라클' 성질(참 변수를 정확히 골라냄)"]], 1.75, [1.6, 3.2, 3.5], 11, 0.5);
  img(s, "c17_lasso", 9.0, 1.7, 3.9, 2.75);
  bullets(s, ["그림: 규제를 강하게 할수록(왼쪽) 계수가 하나씩 0이 되고, 옥수수를 설명하는 핵심 품목(대두·밀 등)만 남음", "대시보드: ④ 수익률 예측의 설명변수 선별, 멀티모달 54개 특성 압축에 사용"], { x: 0.6, y: 5.05, w: 12.2, h: 1.6, fontSize: 12.5 });
  foot(s, "그림 자료: 19개 품목 주간 수익률(2007–2026)로 옥수수를 설명한 LASSO 경로.");
}
{ // 차원 축소·요인
  const s = base();
  header(s, "Ⅱ", "② 차원 축소·요인 모형", "수십 개 변수를 소수의 '공통 흐름(요인)'으로 압축 — 정보는 살리고 잡음은 줄임");
  famTable(s, [["모델", "하는 일", "쓰임"], ["PCA", "변수들이 가장 넓게 퍼진 방향(주성분)을 찾음", "원자재 공통 요인, 시장 구조 파악"], ["PCR · PLS", "요인으로 압축 후 회귀(PLS는 목표와 관련 큰 방향 우선)", "변수가 많을 때 예측"],
    ["3PRF", "목표 변수에 맞춘 3단계 요인 추출", "거시 변수 다수 → 예측 요인"], ["FAVAR", "요인 + VAR로 거시 충격 전파 분석", "금리·달러 충격의 원자재 영향"], ["Ledoit-Wolf", "공분산 추정치를 안정적인 목표 쪽으로 수축", "포트폴리오 비중 계산의 잡음 감소"]], 1.75, [1.5, 3.8, 3.0], 11, 0.5);
  img(s, "c17_pca", 9.1, 1.7, 3.8, 2.65);
  bullets(s, [`19개 품목 움직임의 ${D.pc1.toFixed(0)}%를 첫 번째 공통 요인(원자재 전반의 위험선호)이 설명 → 나머지는 섹터·품목 고유 요인`, "대시보드: ① 시장 구조(PCA·FAVAR), ⑤ 포트폴리오(Ledoit-Wolf 공분산)"], { x: 0.6, y: 5.05, w: 12.2, h: 1.6, fontSize: 12.5 });
  foot(s, "그림 자료: 19개 품목 주간 수익률 상관행렬 고유값(2007–2026).");
}
{ // 비선형·딥러닝·베이지안
  const s = base();
  header(s, "Ⅱ", "③–⑤ 비선형 · 딥러닝 · 베이지안/불확실성", "선형으로 못 잡는 관계, 순서 패턴, 그리고 '얼마나 확신하나'까지 다루는 모델들");
  card(s, 0.6, 1.75, 3.95, 4.3, "③ 비선형·트리 앙상블", "• Random Forest: 나무 여러 그루의 평균\n• XGBoost · LightGBM: 앞 나무의 오차를 다음 나무가 보완(부스팅)\n• Gaussian Process: 비슷한 입력 → 비슷한 출력, 불확실성 함께 제공\n• Kernel Ridge: 비선형 회귀", GOLD, { bs: 12 });
  card(s, 4.7, 1.75, 3.95, 4.3, "④ 딥러닝", "• LSTM: 순서를 기억하는 신경망\n• Dense NN: 다층 신경망\n• TFT · iTransformer · PatchTST: 어텐션 기반 시계열\n• Chronos · TimesFM: 대규모 사전학습 파운데이션 모델\n(다음 장 상세)", RED, { bs: 12 });
  card(s, 8.8, 1.75, 4.0, 4.3, "⑤ 베이지안·앙상블·불확실성", "• BMA: 여러 모델을 사후확률로 가중 평균\n• Horseshoe · Minnesota BVAR: 베이지안 축소\n• Stacking · CSR: 모델 조합\n• 분위 회귀: 하위 5% 꼬리 직접 추정\n• Conformal: 분포 가정 없이 예측 범위 보장", "6E7F74", { bs: 12 });
  bullets(s, ["단일 모델보다 '여러 모델의 조합·불확실성 범위'가 실무 판단에 유용 — 대시보드는 모든 모델을 같은 학습/시험 틀로 비교"], { x: 0.6, y: 6.2, w: 12.2, h: 0.7, fontSize: 12.5 });
  foot(s, "출처: ④ 수익률 예측(대표 7 + 고급 20모델), ⑦ 모델 도감.");
}
{ // 대안 AI 시계열
  const s = base();
  header(s, "Ⅱ", "대안 AI 시계열 모델", "대규모 시계열로 미리 학습한 '파운데이션 모델'과 트랜스포머 계열 최신 모델을 같은 조건으로 검증");
  const fam = [["파운데이션 (학습 없이 예측)", "Chronos (Amazon) · TimesFM (Google)\n수많은 시계열로 사전학습 → 새 품목도 바로 예측", G], ["추세 + 계절", "Prophet (Meta)\n추세 변곡점과 연간 계절을 분리", GOLD], ["트랜스포머 · 어텐션", "TFT · iTransformer · PatchTST\n과거 중 중요한 시점·품목에 집중(어텐션)", BLUE], ["연속 시간 · 그래프", "Neural ODE · 시공간 GNN\n불규칙 시간 · 품목 간 연결망 반영", "6E7F74"]];
  fam.forEach((c, i) => card(s, 0.6 + (i % 2) * 3.3, 1.75 + Math.floor(i / 2) * 1.9, 3.1, 1.75, c[0], c[1], c[2], { hs: 13, bs: 11 }));
  img(s, "c18_fan", 7.3, 1.7, 5.5, 2.8);
  bullets(s, [{ t: "검증 결과(19개 품목, 2012–2026, 매주 4주 뒤 예측)", b: true, c: G }, "기존 52주 모멘텀·가치 전략이 여전히 가장 견고 → AI 모델은 보조 지표로 활용", "예측 범위(분위) 출력은 위험 범위 산출에 유용, 변동성은 GARCH 1위 · Chronos-2 3위"], { x: 7.3, y: 4.65, w: 5.5, h: 2.2, fontSize: 12.5 });
  foot(s, "출처: ⑤ 포트폴리오 › 대안 시계열 서브탭, ④ 예측 최신 시계열 모델 검증. 부채꼴 그림은 출력 형태 설명용 예시.");
}
{ // 멀티모달
  const s = base();
  header(s, "Ⅱ", "멀티모달 데이터 — 가격 밖의 정보", "텍스트·리포트·포지션·기상·거시를 품목별 주간 신호로 변환 (모든 값은 발표 이후에만 사용)");
  const br = [["① 텍스트", "USDA WASDE · EIA STEO · 연준 베이지북 서술 → FinBERT(금융 문장 감성 모델) 점수"], ["② 리포트 수치", "WASDE 재고율·생산 수정, EIA 주간 재고, LME 구리 재고, NASS 작황 우량 비율"], ["③ 포지션", "CFTC COT 19개 품목 투기·상업 순포지션, 쏠림, 4주 변화"],
    ["④ 기상", "엘니뇨(ONI)·남방진동(SOI)·지구 기온 + 주산지 22곳 강수·기온·냉난방 수요"], ["⑤ 매크로", "VIX · 신용 스프레드 · 달러 4주 변화"], ["⑥ 금융", "실질금리 · 기대인플레 · 금리차 · 환율 · 금융여건 · 원유/금 변동성"], ["⑦ 월간 거시", "CPI · 실업률 · 산업생산 · M2 · 소비자심리 · PPI · 고용"]];
  br.forEach((b, i) => { const x = 0.6 + (i % 4) * 3.1, y = 1.75 + Math.floor(i / 4) * 2.05; card(s, x, y, 2.95, 1.9, b[0], b[1], i < 3 ? BLUE : G, { hs: 14, bs: 11 }); });
  card(s, 9.9, 3.8, 2.95, 1.9, "검증 결과", "기존 가치 혼합(0.89)을 넘지 못함. 갈래 중 COT 포지션이 가장 나음. 구매 판단 AUC 개선 +0.01 미만", RED, { hs: 14, bs: 11, fill: "FBEDE7" });
  bullets(s, ["공통 값(기상·거시)은 '품목별 과거 민감도 × 현재 값'으로 품목 신호화", "신호 방향은 그 시점까지의 예측력으로만 결정(사후 부호 선택 없음)"], { x: 0.6, y: 6.0, w: 12.2, h: 0.9, fontSize: 12.5 });
  foot(s, "출처: ⑤ 포트폴리오 › 멀티모달 서브탭(2007–2026), ★ 구매 '보완 실험 2'. GDELT 뉴스는 수집 제한으로 제외.");
}

// ════════════════ Ⅲ. 대시보드 구성
divider("Ⅲ", "대시보드 구성", "지금 쓰는 기능 · 미래 사업 준비 · 검증 인프라");
{
  const s = base();
  header(s, "Ⅲ", "전체 구조 — 탭 지도", "의사결정에 바로 쓰는 탭과, 미래 사업·검증을 위한 탭을 분리");
  const grp = [["지금 쓰는 것", BLUE, [["★ 구매 의사결정", "지금 살까, 미룰까 · 4주 가격 범위"]]], ["미래 사업 준비", G, [["◆ 리스크", "VaR·ES · 변동성 예측 경주"], ["① 시장 구조", "상관 · PCA · 스필오버"], ["② 레짐", "HMM · MS · TAR · Kalman"], ["③ 변동성·공분산", "GARCH · DCC · BEKK"], ["⑤ 포트폴리오", "효율적 투자선 · 멀티 전략"]]],
    ["검증·연구 인프라", GOLD, [["④ 수익률 예측", "ML · AI 시계열 · 패널"], ["⑥ 실험 결과", "재실행 · 실행 기록"], ["⑦ 모델 도감", "모델 원리 2D·3D"], ["⑧ 데이터·방법론", "출처 · FAQ · 롤오버"]]]];
  let x = 0.6;
  grp.forEach(([name, col, tabs], gi) => {
    const w = gi === 0 ? 2.9 : 4.55;
    s.addText(name, { x, y: 1.75, w, h: 0.45, fontFace: F, fontSize: 16, bold: true, color: col, margin: 0 });
    tabs.forEach((t, i) => { const yy = 2.3 + i * 0.9;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: yy, w, h: 0.78, fill: { color: TINT }, line: { color: LINE }, rectRadius: 0.08 });
      dot(s, x + 0.18, yy + 0.28, col, 0.22);
      s.addText([{ text: t[0], options: { bold: true, fontSize: 13.5, color: INK, breakLine: true } }, { text: t[1], options: { fontSize: 10.5, color: MUTED } }], { x: x + 0.55, y: yy, w: w - 0.65, h: 0.78, fontFace: F, valign: "middle", margin: 0 }); });
    x += w + 0.35;
  });
  foot(s, "⑨ 설정(화면 배경) 탭 별도. 실행: Streamlit 웹 앱(사내 PC), 데이터 매일 자동 갱신.");
}
{
  const s = base();
  header(s, "Ⅲ", "데이터 파이프라인", "매일 자동 수집 → 정제 → 롤 보정 → 분석 → 대시보드");
  const st = [["수집", "Yahoo · EIA · USDA\nCFTC · FRED · NOAA"], ["정제", "이상치 · 누락\n품질 점검 로그"], ["롤 보정", "선물 월물 교체일\n가격 이음새 보정"], ["분석 엔진", "리스크 · 레짐 · 예측\n포트폴리오 · 구매"], ["대시보드", "10개 탭\n결과 고정·재실행"]];
  st.forEach((t, i) => { const x = 0.6 + i * 2.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.0, w: 2.15, h: 1.6, fill: { color: i === 4 ? GOLD : G }, line: { color: i === 4 ? GOLD : G }, rectRadius: 0.1 });
    s.addText([{ text: t[0], options: { bold: true, fontSize: 16, breakLine: true } }, { text: t[1], options: { fontSize: 10.5 } }], { x, y: 2.0, w: 2.15, h: 1.6, fontFace: F, color: WHITE, align: "center", valign: "middle", margin: 2 });
    if (i < 4) arrow(s, x + 2.18, 2.8, x + 2.47, 2.8, MUTED); });
  stat(s, 0.7, 4.1, 2.8, "36개", "원자재(곡물·에너지·금속·소프트·축산)"); stat(s, 3.7, 4.1, 2.8, "1990~", "가격 자료 시작(품목별 상이)"); stat(s, 6.7, 4.1, 2.8, "7갈래", "텍스트·리포트·COT·기상·거시 등"); stat(s, 9.7, 4.1, 3.1, "매일", "증분 수집·검증 자동화");
  bullets(s, [{ t: "곡물 3–4종만 골라 잘 맞는 결과를 보여 주지 않도록, 19–25개 원자재 전체에 같은 규칙을 적용해 검증(품목 선택 편향 방지)", b: true, c: G }, "가격 기준: 신호·수익률은 롤 보정 가격, 실제 매입가·크러시 마진은 근월물 가격 · 롤 보정 영향은 방법론 탭에 공개"], { x: 0.6, y: 5.6, w: 12.2, h: 1.3, fontSize: 13 });
  foot(s, "출처: ⑧ 데이터·방법론 탭, run_daily 파이프라인.");
}

// ════════════════ Ⅳ. 구매 의사결정
divider("Ⅳ", "지금 바로 쓰는 기능: 구매 의사결정", "옥수수 · 밀 · 대두 · 대두박 — 지금 살까, 미룰까");
{
  const s = base();
  header(s, "Ⅳ", "판단 로직 — 가격 매력 점수 × 더 쌀 확률", "두 축으로 매주 판단: 지금 싼가(점수) / 기다리면 더 싸질까(확률)");
  const rows = [["구성요소", "내용", "가중치"], ["가격 위치(저점)", "최근 6개월–1년 가격 범위 중 지금이 얼마나 낮은지", "27%"], ["단기 반등", "최근 10일 상승 여부(저점 통과 확인)", "18%"], ["상대가치", "대체재·원료 대비 싼지(크러시 마진, 옥수수/밀 비율 등)", "40%"], ["계절성", "과거 연도들에서 이 달이 평균적으로 쌌는지", "15%"], ["12개월 추세", "구매 시점 선택에선 손해 → 참고만", "0%"]];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? WHITE : INK, fill: { color: i === 0 ? G : (i % 2 ? WHITE : TINT) } } }))),
    { x: 0.6, y: 1.75, w: 6.6, colW: [1.6, 4.1, 0.9], fontFace: F, fontSize: 11.5, border: { type: "solid", color: LINE, pt: 0.5 }, rowH: 0.42 });
  bullets(s, [{ t: "더 쌀 확률", b: true, c: G }, "1–4주 안에 지금보다 쌀 확률을 로지스틱 회귀로 계산", "입력: 점수와 변화, 구성요소, 변동성, 1·4·13주 수익률, 연중 주차", "매년 1월 그 이전 자료로만 재학습, 근거(AUC > 0.55) 품목만 '보류' 권고"], { x: 0.6, y: 4.5, w: 6.6, h: 2.3, fontSize: 12.5 });
  // 판단 지도
  const ox = 7.8, oy = 1.85, gw = 4.9, gh = 4.5;
  s.addShape(pres.shapes.RECTANGLE, { x: ox + gw * 0.6, y: oy, w: gw * 0.4, h: gh, fill: { color: "E3F1E6" }, line: { color: "E3F1E6" } });
  s.addShape(pres.shapes.RECTANGLE, { x: ox, y: oy, w: gw * 0.6, h: gh * 0.45, fill: { color: "F8E3DA" }, line: { color: "F8E3DA" } });
  s.addShape(pres.shapes.RECTANGLE, { x: ox, y: oy + gh * 0.45, w: gw * 0.6, h: gh * 0.55, fill: { color: "F7F1DF" }, line: { color: "F7F1DF" } });
  s.addShape(pres.shapes.RECTANGLE, { x: ox, y: oy, w: gw, h: gh, fill: { color: WHITE, transparency: 100 }, line: { color: "9A9A94", width: 1 } });
  s.addText("적극 매수", { x: ox + gw * 0.62, y: oy + 0.08, w: 1.8, h: 0.3, fontFace: F, fontSize: 11, bold: true, color: "2E7D3E", margin: 0 });
  s.addText("매수 보류 (근거 품목만)", { x: ox + 0.08, y: oy + 0.08, w: 2.8, h: 0.3, fontFace: F, fontSize: 11, bold: true, color: RED, margin: 0 });
  s.addText("판단 유보", { x: ox + 0.08, y: oy + gh - 0.38, w: 1.8, h: 0.3, fontFace: F, fontSize: 11, bold: true, color: "9A7A1A", margin: 0 });
  const pts = [["옥수수", 60, 33, "2E7D3E", 0], ["밀", 41, 47, "C9A227", 0], ["대두", 43, 32, "C9A227", 0], ["대두박", 35, 45, "C9A227", 1]];
  pts.forEach(([n, sc, pr, col, left]) => { const x = ox + gw * sc / 100, y = oy + gh * (1 - pr / 100); dot(s, x - 0.13, y - 0.13, col, 0.26); s.addText(n, { x: left ? x - 1.15 : x + 0.15, y: left ? y + 0.05 : y - 0.18, w: 1.0, h: 0.34, align: left ? "right" : "left", fontFace: F, fontSize: 11, bold: true, color: INK, margin: 0 }); });
  s.addText("가격 매력 점수 →  (문턱 60)", { x: ox, y: oy + gh + 0.05, w: gw, h: 0.3, fontFace: F, fontSize: 10.5, color: MUTED, align: "center", margin: 0 });
  s.addText("더 쌀 확률 ↑ (문턱 55%)", { x: ox - 0.1, y: oy - 0.33, w: 3, h: 0.3, fontFace: F, fontSize: 10.5, color: MUTED, margin: 0 });
  foot(s, "판단 지도 위치는 기준일 2026-10-26 값. 출처: ★ 구매 의사결정 탭(추천 프리셋).");
}
{ // 횡단면 AI
  const s = base();
  header(s, "Ⅳ", "횡단면 분석 — 품목을 나란히 놓고 '비슷한 상태'끼리 묶기", "시계열이 '한 품목의 시간 변화'라면 횡단면은 '같은 시점의 품목 비교' — 지금 어디부터 살지 판단");
  const st = [["① 특성 계산", "품목마다 가격 매력 점수·위치·반등·상대가치·계절성 + 13·52주 수익률·고점 대비·변동성·추세"], ["② 표준화", "특성마다 품목 간 평균 0 · 표준편차 1로 맞춤(단위 차이 제거)"], ["③ 군집 (비지도 학습)", "K-평균 · 계층 Ward · GMM으로 가까운 품목끼리 묶음, 군집 수는 실루엣 점수로 자동 선택"], ["④ 경계 설명 (분류 모델)", "LDA = 군집이 가장 잘 갈리는 2D 지도 · 로지스틱 계수 = 어떤 특성이 군집을 가르나 · SVM = 군집이 얼마나 뚜렷한가"]];
  st.forEach((t, i) => { const y = 1.75 + i * 1.2;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y: y + 0.12, w: 0.5, h: 0.5, fill: { color: i < 2 ? G : BLUE }, line: { color: i < 2 ? G : BLUE } });
    s.addText(String(i + 1), { x: 0.6, y: y + 0.12, w: 0.5, h: 0.5, fontFace: F, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
    s.addText([{ text: t[0].slice(2), options: { bold: true, color: G, fontSize: 14, breakLine: true } }, { text: t[1], options: { fontSize: 11.5, color: INK } }], { x: 1.3, y, w: 5.6, h: 1.05, fontFace: F, valign: "top", margin: 0 }); });
  // 군집 도식
  const blobs = [["군집 1 · 매력 묶음 (평균 55점)", 7.3, 1.8, 2.8, "E3F1E6", "2E7D3E", ["옥수수", "밀", "대두유", "천연가스", "금", "백금"]], ["군집 2", 10.4, 1.8, 2.3, "EEF2F7", BLUE, []], ["군집 3", 10.4, 3.85, 2.3, "F7F1DF", "B8901C", []]];
  blobs.forEach(([n, x, y, d, fill, col, its]) => { s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d * 0.85, fill: { color: fill }, line: { color: col, width: 1.25, dashType: "dash" } });
    s.addText(n, { x, y: y + 0.12, w: d, h: 0.35, fontFace: F, fontSize: 11, bold: true, color: col, align: "center", margin: 0 });
    if (its.length) s.addText(its.join(" · "), { x: x + 0.25, y: y + 0.6, w: d - 0.5, h: d * 0.85 - 0.8, fontFace: F, fontSize: 11.5, color: INK, align: "center", valign: "middle", margin: 0 });
    else for (let k = 0; k < 5; k++) dot(s, x + 0.45 + (k % 3) * 0.55, y + 0.75 + Math.floor(k / 3) * 0.5, col, 0.18); });
  bullets(s, ["군집 1 품목에 ★ 판단 카드 '매력 묶음' 배지 표시 → 여러 품목 중 우선순위 판단", "같은 군집 = 비슷한 이유로 싸진 품목 → 한꺼번에 사면 분산 효과 작음(상관 함께 표시)", "오늘 상관 = 최근 2년, 최근일수록 큰 가중(반감기 8주)"], { x: 7.3, y: 5.85, w: 5.5, h: 1.1, fontSize: 11 });
  foot(s, "군집 1 품목은 기준일 2026-10-26 ★ 판단 카드 배지 기준(19개 품목). 출처: ★ 구매 의사결정 › B. 횡단면 하위 탭.");
}
{
  const s = base();
  header(s, "Ⅳ", "이번 주 판단 — 기준일 2026-10-26", "옥수수는 저점권 매수 신호, 밀·대두·대두박은 비싼 편이라 급하지 않으면 다음 주 재확인");
  const cs = [["옥수수", "적극 매수", "2E7D3E", "60", "33%", "$177.9/t · 카고 $5.34M", "위치 89↑ · 반등 8↓ · 상대가치 65↑\n52주 추세 −6% (하락 중)", "학습 AUC 0.56 · 시험 0.55\n시험 판단당 +12.6bp (무작위 −8.9)", "−0.52M ~ +0.58M"],
    ["밀 (CBOT)", "판단 유보 · 대기", "B8901C", "41", "47%", "$229.1/t · 카고 $6.87M", "반등 9↓ · 상대가치 34↓ · 위치 62\n52주 추세 +15% (상승 중)", "학습 AUC 0.50 · 시험 0.51\n기준치 미달 → 확률 미사용", "−0.60M ~ +0.77M"],
    ["대두", "판단 유보 · 대기", "B8901C", "43", "32%", "$456.5/t · 카고 $13.69M", "반등 13↓ · 상대가치 42 · 위치 60\n52주 추세 +9% (상승 중)", "학습 AUC 0.54 · 시험 0.51\n기준치 미달 → 확률 미사용", "−0.84M ~ +1.27M"],
    ["대두박", "판단 유보 · 대기", "B8901C", "35", "45%", "$373.0/t · 카고 $11.19M", "위치 15↓ · 반등 5↓ · 상대가치 59\n52주 추세 +15% (상승 중)", "학습 AUC 0.57 · 시험 0.46\n시험에서 약화 → 재검토 대상", "−1.04M ~ +1.92M"]];
  cs.forEach((c, i) => { const x = 0.6 + i * 3.1, y = 1.75, w = 2.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 5.05, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addText(c[0], { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: 0.45, fontFace: F, fontSize: 17, bold: true, color: INK, margin: 0 });
    dot(s, x + 0.2, y + 0.72, c[2], 0.24); s.addText(c[1], { x: x + 0.52, y: y + 0.66, w: w - 0.7, h: 0.36, fontFace: F, fontSize: 13, bold: true, color: c[2], margin: 0 });
    s.addText([{ text: c[3], options: { fontSize: 30, bold: true, color: G } }, { text: " 점", options: { fontSize: 12, color: MUTED } }], { x: x + 0.2, y: y + 1.1, w: 1.5, h: 0.7, fontFace: F, margin: 0 });
    s.addText([{ text: c[4], options: { fontSize: 22, bold: true, color: BLUE } }, { text: "\n더 쌀 확률", options: { fontSize: 10, color: MUTED } }], { x: x + 1.55, y: y + 1.05, w: 1.3, h: 0.8, fontFace: F, margin: 0 });
    s.addText(c[5], { x: x + 0.2, y: y + 1.95, w: w - 0.4, h: 0.35, fontFace: F, fontSize: 11, color: MUTED, margin: 0 });
    s.addText([{ text: "근거", options: { bold: true, color: GOLD, breakLine: true } }, { text: c[6], options: { breakLine: true } }, { text: "검증", options: { bold: true, color: GOLD, breakLine: true } }, { text: c[7], options: { breakLine: true } }, { text: "4주 가격 범위(카고 금액)", options: { bold: true, color: GOLD, breakLine: true } }, { text: c[8] + " $" }],
      { x: x + 0.2, y: y + 2.35, w: w - 0.4, h: 2.6, fontFace: F, fontSize: 10.5, color: INK, margin: 0, valign: "top", paraSpaceAfter: 2 }); });
  foot(s, "카고 = 30,000톤 가정. 점수 문턱 60, 확률 문턱 55%(AUC > 0.55 품목만 적용). 출처: ★ 구매 의사결정 탭, 기준일 2026-10-26.");
}
{
  const s = base();
  header(s, "Ⅳ", "검증 — 동전 던지기(50%)보다 나은가", "학습 기간에 고른 설정을 시험 기간에 그대로 적용해 확인 — 근거 있는 품목에만 '미루기' 권고");
  s.addChart(pres.charts.BAR, [{ name: "학습 2008–16", labels: ["옥수수", "밀", "대두", "대두박"], values: [0.56, 0.50, 0.54, 0.57] }, { name: "시험 2017–", labels: ["옥수수", "밀", "대두", "대두박"], values: [0.55, 0.51, 0.51, 0.46] }],
    C({ x: 0.5, y: 1.7, w: 6.6, h: 4.4, barDir: "col", chartColors: [G, GOLD], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0.4, valAxisMaxVal: 0.62, showLegend: true, legendPos: "b", showTitle: true, title: "'더 쌀 확률' 예측 AUC — 곡물 4종 (0.50 = 동전)" }));
  stat(s, 7.5, 1.75, 1.7, "25개", "검증 품목 수", G); stat(s, 9.2, 1.75, 1.8, "0.570", "근거 9개 시험 AUC", BLUE); stat(s, 11.0, 1.75, 1.8, "+23bp", "판단당 절감\n(무작위 −14bp)", G);
  bullets(s, [{ t: "해석", b: true, c: G }, "옥수수: 학습·시험 모두 기준 근처(0.56/0.55) → 판단 활용 우선 품목", "밀·대두: 동전 수준 → '더 쌀 확률'은 쓰지 않고 가격 매력 점수만", "대두박: 학습 0.57 → 시험 0.46으로 약화 → 재검토 대상(정직하게 공개)",
    { t: "검증 범위", b: true, c: G }, "곡물만이 아니라 원자재 25개 품목 전체에 같은 규칙을 적용 → 근거가 확인된 9개 품목에만 '미루기' 권고(선택 편향 방지)"], { x: 7.5, y: 3.15, w: 5.3, h: 3.7, fontSize: 12.5 });
  foot(s, "근거 품목 9개 = 학습 AUC > 0.55(코코아·면화·비육 송아지·돈육·생우·오렌지주스·휘발유·옥수수·대두박). 출처: ★ 구매 탭 품목별 AUC, HANDOFF 검증 기록.");
}
{
  const s = base();
  header(s, "Ⅳ", "반복 시행의 가치 — 작은 우위의 누적", "판단 한 번의 차이는 작아도, 매주 반복하면 연간 원가 절감으로 누적");
  const per = 5.34e6 * 12.6 / 1e4;
  s.addChart(pres.charts.BAR, [{ name: "연간 절감 ($천)", labels: ["분기 1회 (4번)", "월 1회 (12번)", "격주 (26번)", "매주 (52번)"], values: [4, 12, 26, 52].map((n) => Math.round(per * n / 1000)) }],
    C({ x: 0.5, y: 1.7, w: 6.8, h: 4.5, barDir: "col", chartColors: [G], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0", showLegend: false, showTitle: true, title: "옥수수 카고 1건 판단 빈도별 연간 절감 (예시, $천)", valAxisLabelFormatCode: "#,##0" }));
  stat(s, 7.7, 1.75, 2.5, `$${(per / 1000).toFixed(1)}천`, "판단 1회당 기대 절감 (옥수수 카고 1건)", G);
  stat(s, 10.3, 1.75, 2.5, "+12.6bp", "옥수수 시험 기간 판단당 절감", BLUE);
  bullets(s, [{ t: "계산 방식", b: true, c: G }, "판단당 절감 bp × 카고 금액($177.9/t × 30,000톤 = $5.34M) × 연간 판단 횟수", "무작위 미루기(−8.9bp) 대비가 아니라 '즉시 구매' 대비 기준(보수적)", { t: "유의 사항", b: true, c: RED }, "과거 평균이 미래에도 유지된다는 가정 — 실제 물량·횟수로 재계산 필요", "판단 하나하나는 손해일 수 있음(확률적 우위) → 반복할수록 평균에 수렴"], { x: 7.7, y: 3.15, w: 5.1, h: 3.7, fontSize: 12.5 });
  foot(s, "예시 계산: 카고 30,000톤·가격은 기준일 2026-10-26 앱 값. 실제 연간 구매 물량·횟수는 확인 필요(사내 수치).");
}

// ════════════════ Ⅴ. 리스크·포트폴리오
divider("Ⅴ", "미래 사업 준비: 리스크 · 포트폴리오", "동조성 · 변동성 전이 · 효율적 투자선 · 무엇이 통하나");
{
  const s = base();
  header(s, "Ⅴ", "동조성 — 무엇이 함께 움직이나", "같은 섹터는 강하게 동조, 섹터 간은 약함 → 위기 때는 섹터를 넘어 함께 움직임");
  img(s, "c26_corr", 0.5, 1.6, 6.2, 5.3);
  img(s, "c27_corr3d", 6.9, 1.55, 5.9, 3.6);
  bullets(s, ["곡물(옥수수·대두·대두박·밀)과 에너지(원유·휘발유·난방유)는 섹터 내 상관 높음", "2008·2020·2022 위기 구간에 품목 쌍 상관이 함께 솟음 → 분산투자 효과 약화", "→ 리스크 한도는 평시가 아닌 위기 상관 기준으로 설정 필요"], { x: 6.9, y: 5.25, w: 5.9, h: 1.6, fontSize: 12.5 });
  foot(s, "자료: 주간 로그수익률(롤 보정), 상관 2007–2026, 지형은 52주 롤링 상관(DCC는 ③ 탭).");
}
{
  const s = base();
  header(s, "Ⅴ", "변동성 전이 — 위험은 어디서 어디로 옮겨 가나", "BEKK(다변량 GARCH)로 '한 품목의 충격 → 다른 품목의 변동성' 경로를 기간별로 추정");
  const rows = [["기간", "전이가 가장 많은 3품목 조합", "유의 전이 경로 수"], ["전체", "대두 · 구리 · 금", "6 (충격 3 + 변동성 3)"], ["2016년까지", "대두 · 구리 · 금", "5 (충격 2 + 변동성 3)"], ["2017년 이후", "대두박 · 원유 · 금", "6 (충격 4 + 변동성 2)"]];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? WHITE : INK, fill: { color: i === 0 ? G : (i % 2 ? WHITE : TINT) } } }))), { x: 0.6, y: 1.8, w: 7.0, colW: [1.7, 3.0, 2.3], fontFace: F, fontSize: 13, border: { type: "solid", color: LINE, pt: 0.5 }, rowH: 0.55 });
  stat(s, 8.2, 1.8, 4.5, "금", "세 기간 모두 전이 중심에 포함된 품목", GOLD);
  stat(s, 8.2, 3.3, 4.5, "원유", "2017년 이후 새로 전이 중심에 등장", RED);
  bullets(s, [{ t: "실무 함의", b: true, c: G }, "곡물 가격 위험이 곡물 내부에서만 오지 않음 — 원유·금속 충격이 곡물 변동성으로 전이", "2017년 이후 에너지(원유) → 대두박 경로 부상 → 바이오연료·크러시 마진 연계 위험 점검 필요", "BEKK는 수익률이 아니라 '위험(분산)'의 전이 — 방향 예측 도구 아님"], { x: 0.6, y: 4.3, w: 12.2, h: 2.5, fontSize: 13 });
  foot(s, "출처: ③ 변동성·공분산 › BEKK 품목 조합 자동 찾기(근사 점수 → Full BEKK 확인, |t| ≥ 1.96 기준).");
}
{
  const s = base();
  header(s, "Ⅴ", "포트폴리오 — 목표 변동성을 높이면 수익도 비례해 커짐", "위험 1단위당 수익(Sharpe)은 유지된 채 수익 규모만 커짐 — 감내 가능한 낙폭 수준에서 목표를 정하는 것이 핵심");
  img(s, "c28_frontier3d", 0.4, 1.6, 5.4, 3.9);
  const vt = D.voltgt;
  s.addChart(pres.charts.LINE, [{ name: "연수익 산술 (%)", labels: vt.tgt.map((t) => t + "%"), values: vt.arith }, { name: "연수익 복리 (%)", labels: vt.tgt.map((t) => t + "%"), values: vt.cagr }, { name: "최대 낙폭 (%)", labels: vt.tgt.map((t) => t + "%"), values: vt.mdd }],
    C({ x: 6.0, y: 1.6, w: 6.8, h: 3.9, chartColors: [G, GOLD, RED], lineSize: 2.5, lineDataSymbolSize: 6, showLegend: true, legendPos: "b", showTitle: true, title: "가치 혼합 — 목표 변동성별 결과 (2007–2026, 배율 상한 4배)" }));
  stat(s, 0.7, 5.6, 2.6, "9.6%", "목표 10% 연수익 (최대 낙폭 −24.5%)", G);
  stat(s, 3.4, 5.6, 2.6, "18.6%", "목표 20% 연수익 (최대 낙폭 −45%)", GOLD);
  stat(s, 6.1, 5.6, 2.6, "27.5%", "목표 30% 연수익 (최대 낙폭 −60%)", RED);
  bullets(s, ["목표를 여유 있게 잡을수록 수익은 거의 비례 상승, 낙폭도 함께 커짐", "복리 수익은 덜 늘어남(50%: 산술 45% vs 복리 36%)"], { x: 8.8, y: 5.6, w: 4.0, h: 1.3, fontSize: 11.5 });
  foot(s, "편도 거래비용 5bp 차감, 차입·증거금 이자 미반영, 연속월물 기준(ETF 기준 대비 연 약 3%p 과대 가능). 출처: ⑤ 목표변동성별 수익율 Backtest.");
}
{
  const s = base();
  header(s, "Ⅴ", "검증 결과 — 채택한 것과 보류한 것", "새 데이터·새 모델을 더해도 '시험 기간'에 개선이 남지 않으면 채택하지 않음 → 장기 강건한 조합만 남김");
  card(s, 0.6, 1.75, 3.9, 4.0, "채택 · 장기 강건", "• 52주 시계열 모멘텀\n• 가치·최소분산 혼합 (Sharpe 0.88)\n• 변동성 예측: GARCH · MIDAS\n• 꼬리 위험: FHS · EVT VaR\n• 구매: 근거 품목 미루기 (시험 AUC 0.57)", G, { bs: 12.5 });
  card(s, 4.7, 1.75, 3.9, 4.0, "보류 · 추가 효과 제한적", "• AI 시계열 모델 추가 (시험 기간 개선 없음)\n• 멀티모달 추가 (학습 최적 비중 0%)\n• 기관 리포트 발표 후 신호\n• 레짐 기반 타이밍 전환\n→ 맥락·보조 지표로만 표시", RED, { bs: 12.5, fill: "FBEDE7" });
  s.addChart(pres.charts.BAR, [{ name: "기존 3종", labels: ["학습 최적 (~2016)", "시험 실제 (2017~)"], values: [1.24, 0.87] }, { name: "기존 3종 + 대안 AI 6종", labels: ["학습 최적 (~2016)", "시험 실제 (2017~)"], values: [1.75, 0.19] }],
    C({ x: 8.8, y: 1.7, w: 4.0, h: 4.1, barDir: "col", chartColors: ["8A8A84", RED], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: true, legendPos: "b", valAxisMinVal: 0, valAxisMaxVal: 2, showTitle: true, title: "AI 모델을 더했을 때 Sharpe" }));
  bullets(s, ["학습 기간에는 AI 모델을 더하면 크게 좋아 보였으나(1.24 → 1.75), 같은 비중을 시험 기간에 쓰면 오히려 악화(0.87 → 0.19)", "→ '좋아 보이는 결과'의 상당 부분은 사후 선택. 학습/시험 분리가 의사결정 품질을 지킴"], { x: 0.6, y: 5.95, w: 12.2, h: 0.95, fontSize: 12.5 });
  foot(s, "출처: ⑤ 대안 시계열 서브탭 스패닝 투자선(학습 2012-07–2016, 시험 2017–2026), HANDOFF 핵심 결론.");
}

// ════════════════ Ⅵ. 결론
divider("Ⅵ", "결론 · 제언", "전사 원자재 리스크 컨트롤 타워로의 확장 · 단계별 활용 제안");
{ // 컨트롤 타워·확장성
  const s = base();
  header(s, "Ⅵ", "확장 비전 — 전사 원자재 리스크 컨트롤 타워", "금융 트레이딩을 하지 않더라도, 부서별로 나뉜 원자재 매매를 통합 관리해 회사 전체의 가격 변동성 위험을 줄일 수 있음");
  const depts = [["곡물", "옥수수·밀·대두·대두박"], ["팜유", "CPO·PKO (팜 농장 법인)"], ["에너지", "연료·운임 연동"], ["기타 원자재", "금속·소프트 등"]];
  depts.forEach((d, i) => { const y = 1.8 + i * 1.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 2.4, h: 0.85, fill: { color: TINT }, line: { color: LINE }, rectRadius: 0.08 });
    s.addText([{ text: d[0] + " 부서", options: { bold: true, fontSize: 13, color: INK, breakLine: true } }, { text: d[1], options: { fontSize: 10, color: MUTED } }], { x: 0.6, y, w: 2.4, h: 0.85, fontFace: F, align: "center", valign: "middle", margin: 0 });
    arrow(s, 3.05, y + 0.42, 3.75, 3.85, MUTED); });
  s.addShape(pres.shapes.OVAL, { x: 3.8, y: 2.6, w: 2.5, h: 2.5, fill: { color: G }, line: { color: G } });
  s.addText("리스크\n컨트롤 타워", { x: 3.8, y: 2.6, w: 2.5, h: 2.5, fontFace: F, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  const outs = ["전사 포지션 합산 · 순 위험(VaR)", "품목 간 상관·전이 반영 → 자연 헤지 식별", "목표 변동성 기준 위험 한도 배분", "구매 시점·물량의 통합 의사결정"];
  outs.forEach((o, i) => { const y = 1.95 + i * 0.95; arrow(s, 6.35, 3.85, 6.85, y + 0.32, GOLD);
    s.addText(o, { x: 6.9, y, w: 2.6, h: 0.65, fontFace: F, fontSize: 11.5, bold: true, color: G, valign: "middle", margin: 0 }); });
  card(s, 9.7, 1.75, 3.1, 2.45, "학계·헤지펀드 검증 방법론", "• 평균-분산: Markowitz (1952)\n• GARCH·DCC: Bollerslev (1986), Engle (2002)\n• 레짐 전환: Hamilton (1989)\n• 시계열 모멘텀: Moskowitz·Ooi·Pedersen (2012)\n• 가치·모멘텀: Asness 외 (2013)", BLUE, { hs: 13, bs: 10.5 });
  card(s, 9.7, 4.35, 3.1, 2.4, "모든 가격 자산으로 확장", "원자재뿐 아니라 주식·채권·환율 등 가격이 있는 모든 자산에 같은 엔진(리스크·전략·검증)을 그대로 적용 가능", GOLD, { hs: 13, bs: 11.5 });
  bullets(s, ["부서별로는 작아 보이는 위험도, 합치면 상관 때문에 커지거나(동조) 서로 상쇄됨(자연 헤지) — 통합해서 봐야 실제 위험이 보임"], { x: 0.6, y: 6.15, w: 8.9, h: 0.75, fontSize: 12 });
  foot(s, "부서 구분은 개념 예시. 인용 방법론은 대시보드에 구현된 표준 모형의 원전.");
}
{
  const s = base();
  header(s, "Ⅵ", "활용 제안 — 3단계", "검증된 기능부터 작게 시작해 실적으로 확장");
  const st = [["1단계 · 즉시", "주간 구매 판단 공유", "• 식량사업실 주간 회의에 ★ 판단 카드 공유\n• 옥수수 우선, 밀·대두는 점수만 참고\n• 실제 구매 결과를 기록해 사후 검증", BLUE], ["2단계 · 6개월", "리스크 관리 연계", "• 카고 단위 4주 가격 범위(VaR)로 물량 위험 금액화\n• 위기 상관 기준 리스크 한도·헤지 비율 참고\n• 변동성 전이(원유→대두박 등) 모니터링", G], ["3단계 · 1년+", "포트폴리오 사업 검토", "• 소규모 페이퍼 포트폴리오 운용(목표 변동성 10%)\n• 실적·낙폭을 실제 시간으로 검증\n• 검증 후 독립 수익 사업화 여부 판단", GOLD]];
  st.forEach((c, i) => { const x = 0.6 + i * 4.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: 3.95, h: 3.6, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addText(c[0], { x: x + 0.2, y: 1.9, w: 3.5, h: 0.35, fontFace: F, fontSize: 12, bold: true, color: c[3], margin: 0 });
    s.addText(c[1], { x: x + 0.2, y: 2.25, w: 3.5, h: 0.5, fontFace: F, fontSize: 18, bold: true, color: INK, margin: 0 });
    s.addText(c[2], { x: x + 0.2, y: 2.9, w: 3.6, h: 2.3, fontFace: F, fontSize: 12, color: INK, margin: 0, valign: "top", paraSpaceAfter: 4 });
    if (i < 2) arrow(s, x + 3.97, 3.55, x + 4.13, 3.55, MUTED); });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.6, w: 12.2, h: 1.25, fill: { color: "FBF4E1" }, line: { color: GOLD, width: 1 }, rectRadius: 0.08 });
  s.addText([{ text: "의사결정 요청   ", options: { bold: true, color: RED } }, { text: "① 1단계(식량사업실 주간 판단 공유) 시범 운영 승인  ② 실제 구매 기록(일자·물량·단가) 공유 협조 — 사후 검증용  ③ 3단계 검토를 위한 페이퍼 운용 기준(위험 한도) 협의" }],
    { x: 0.85, y: 5.6, w: 11.8, h: 1.25, fontFace: F, fontSize: 13, color: INK, valign: "middle", margin: 0 });
}
{
  const s = base();
  header(s, "Ⅵ", "리스크 · 한계와 다음 단계", "도구는 판단을 대체하지 않음 — 한계를 알고 쓰는 것이 핵심");
  const rk = [["과거 ≠ 미래", "검증 성과는 과거 자료 기준. 구조 변화(정책·기후) 시 약화 가능 → 실시간 기록으로 지속 점검", RED], ["짧은 표본·과적합", "시험 기간 약 10년, 품목별 판단 수 제한 → 학습/시험 분리·위약 비교로 방어", RED],
    ["데이터 한계", "연속월물 롤 비용 과대(ETF 기준 연 약 3%p), 일부 품목 롤 미보정 → 방법론 탭에 공개", GOLD], ["활용 범위", "확률적 우위 — 개별 판단은 틀릴 수 있음. 물량·계약 조건 등 현장 판단과 병행", GOLD]];
  rk.forEach((r, i) => card(s, 0.6 + (i % 2) * 4.0, 1.75 + Math.floor(i / 2) * 2.25, 3.8, 2.05, r[0], r[1], r[2], { bs: 12 }));
  card(s, 8.8, 1.75, 4.0, 4.3, "다음 단계", "• 실제 구매 기록 연동 → 판단별 사후 성과 추적\n• 팜유(CPO)·운임 확장 — 팜 사업과 연계\n• 대두박 등 시험 약화 품목 재검증\n• 위기 국면 조기 경보(레짐) 맥락 배지\n• 발표 자료 HTML판(인터랙티브 차트)", BLUE, { bs: 12.5 });
}

// ════════════════ 별첨
divider("별첨", "용어 · 데이터 · 검증 원칙 · 모델 목록");
{
  const s = base();
  header(s, "A1", "용어 정의");
  const rows = [["용어", "뜻"], ["AUC", "맞힐 확률의 지표. 0.5 = 동전 던지기, 1.0 = 완벽. '더 쌀 확률'이 실제와 얼마나 일치하는지"], ["Sharpe", "연 수익 ÷ 연 변동성. 위험 1단위당 수익(1.0 이상이면 우수)"], ["MDD (최대 낙폭)", "고점 대비 가장 크게 떨어진 비율"],
    ["VaR / ES", "VaR = 정해진 확률(99%)로 넘지 않을 손실선, ES = 그 선을 넘었을 때의 평균 손실"], ["bp", "0.01%. 10bp = 0.1%"], ["변동성", "수익률 표준편차의 연 단위 값. 가격 출렁임의 폭"], ["스패닝", "새 전략을 더했을 때 효율적 투자선이 바깥으로 밀려나는지(새 정보가 있는지) 검정"],
    ["위약(Placebo)", "신호 순서를 무작위로 섞은 가짜 신호. 진짜 신호가 이보다 확실히 나아야 채택"], ["Look-ahead bias", "그 시점에 알 수 없던 미래 정보가 섞여 성과가 부풀려지는 오류"], ["목표 변동성", "포트폴리오 위험 수준을 미리 정하고 비중(레버리지)을 조절하는 방식"]];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, color: i === 0 ? WHITE : INK, fill: { color: i === 0 ? G : (i % 2 ? WHITE : TINT) } } }))), { x: 0.6, y: 1.35, w: 12.2, colW: [2.4, 9.8], fontFace: F, fontSize: 12, border: { type: "solid", color: LINE, pt: 0.5 }, rowH: 0.47 });
}
{
  const s = base();
  header(s, "A2", "데이터 출처 · 기간");
  const rows = [["구분", "출처", "내용", "기간"], ["가격", "Yahoo Finance · 거래소", "36개 원자재 선물 근월물·연속월물(롤 보정)", "1990~ (품목별)"], ["에너지 재고·전망", "EIA (STEO·주간)", "원유·연료·가스 재고, 가격 전망 서술", "2010~"], ["곡물 수급", "USDA WASDE · NASS", "기말재고·재고율·생산 수정, 작황 우량 비율, 서술 문단", "1997~"],
    ["포지션", "CFTC COT", "19개 품목 투기·상업 순포지션", "1990년대~"], ["거시·금융", "FRED", "금리·인플레·VIX·신용 스프레드·환율·금융여건", "1990~"], ["기상", "NOAA · Open-Meteo", "엘니뇨·남방진동·지구 기온, 주산지 22곳 강수·기온", "1990~"], ["텍스트", "연준 베이지북 · WASDE · STEO", "FinBERT 금융 감성 점수", "2007~"]];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? WHITE : INK, fill: { color: i === 0 ? G : (i % 2 ? WHITE : TINT) } } }))), { x: 0.6, y: 1.35, w: 12.2, colW: [1.9, 3.0, 5.3, 2.0], fontFace: F, fontSize: 12, border: { type: "solid", color: LINE, pt: 0.5 }, rowH: 0.55 });
  foot(s, "모든 데이터는 발표·공개 이후 시점부터 사용. 세부 출처·품목별 기간은 대시보드 ⑧ 데이터·방법론 탭.");
}
{
  const s = base();
  header(s, "A3", "검증 원칙 — 결과를 믿을 수 있게 만드는 규칙");
  const ps = [["미래정보 차단", "모든 신호는 결정 시점까지 데이터만, 체결은 다음 거래일. 미래 데이터를 바꿔도 과거 판단이 안 바뀌는지 자동 테스트"], ["학습 / 시험 분리", "설정은 학습 기간에만 고르고 시험 기간은 확인용. 시험 결과를 보고 설정을 바꾸지 않음"],
    ["하위 기간 일관성", "2008–16 · 2017– 두 기간 모두 통한 설정만 채택. 표에서 최고값 고르기 금지"], ["위약·대조군 비교", "무작위 신호·자기 가격만 쓴 대조군보다 확실히 나을 때만 효과로 인정"], ["품목 선택 편향 방지", "잘 맞는 품목만 고르지 않고 원자재 19–25개 전체에 같은 규칙 적용. 결과는 기준일로 고정·실행 기록 보관"], ["전체 테스트", "단위 테스트 22개 + 앱 전체 자동 테스트 통과 후 배포"]];
  ps.forEach((p, i) => card(s, 0.6 + (i % 3) * 4.15, 1.4 + Math.floor(i / 3) * 2.75, 3.95, 2.55, p[0], p[1], i < 3 ? G : BLUE, { bs: 12.5 }));
}
{
  const s = base();
  header(s, "A4", "모델 목록 — 대시보드에 들어간 주요 모델");
  const rows = [["분야", "모델", "쓰임"], ["변동성", "EWMA · GARCH · GJR · EGARCH · HAR · MIDAS", "다음 기간 위험 예측, VaR"], ["다변량 위험", "DCC · ADCC · BEKK · DY 스필오버 · TVP-VAR", "상관 동학, 충격·변동성 전이"], ["레짐", "HMM · Markov-switching · TAR · STAR · Kalman TVP", "숨은 국면, 시변 관계"],
    ["꼬리 위험", "정규 · t · 역사적 · FHS · EVT VaR/ES", "극단 손실 추정"], ["예측(통계·ML)", "OLS · Ridge · LASSO · PCA · PLS · XGBoost · LightGBM · LSTM · 패널 · MIDAS", "변수 선별, 예측력 검증"], ["AI 시계열", "Chronos · TimesFM · Prophet · TFT · iTransformer · PatchTST · Neural ODE · GNN", "최신 모델 비교 검증"],
    ["포트폴리오", "평균-분산(Newton) · 최소분산 · 리스크 패리티 · 목표 변동성 · 효율적 투자선", "비중 결정, 위험 한도"], ["전략", "52주 모멘텀 · 가치 · 섹터 따라잡기 · 칼만 추세 · COT", "멀티 전략 백테스트"], ["텍스트", "FinBERT", "리포트 서술 감성 점수"]];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, color: i === 0 ? WHITE : INK, fill: { color: i === 0 ? G : (i % 2 ? WHITE : TINT) } } }))), { x: 0.6, y: 1.35, w: 12.2, colW: [1.9, 6.9, 3.4], fontFace: F, fontSize: 11.5, border: { type: "solid", color: LINE, pt: 0.5 }, rowH: 0.5 });
  foot(s, "각 모델의 원리 그림(2D·3D)은 대시보드 ⑦ 모델 도감 탭.");
}

if (NOTES.length !== page) console.warn("NOTES", NOTES.length, "!= slides", page);
pres.writeFile({ fileName: OUT }).then((f) => console.log("written", f, "slides", page));
