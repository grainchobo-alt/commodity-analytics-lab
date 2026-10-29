// 발표 자료 v2: 10분 임원 보고판 (본문 13장 + 별첨 1장, 비전공 청중 기준 — 기법 이름은 별첨으로)
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const IMG = (n) => path.join(__dirname, "img", n + ".png");
const OUT = process.argv[2] || path.join(__dirname, "deck_v2.pptx");
const NOTES = require("./notes_v2.js");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "퀀트 기법 기반 곡물·원자재 Market Intelligence (10분 보고)";
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
    ["확장성", "전사", "부서별 원자재 매매를 묶어 관리하는\n리스크 컨트롤 타워로 확장 가능\n→ 주식·채권 등 모든 가격 자산에 같은 엔진", BLUE],
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
  box(0.6, 1.9, "고객사", "Inquiry (물량·시기)", BLUE); box(3.4, 1.9, "당사", "공급가 + 운임 + 마진", G); box(6.2, 1.9, "공급사·선사", "그 시점 시세로 견적", "6E7F74");
  arrow(s, 2.85, 2.4, 3.35, 2.4); arrow(s, 5.65, 2.4, 6.15, 2.4);
  bullets(s, [{ t: "지금의 한계", b: true, c: RED }, "고객도 시세를 실시간으로 앎 → 매번 최저가 공급선 선택", "시세 변동 위험은 그대로 노출, 우리만의 판단 근거 부족"], { x: 0.6, y: 3.2, w: 8.0, h: 1.6, fontSize: 13.5 });
  card(s, 0.6, 4.95, 3.95, 1.85, "단기 · 원가 절감", "주간 '지금 살까, 미룰까' 판단 근거\n시세 위험을 물량 금액으로 환산", BLUE, { bs: 12.5 });
  card(s, 4.7, 4.95, 3.95, 1.85, "중장기 · 독립 수익", "여러 원자재를 묶어 운영하는\n포트폴리오 사업의 실험 기반", GOLD, { bs: 12.5 });
  card(s, 8.95, 1.75, 3.85, 5.05, "발표자 배경", "• 식량사업실 곡물 트레이딩 약 2년\n  — '언제·얼마나 살까'를 현장에서 고민\n\n• 퀀트 석사\n  — 시계열·리스크·포트폴리오 이론\n\n• 팜사업운영그룹\n  — 농장 운영·신규 사업·시황 분석\n\n→ 현장 질문에 정량 답을 주는 도구로 구축", G, { bs: 12 });
}

// ════════════════ 4 데이터 유형
{
  const s = base();
  header(s, "Ⅱ", "데이터는 두 방향으로 읽음 — 시계열과 횡단면", "같은 표라도 '한 품목의 시간 흐름'으로 보느냐, '같은 날 품목끼리 비교'로 보느냐에 따라 질문이 달라짐");
  img(s, "c40_xs_ts", 0.4, 1.65, 7.9, 4.6);
  card(s, 8.6, 1.75, 4.2, 2.35, "시계열 (가로로 읽기)", "한 품목의 가격이 시간에 따라 어떻게 변하나\n• 질문: 다음 달 옥수수는 오를까?\n• 쓰임: 위험 크기 예측, 추세 판단", BLUE, { bs: 12.5 });
  card(s, 8.6, 4.25, 4.2, 2.35, "횡단면 (세로로 읽기)", "같은 시점에 어느 품목이 상대적으로 싼가\n• 질문: 이번 달 무엇부터 살까?\n• 쓰임: 품목 순위, 비슷한 품목 묶기", "9A7A1A", { bs: 12.5 });
  foot(s, "자료: 원자재 12개 월간 수익률(롤 보정), 2025-05 ~ 2026-10(10월은 26일까지).");
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

// ════════════════ 7 활용한 AI·통계 모델 (한 장)
{
  const s = base();
  header(s, "Ⅱ", "활용한 AI·통계 모델 — 한 장으로", "역할별 8개 묶음, 모두 같은 규칙(과거로 학습 → 이후 기간으로 시험)으로 비교해 쓸 것만 남김");
  const ms = [
    ["쓸모 있는 변수 고르기", "수십 개 후보 지표 중 실제로 도움 되는 것만 남기는 필터", "Lasso · Ridge · Elastic Net 등", G],
    ["공통 흐름으로 요약", "20개 품목의 움직임을 몇 개의 큰 흐름으로 압축", "PCA · PLS · 요인 모형", G],
    ["스무고개 나무 묶음", "질문 나무 수백 개의 다수결로 복잡한 관계 포착", "Random Forest · XGBoost · LightGBM", BLUE],
    ["순서를 기억하는 신경망", "과거 패턴의 순서를 기억해 다음 움직임 추정", "LSTM · TFT · PatchTST · iTransformer", BLUE],
    ["사전학습 AI (시계열판 ChatGPT)", "수많은 시계열을 미리 학습 → 추가 학습 없이 바로 예측", "Chronos (아마존) · TimesFM (구글)", RED],
    ["문서를 읽는 AI", "USDA·EIA 보고서, 연준 경기 보고서 문장을 긍정·부정 점수로", "FinBERT (금융 특화 언어 모델)", RED],
    ["위험 크기 예측", "큰 출렁임이 몰려오는 성질로 다음 기간 위험 폭 계산", "GARCH 계열 · MIDAS", GOLD],
    ["시장 국면 판별", "평시·위기를 확률로 구분하고 변하는 관계를 실시간 추적", "국면 전환 모형 · 칼만 필터", GOLD],
  ];
  ms.forEach((m, i) => {
    const x = 0.6 + (i % 4) * 3.1, y = 1.7 + Math.floor(i / 4) * 2.45, w = 2.95, h = 2.3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addShape(pres.shapes.RECTANGLE, { x, y: y + 0.12, w: 0.08, h: 0.5, fill: { color: m[3] }, line: { color: m[3] } });
    s.addText(m[0], { x: x + 0.2, y: y + 0.1, w: w - 0.3, h: 0.55, fontFace: F, fontSize: 13.5, bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText(m[1], { x: x + 0.2, y: y + 0.72, w: w - 0.35, h: 0.95, fontFace: F, fontSize: 11.5, color: INK, margin: 0, valign: "top" });
    s.addText(m[2], { x: x + 0.2, y: y + 1.7, w: w - 0.35, h: 0.5, fontFace: F, fontSize: 10, color: MUTED, margin: 0, valign: "top" });
  });
  bullets(s, ["결론: 최신 AI도 가격 '방향'을 꾸준히 맞히지는 못함(앞 장의 이유) → 위험 계산·국면 판별·보조 신호로 활용하고, 시험 기간에 개선이 없는 모델은 채택하지 않음"], { x: 0.6, y: 6.6, w: 12.2, h: 0.4, fontSize: 11.5 });
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
  st.forEach((c, i) => {
    const x = 0.6 + i * 4.15, w = 3.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.7, w, h: 3.85, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.7, w, h: 0.12, fill: { color: c[5] }, line: { color: c[5] } });
    s.addText(c[0], { x: x + 0.2, y: 1.95, w: w - 0.4, h: 0.45, fontFace: F, fontSize: 18, bold: true, color: c[5], margin: 0 });
    s.addText(c[1], { x: x + 0.2, y: 2.4, w: w - 0.4, h: 0.4, fontFace: F, fontSize: 13.5, bold: true, color: INK, margin: 0 });
    s.addText([{ text: "규칙", options: { bold: true, color: GOLD, breakLine: true } }, { text: c[2], options: { breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 5 } }, { text: "직관", options: { bold: true, color: GOLD, breakLine: true } }, { text: c[3] }],
      { x: x + 0.2, y: 2.9, w: w - 0.35, h: 2.0, fontFace: F, fontSize: 11.5, color: INK, margin: 0, valign: "top" });
    s.addText([{ text: `Sharpe ${c[4]}`, options: { bold: true, fontSize: 16, color: c[5] } }, { text: "  (단독, 2007–2026)", options: { fontSize: 10, color: MUTED } }], { x: x + 0.2, y: 4.95, w: w - 0.4, h: 0.45, fontFace: F, margin: 0 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.75, w: 12.2, h: 1.1, fill: { color: "FBF4E1" }, line: { color: GOLD, width: 1 }, rectRadius: 0.08 });
  s.addText([{ text: `세 전략을 섞으면 Sharpe ${SH["가치 혼합"]}   `, options: { bold: true, color: RED, fontSize: 16 } }, { text: "최소분산 50% + 모멘텀 25% + 가치 25% — 각 전략 단독(0.4~0.6)보다 위험 대비 성과가 크게 개선 (한 전략이 부진할 때 다른 전략이 보완)", options: { fontSize: 12.5, color: INK } }],
    { x: 0.85, y: 5.75, w: 11.8, h: 1.1, fontFace: F, valign: "middle", margin: 0 });
  foot(s, "Sharpe = 연 수익 ÷ 연 변동성(위험 1단위당 수익, 1에 가까우면 우수). 각 목표 변동성 10%, 편도 거래비용 5bp 차감, 연속월물 기준.");
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

// ════════════════ 12 확장 비전
{
  const s = base();
  header(s, "Ⅴ", "확장 비전 — 전사 원자재 리스크 컨트롤 타워", "금융 트레이딩을 하지 않더라도, 부서별로 나뉜 원자재 매매를 통합해 회사 전체의 가격 변동 위험을 줄일 수 있음");
  const depts = [["곡물", "옥수수·밀·대두·대두박"], ["팜유", "CPO·PKO (팜 농장 법인)"], ["에너지", "연료·운임 연동"], ["기타 원자재", "금속·소프트 등"]];
  depts.forEach((d, i) => { const y = 1.8 + i * 1.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 2.4, h: 0.85, fill: { color: TINT }, line: { color: LINE }, rectRadius: 0.08 });
    s.addText([{ text: d[0] + " 부서", options: { bold: true, fontSize: 13, color: INK, breakLine: true } }, { text: d[1], options: { fontSize: 10, color: MUTED } }], { x: 0.6, y, w: 2.4, h: 0.85, fontFace: F, align: "center", valign: "middle", margin: 0 });
    arrow(s, 3.05, y + 0.42, 3.75, 3.85, MUTED); });
  s.addShape(pres.shapes.OVAL, { x: 3.8, y: 2.6, w: 2.5, h: 2.5, fill: { color: G }, line: { color: G } });
  s.addText("리스크\n컨트롤 타워", { x: 3.8, y: 2.6, w: 2.5, h: 2.5, fontFace: F, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  const outs = ["전사 포지션 합산 · 순 위험 금액", "품목끼리 상쇄되는 자연 헤지 식별", "목표 위험 기준 한도 배분", "구매 시점·물량의 통합 의사결정"];
  outs.forEach((o, i) => { const y = 1.95 + i * 0.95; arrow(s, 6.35, 3.85, 6.85, y + 0.32, GOLD);
    s.addText(o, { x: 6.9, y, w: 2.6, h: 0.65, fontFace: F, fontSize: 11.5, bold: true, color: G, valign: "middle", margin: 0 }); });
  card(s, 9.7, 1.75, 3.1, 2.45, "학계·헤지펀드의 표준 방법", "• 포트폴리오 이론 — 노벨경제학상(1990)\n• 위험 크기 예측 — 노벨경제학상(2003)\n• 추세·가치 전략 — 대형 헤지펀드의 핵심 운용 방식", BLUE, { hs: 13, bs: 11 });
  card(s, 9.7, 4.35, 3.1, 2.4, "모든 가격 자산으로 확장", "원자재뿐 아니라 주식·채권·환율 등 가격이 있는 모든 자산에 같은 엔진(위험 계산·전략·검증)을 그대로 적용 가능", GOLD, { hs: 13, bs: 11.5 });
  bullets(s, ["부서별로는 작아 보이는 위험도, 합치면 함께 움직여 커지거나 서로 상쇄됨 — 통합해서 봐야 실제 위험이 보임"], { x: 0.6, y: 6.15, w: 8.9, h: 0.75, fontSize: 12 });
  foot(s, "부서 구분은 개념 예시.");
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
    ["무작위 걸음 / 시뮬레이션", "랜덤워크·기하 브라운 운동, 몬테카를로 400회", "본문 5장"],
    ["위험 1단위당 수익", "Sharpe = 연 수익 ÷ 연 변동성", "⑤ 포트폴리오"],
    ["과거로 배우고 이후로 시험", "학습/시험 분리, look-ahead bias 차단, 위약 비교", "⑧ 방법론"]];
  table(s, rows, { x: 0.6, y: 1.75, w: 12.2, colW: [3.6, 5.8, 2.8], rowH: 0.5, fontSize: 11.5, boldFirst: true });
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("written", f, "slides", page));
if (NOTES.length !== page) console.warn(`발표자 노트 ${NOTES.length}개 ≠ 장 수 ${page}`);
