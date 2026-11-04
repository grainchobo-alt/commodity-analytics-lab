// 실험 요약 워드 문서 — 그림은 make_figs.py, 숫자는 docs/백테스트_로그.md · data/*_compare.csv
// 실행: set NODE_PATH=docs\발표\scripts\node_modules && node docs\실험요약\build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel,
  WidthType, ShadingType, BorderStyle, LevelFormat, Footer, PageNumber,
} = require("docx");

const HERE = __dirname;
const FONT = "Malgun Gothic";
const INK = "1E3A2F", GREY = "5B6575", RED = "C8553D", GREEN = "2E7D52";
const W = 9638;                                    // A4 본문 폭(여백 1.8cm) DXA

const P = (t, o = {}) => new Paragraph({ spacing: { after: 80 }, ...o.p, children: [].concat(t).map((x) => typeof x === "string" ? new TextRun({ text: x, font: FONT, size: o.size || 20, color: o.color, bold: o.bold }) : x) });
const R = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 20, bold: o.bold, color: o.color, italics: o.italics });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const B = (t, lvl = 0) => new Paragraph({ numbering: { reference: "b", level: lvl }, spacing: { after: 40 }, children: [].concat(t).map((x) => typeof x === "string" ? R(x) : x) });
const IMG = (f, w, h, cap) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(HERE, "figs", f)), transformation: { width: w, height: h } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [R(cap, { size: 17, color: GREY })] }),
];
const border = { style: BorderStyle.SINGLE, size: 4, color: "C9CFC9" };
const cell = (t, w, o = {}) => new TableCell({
  width: { size: w, type: WidthType.DXA }, borders: { top: border, bottom: border, left: border, right: border },
  shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: "auto" } : undefined, margins: { top: 50, bottom: 50, left: 90, right: 90 },
  children: [new Paragraph({ alignment: o.align || AlignmentType.LEFT, children: [R(t, { size: 17, bold: o.bold, color: o.color })] })],
});
function table(head, rows, widths, opt = {}) {
  const sum = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: sum, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: head.map((h, i) => cell(h, widths[i], { fill: "1E3A2F", color: "FFFFFF", bold: true })) })]
      .concat(rows.map((r, k) => new TableRow({ children: r.map((v, i) => cell(String(v), widths[i], { fill: (opt.hl || []).includes(k) ? "E8F3EC" : (k % 2 ? "F7F8F6" : undefined), bold: (opt.hl || []).includes(k), align: i > 0 && opt.num ? AlignmentType.CENTER : AlignmentType.LEFT })) }))),
  });
}
const note = (t) => P(R(t, { size: 16, color: GREY }), { p: { spacing: { before: 40, after: 160 } } });

const children = [
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 60 }, children: [R("3축 혼합 전략 · 위험 신호 백테스트 요약", { size: 36, bold: true, color: INK })] }),
  new Paragraph({ spacing: { after: 200 }, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "C9A04A", space: 4 } },
    children: [R("2026-11-04 · 자료 기준일 2026-10-26 · 상세 기록 docs/백테스트_로그.md", { size: 18, color: GREY })] }),

  H1("1. 결론"),
  B([R("신호로 비중을 바꾸는 방법은 모두 미채택", { bold: true }), R(" — HMM · 칼만 · DCC · 뉴스 · VIX · 상관 신호로 3축 비중을 갈아타거나, 현금으로 피하거나, 목표 변동성을 바꾸거나, 강한 추세에 몰아도 시험 기간(2017–2026) Sharpe가 고정 50/25/25(1.07)를 유의하게 넘지 못함.")]),
  B([R("상관 ↑ + 같이 오르는 특수 구간(2022 곡물 · 원유, 2025–26 금 · 은)", { bold: true }), R(" — 금 · 은은 상승 추세 집중이 잡았지만(+44%) 같은 규칙이 2008–16년엔 손실. 2022년은 곡물 상관이 오히려 낮았고(0.45) 급등이 2–3주 만에 끝나 추세 신호가 늦게 탐.")]),
  B([R("쓸 만한 것은 위험 관리 2가지", { bold: true }), R(" — ① 곡물 합계 비중 상한 25%: 수익 그대로(1.09) · VaR 2.87 → 2.25% · 곡물 VaR 몫 62 → 34% (채택 후보) ② 손실 한도 경보: 비중은 그대로 두고 한도 접근 시 담당자에게 알림(운영 도구).")]),

  H1("2. 무엇을 했나 — 재료 · 신호 · 쓰는 방식 · 검증"),
  ...IMG("f1_flow.png", 610, 355, "그림 1. 실험 전체 구조: 3축 재료에 7가지 신호를 8가지 방식(A–H)으로 써 보고, 같은 검증 틀로 판정"),
  table(["재료 · 신호", "무엇을 보나 (한 줄)"], [
    ["MinVar 롱", "변동성이 작고 서로 덜 엮인 품목을 롱 — 위험을 가장 작게 하는 조합(품목당 상한 25%)"],
    ["52주 모멘텀", "지난 1년 오른 품목 롱 · 내린 품목 숏 — 추세 추종"],
    ["가치 = 평균회귀", "5년 평균보다 싼 품목 롱 · 비싼 품목 숏 — 제자리로 돌아온다는 가정"],
    ["HMM 위기 국면", "원자재 6 + 거시 8로 '지금 위기 국면일 확률'(숨은 국면 추정)"],
    ["칼만 평균 추적", "각 축의 '요즘 평균 수익'을 매주 갱신 — 안 먹히는 축을 빨리 알아챔"],
    ["DCC 동조성 · 공분산", "품목 · 축 간 상관이 시점마다 어떻게 변하는지(다변량 GARCH)"],
    ["흡수비율 · VIX · 변동성", "품목들이 한 요인으로 몰리는 정도 · 시장 공포 · 출렁임 국면"],
    ["뉴스 감성", "연준 베이지북 · USDA WASDE · EIA STEO 문장의 긍정 · 부정(FinBERT)"],
    ["곡물 상관 + 오름 방향", "곡물 5개 26주 상관이 과거 상위권 + 직전 13주 수익 > 0 (함께 오르는 동조)"],
  ], [2400, W - 2400]),
  note("검증 틀(모든 실험 공통): 판단 시점까지 자료만 · 설정은 학습 2008–16에서만 선택 · 시험 2017–26 확인 · 거래 5bp + 비중 변경 비용 · 위약(가짜 신호) · 대조군(늘 같은 비중)"),

  H1("3. 실험별 결과"),
  ...IMG("f2_results.png", 600, 335, "그림 2. 실험별 시험 Sharpe(각 실험의 학습 최고 설정) — 점선 = 기준 1.07"),
  table(["실험", "방법", "결과 (시험 Sharpe · 기준 1.07)", "판정"], [
    ["A 갈아타기", "매년 직전 3 · 5년 최적 비중 · 성과/위기/투자선 이탈 트리거", "0.58–0.89 · 직전 5년 최적점은 다음 해 15번 중 3번만 이김", "미채택"],
    ["B 메타 레이어", "6개 신호로 다음 13주 1등 축 예측 → 기울이기", "1.05 · 1등 축 적중 29%(무작위 33%)", "미채택"],
    ["C 적응형 비중", "칼만 · HMM · 뉴스 · DCC 관점으로 ±25%p", "칼만 0.79(학습만 우세) · HMM 1.08 · 결합 0.85", "미채택"],
    ["D 현금 피신", "신호 9종 켜지면 50 · 100% 현금", "0.63–1.12 · 낙폭 감소는 '덜 투자한 몫'", "미채택"],
    ["E 목표 변동성 동적", "신호 개수로 12 · 10 · 7 · 5% + 국채 이자", "0.96–1.02", "미채택"],
    ["F 상관↑ + 추세 집중", "상관 급등(+ 같이 오름) 때 강한 추세 상위 2–3개 롱", "0.91–1.07 · 위약 p 0.48", "미채택"],
    ["J 짧은 추세(4–8주)", "급등을 빨리 타려고 4 · 8주 추세를 매주", "단독 0.23–0.69 · 회전율 연 20배+", "미채택"],
    ["G · I 품목군 상한", "곡물 · 에너지 · 축산 합계 비중 상한", "곡물 25%: 1.09 · VaR 2.87 → 2.25%", "채택 후보"],
    ["H 손실 한도 경보", "VaR · 역스트레스 · 상관 · 낙폭 등 10항목 경보", "경보 뒤 큰 손실 비율 0–14% → 예측용 아님", "운영 도구"],
  ], [1700, 3000, 3600, W - 8300], { hl: [7, 8] }),

  H1("4. 특수 케이스 — '상관이 높으면서 같이 오르는' 구간 (실험 F · J)"),
  H2("4-1. 요청과 설계"),
  P("요청: 평소처럼 분산하지 말고, 2022–23 우크라이나 전쟁(옥수수 · 밀 · 원유)이나 2025–26 금 · 은 급등처럼 원자재가 함께 오를 때는 모멘텀을 키우고 품목 한도 없이 강한 추세 품목에 집중. 단, 함께 폭락하는 동조에는 하지 않음."),
  ...IMG("f3_f_logic.png", 620, 260, "그림 3. 판단 흐름 — 매주 ① 상관 급등 ② 같이 오르는 중이면 ③ 강한 추세 집중, 아니면 기본 50/25/25"),
  B([R("① 상관 급등: ", { bold: true }), R("곡물 5개 26주 평균 상관의 과거 분위 > 0.8 · 0.9, 또는 원자재 8개 DCC 동조성 분위, 흡수비율 4주 급등 z")]),
  B([R("② 방향 조건: ", { bold: true }), R("직전 13주 곡물(DCC · 흡수비율은 바스켓) 수익 > 0 이면 '상승 동조' — 비교용으로 방향 무관 · 하락 동조도 함께 시험")]),
  B([R("③ 켜지면 7가지 비중 중 학습 최고: ", { bold: true }), R("모멘텀 50% · 강한 추세 5개(롱 · 숏) 집중 · 상승 추세 상위 2 · 3개만 롱(50 · 100%) · MinVar 품목 한도 해제")]),
  B([R("④ 보완(J): ", { bold: true }), R("26 · 52주 추세가 늦다는 결과를 보고 4 · 8 · 13주 짧은 추세를 매주 따라가는 판도 시험")]),
  H2("4-2. 결과"),
  table(["규칙", "학습 → 시험 Sharpe", "최대 낙폭", "2022–23 수익"], [
    ["고정 50/25/25 (기준)", "0.61 → 1.07", "−24.5%", "+29.0%"],
    ["늘 상승 추세 집중 3개 100% (대조군)", "−0.01 → 0.77", "−40.2%", "+14.3%"],
    ["상관 > 0.8 → 모멘텀 집중 100% (학습 최고)", "0.81 → 0.91", "−20.4%", "+22.0%"],
    ["상관 > 0.9 · 상승 동조 → 상승 추세 집중", "0.74 → 1.07", "−20.4%", "+29.0% (2022–23 한 번도 안 켜짐)"],
    ["짧은 추세 8주 · 5개 혼합 50% (J)", "0.26 → 1.01", "–", "2022 1분기 +18% · 2분기 −3%"],
  ], [3900, 1900, 1400, W - 7200], { num: true }),
  ...IMG("f4_cases.png", 620, 375, "그림 4. 위: 가격(시작 = 100) · 아래: 곡물 5개 26주 상관과 신호 문턱(과거 상위 10%)"),
  H2("4-3. 왜 안 됐나"),
  B([R("2022년은 '상관 급등'이 아니었음: ", { bold: true }), R("곡물 상관은 전쟁 직전(2021년 말 약 0.7)에 높았고 전쟁 중엔 0.4대로 내려감(연평균 0.45, 2011–12년 0.7–0.8보다 낮음) → 상관 신호가 켜지지 않음.")]),
  B([R("급등이 너무 짧음: ", { bold: true }), R("밀은 2월 24일–3월 8일 약 2주에 끝났고, 2분기 밀 −24%, 3분기 원유 −30%로 되돌림. 26 · 52주는 물론 4주 추세도 늦게 타고 늦게 내림(4주 판은 대두박 · 대두 · 면화를 들었음).")]),
  B([R("금 · 은은 잡았지만 규칙으로는 못 이김: ", { bold: true }), R("상승 추세 집중이 2025–26년 금 · 은을 들고 +44%(52주 모멘텀 +16%). 그러나 같은 규칙이 2008–16년엔 손실(학습 Sharpe −0.01) — 사후에 맞는 해만 보면 좋아 보이는 전형.")]),
  B([R("기준 혼합도 이미 그 구간을 탔음: ", { bold: true }), R("50/25/25가 2022년 1분기 +20%(MinVar가 곡물 · 축산을 보유), 2022–23년 +29% — 집중판보다 높음.")]),

  H1("5. 쓸 만한 것 — 위험 한도와 경보"),
  ...IMG("f5_graincap.png", 560, 200, "그림 5. 곡물 합계 비중 상한(총노출 대비) — 수익은 그대로, VaR 중 곡물 몫은 62% → 34%"),
  table(["품목군 상한", "학습 → 시험 Sharpe", "최대 낙폭", "지금 20일 VaR", "판정"], [
    ["없음", "0.62 → 1.07", "−24.5%", "2.87%", "기준"],
    ["곡물 25%", "0.62 → 1.09", "−24.5%", "2.25%", "채택 후보"],
    ["에너지 25 · 15%", "0.62 → 1.06–1.07", "−24.5%", "2.87%", "효과 없음(평소 몫 11%)"],
    ["축산 15%", "0.60 → 0.97", "−23.6%", "2.68%", "손해(분산 몫 감소)"],
    ["곡물 · 에너지 · 축산 각 25%", "0.64 → 1.06", "−20.7%", "2.16%", "낙폭 우선이면 검토"],
  ], [2900, 1900, 1300, 1500, W - 7600], { hl: [1], num: true }),
  P([R("손실 한도 경보(risk_alert.py): ", { bold: true }), R("매주 10개 항목(역스트레스 여유 · 20일 VaR · 상관 스트레스 VaR · 곡물 VaR 몫 · 곡물 상관 · DCC 동조성 · 흡수비율 · HMM · 전략 낙폭 · 자료 경과)을 정상/주의/경고로 표시하고 담당자에게 보낼 문안을 만듦. 2026-11-04 점검: 주의 2개(상관 스트레스 VaR 8.6%, 곡물 VaR 몫 62%), 경고 1개(가격 자료 7영업일 경과). 자료 갱신 배치(run_daily.bat) 끝에 연결.")], { p: { spacing: { before: 120, after: 80 } } }),

  H1("6. 한계 · 다음 단계"),
  B("결과는 2008–2026년 주간 자료 한 번의 역사 — 표본이 짧아 Sharpe 차이 0.1 안팎은 구별 불가(Sharpe 표준오차 약 ±0.3)."),
  B("50/25/25 자체도 학습 기간 후보 14개 중 1위로 고른 것 — 동일 비중(1.11)과 함께 비교하는 것이 정직함."),
  B("결정 필요: 곡물 상한 25%(또는 3품목군 25%)를 대시보드 혼합의 위험 한도로 넣을지, 경보 문안을 누구에게 어떤 수단으로 보낼지."),
  note("재현: python strategy_switch.py · strategy_meta.py · strategy_adaptive.py · strategy_cash.py · strategy_voltgt_dyn.py · strategy_crisis_mom.py · strategy_short_trend.py · strategy_graincap.py · strategy_groupcap.py · stress_portfolio.py · risk_alert.py"),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 26, bold: true, color: INK }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 22, bold: true, color: GREEN }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } },
                                                   { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 240 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1000, bottom: 1000, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [R("3축 혼합 전략 백테스트 요약 · ", { size: 16, color: GREY }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: GREY })] })] }) },
    children,
  }],
});
const out = path.join(HERE, "3축_신호_백테스트_요약_20261104.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("saved", out); });
