"""손실 한도 경보(운영용) — 위험 신호로 비중을 바꾸지 않고, 한도에 가까워지면 담당자에게 알릴 '경보'만 낸다.

매주 점검(지금 50/25/25 품목 순비중 · 자본 1천만 달러 가정, 한도는 LIMITS에서 조정)
  ① 역스트레스 여유: 자본 10% 손실에 필요한 '전 품목 동반 이동' 폭 — 작을수록 위험
  ② 20일 99% VaR(FHS) · ③ 상관 스트레스 VaR(품목 간 상관 0.9) · ④ 곡물 VaR 기여 비중
  ⑤ 상관 급등: 곡물 상관 분위 · DCC 동조성 분위 · 흡수비율 4주 z  ⑥ 위기 국면: HMM 위기 확률
  ⑦ 전략 낙폭: 50/25/25 혼합 고점 대비  ⑧ 자료 기준일: 가격 자료가 오래됐으면 경보 자체를 믿지 말 것
단계: 정상 · 주의 · 경고. 결과 = data/risk_alert_latest.md(담당자에게 보낼 문안) · data/risk_alerts.csv(매 실행 누적 기록).
  ※ 메일 · 메신저 자동 발송은 외부 서비스 연결 · 승인이 필요해 넣지 않음 — 문안 파일을 그대로 붙여 보내면 됨.
점검(과거 보정): 신호로 계산 가능한 ⑤ ⑥ ⑦을 2008–2026 매주 다시 계산해, 경보가 켜진 주 비율과 켜진 뒤 4주 혼합 수익(특히 하위 10% 손실이 얼마나 몰렸나)을 봄
  → 경보가 '수익 개선'은 못 해도 '큰 손실 전에 주의를 주는가'를 확인 (data/risk_alert_calibration.csv).
실행: .venv\\Scripts\\python.exe risk_alert.py [자본 $]
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import risk as RK  # noqa: E402
import strategy_cash as SC  # noqa: E402
import strategy_crisis_mom as CM  # noqa: E402
import strategy_meta as SM  # noqa: E402
import strategy_switch as SS  # noqa: E402
import stress_portfolio as SP  # noqa: E402

# (주의, 경고) — 값이 이 쪽으로 넘어가면 단계 상승. dir: '<' = 작을수록 위험
LIMITS = {"역스트레스 여유 (전 품목 동반 이동 %)": (12, 8, "<"), "20일 99% VaR (% 자본)": (3.0, 4.0, ">"),
          "상관 스트레스 VaR (% 자본)": (8.0, 10.0, ">"), "곡물 VaR 기여 (%)": (50, 65, ">"),
          "곡물 상관 분위": (0.9, 0.97, ">"), "DCC 동조성 분위": (0.9, 0.97, ">"), "흡수비율 4주 z": (1.5, 2.5, ">"),
          "HMM 위기 확률": (0.5, 0.8, ">"), "전략 낙폭 (%)": (-10, -15, "<"), "가격 자료 경과 (영업일)": (3, 6, ">")}


def level(name: str, v: float) -> str:
    a, b, d = LIMITS[name]
    if not np.isfinite(v):
        return "확인 불가"
    bad = (lambda x, t: x < t) if d == "<" else (lambda x, t: x > t)
    return "경고" if bad(v, b) else ("주의" if bad(v, a) else "정상")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 1e7
    L = A.load_levels(roll_adjusted=True)
    asof = L["yf:ZC=F"].dropna().index[-1]
    w, _, k = SP.current_weights(L)
    expo = w * cap; ids = list(w.index)
    Rd = pd.DataFrame({i: RK.returns(L[i]) for i in ids}).dropna(how="all")
    Sd = pd.DataFrame({i: pd.Series(RK.ewma_path(Rd[i].dropna().values, RK.RM_LAMBDA), index=Rd[i].dropna().index) for i in ids})
    Rd, Sd = Rd.loc[Sd.dropna().index], Sd.dropna()
    o = RK.position_risk(Rd, Sd, expo, 0.99, 20)
    s_now = np.sqrt(Sd.iloc[-1].values) / 100
    Cs = np.full((len(ids), len(ids)), 0.9); np.fill_diagonal(Cs, 1)
    cvar = float(np.sqrt(expo.values @ (np.outer(s_now, s_now) * Cs * 20) @ expo.values) * 2.326)
    t = o["table"]; gshare = float(t.loc[[i for i in ids if i in CM.GRAINS], "VaR 기여 ($, 정규)"].sum() / t["VaR 기여 ($, 정규)"].sum() * 100)
    # 신호 (과거 전체 — 보정에도 씀). 주간 신호는 자료가 다 찬 마지막 금요일까지만 — 덜 찬 주는 빠진 품목 때문에 흡수비율 등이 튐
    Lw = L if asof.weekday() == 4 else L.loc[:asof - pd.offsets.Week(weekday=4)]
    S, R, P = SM.sleeves(Lw); idx = S.index
    sig = SC.risk_signals(Lw, R, idx)
    gc = CM.grain_corr_pct(R, idx)
    mix = SS.run_weights(S, SS.const(S, dict(zip(SM.AXES, SM.BASE["고정 50/25/25"]))))
    cum = (1 + mix / 100).cumprod(); dd = (cum / cum.cummax() - 1) * 100
    lag = int(np.busday_count(asof.date(), datetime.now().date()))
    vals = {"역스트레스 여유 (전 품목 동반 이동 %)": 10 / abs(w.sum()) if abs(w.sum()) > 1e-6 else np.inf,
            "20일 99% VaR (% 자본)": o["var_fhs"] / cap * 100, "상관 스트레스 VaR (% 자본)": cvar / cap * 100, "곡물 VaR 기여 (%)": gshare,
            "곡물 상관 분위": float(gc.iloc[-1]), "DCC 동조성 분위": float(sig["동조성"].iloc[-1]), "흡수비율 4주 z": float(sig["흡수비율"].iloc[-1]),
            "HMM 위기 확률": float(sig["HMM"].iloc[-1]), "전략 낙폭 (%)": float(dd.iloc[-1]), "가격 자료 경과 (영업일)": float(lag)}
    rows = [{"점검일": datetime.now().strftime("%Y-%m-%d"), "자료 기준일": f"{asof:%Y-%m-%d}", "항목": n, "값": round(v, 3),
             "주의": LIMITS[n][0], "경고": LIMITS[n][1], "단계": level(n, v)} for n, v in vals.items()]
    T = pd.DataFrame(rows)
    f = A.DATA / "risk_alerts.csv"
    pd.concat([pd.read_csv(f, encoding="utf-8-sig"), T] if f.exists() else [T]).to_csv(f, index=False, encoding="utf-8-sig")
    worst = "경고" if (T["단계"] == "경고").any() else ("주의" if (T["단계"] == "주의").any() else "정상")
    md = [f"# 손실 한도 경보 — {worst}", "", f"점검 {datetime.now():%Y-%m-%d} · 자료 기준일 {asof:%Y-%m-%d} · 대상 50/25/25 혼합 · 자본 ${cap:,.0f} 가정", "",
          "| 항목 | 값 | 주의 | 경고 | 단계 |", "|---|---|---|---|---|"]
    md += [f"| {r['항목']} | {r['값']:,.2f} | {r['주의']} | {r['경고']} | **{r['단계']}** |" if r["단계"] != "정상" else
           f"| {r['항목']} | {r['값']:,.2f} | {r['주의']} | {r['경고']} | {r['단계']} |" for _, r in T.iterrows()]
    md += ["", "※ 경보는 비중을 자동으로 바꾸지 않음 — 담당자가 확인 후 판단(한도 조정 · 헤지 · 목표 변동성 하향 등). 백테스트에서 신호로 비중을 바꾸는 규칙은 모두 미채택(docs/백테스트_로그.md)."]
    (A.DATA / "risk_alert_latest.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    # 과거 보정: 경보가 켜진 주 → 다음 4주 혼합 수익
    fwd = mix[::-1].rolling(4).sum()[::-1].shift(-1).loc["2008":]
    bad10 = fwd < fwd.quantile(0.1)
    hist = {"곡물 상관 분위": gc, "DCC 동조성 분위": sig["동조성"], "흡수비율 4주 z": sig["흡수비율"], "HMM 위기 확률": sig["HMM"], "전략 낙폭 (%)": dd}
    cal = []
    for n, s in hist.items():
        lv = s.reindex(fwd.index).apply(lambda v: level(n, v))
        for stage in ("주의 이상", "경고"):
            on = lv.isin(["주의", "경고"]) if stage == "주의 이상" else lv.eq("경고")
            if on.sum() < 5:
                continue
            cal.append({"항목": n, "단계": stage, "켜진 주 (%)": on.mean() * 100, "켜진 뒤 4주 평균 (%)": fwd[on].mean(), "꺼진 뒤 4주 평균 (%)": fwd[~on].mean(),
                        "켜진 뒤 4주 하위 10% 손실 비율 (%)": bad10[on].mean() * 100, "기준 (무작위 10%)": 10.0})
    C = pd.DataFrame(cal); C.to_csv(A.DATA / "risk_alert_calibration.csv", index=False, encoding="utf-8-sig")
    print("\n과거 보정 (2008–, 주간):"); print(C.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
