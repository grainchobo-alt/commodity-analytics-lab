"""② 전문가 의견 백테스트 검증: 진입 시점(미래정보 없음) · 완벽한 의견은 채택 · 무작위 의견은 미채택 · 표본 부족 처리 · 그래프 ② 계층 연결."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from llm import expert_bt as EB, tools as T

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")

p = T.levels()["yf:ZC=F"].dropna(); idx = p.index
d = idx[3000]                                                                     # 거래일
V = pd.DataFrame({"avail": [d, d], "id": "yf:ZC=F", "direction": 1.0, "confidence": 0.5, "horizon_weeks": 4, "same_close": [True, False], "source": "t"})
R = EB.attach_returns(V).sort_values("same_close", ascending=False)
assert R.entry.iloc[0] == d and R.entry.iloc[1] == idx[3001] and (R.exit.values == np.array([idx[3020], idx[3021]])).all()
assert abs(R.ret_pct.iloc[0] - 100 * np.log(p.iloc[3020] / p.iloc[3000])) < 1e-9
ok("진입: 발표 전 종가 가능 → 그날 종가, 아니면 다음 거래일 종가 · 청산 = +20거래일")

k = np.arange(200, len(idx) - 30, 25)
fut = np.sign(np.log(p.values[k + 20] / p.values[k]))
perfect = pd.DataFrame({"avail": idx[k], "id": "yf:ZC=F", "direction": fut, "confidence": 1.0, "horizon_weeks": 4, "same_close": True, "source": "t"})
e = EB.evaluate(EB.attach_returns(perfect[perfect.direction != 0]))
assert e["hit_rate"] == 1.0 and e["verdict"] == "채택" and e["placebo_p"] < 0.01
Rp = EB.attach_returns(perfect)
fp = [EB.evaluate(Rp.assign(direction=np.random.default_rng(s).choice([-1.0, 1.0], len(Rp))), n_placebo=200)["verdict"] == "채택" for s in range(40)]
assert np.mean(fp) < 0.10, np.mean(fp)                                             # 무작위 의견을 잘못 채택하는 비율
assert EB.evaluate(EB.attach_returns(perfect.iloc[:10]))["verdict"].startswith("표본 부족")
ok(f"판정: 미래를 아는 의견 = 채택(적중 100%), 무작위 의견 40개 중 잘못 채택 {sum(fp)}개({100 * np.mean(fp):.1f}%), 10건 = 표본 부족")

U, C = EB.views_usda(), EB.views_cot()
assert (U.direction != 0).all() and U.same_close.all() and not C.same_close.any() and (C.avail.dt.dayofweek == 4).mean() > 0.95
ok(f"의견원: USDA 전망 수정 {len(U)}건(발표일 종가 진입) · CFTC {len(C)}건(금요일 공표 → 다음 거래일)")

o = T.expert_signals("대두박")
assert set(o["sources"]) == {"saved", "usda", "cot"} and o["sources"]["usda"]["backtest"]["n"] >= 20
ok(f"도구 expert_signals: 대두박 USDA {o['sources']['usda']['backtest']['verdict']} · CFTC {o['sources']['cot']['backtest']['verdict']}")
print(f"\n모두 통과 ({time.time()-t0:.0f}s)")
