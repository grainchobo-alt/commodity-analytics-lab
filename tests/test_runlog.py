"""실행 기록(runlog) 검사: 기록 → 목록 → 판정(마지막이 유효) → 상세(표 복원) → 설정 지문·차이. 임시 폴더(RUNLOG_DIR)에서 실행."""
import os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
tmp = tempfile.mkdtemp(); os.environ["RUNLOG_DIR"] = tmp
import numpy as np, pandas as pd
import runlog as RL

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")

assert RL.load_registry().empty
cfg = dict(대상="yf:ZC=F", 설명변수=["yf:ZS=F"], 모델=["OLS", "Ridge"], 비용bp=5.0, hp=dict(hmm_states=0))
M = pd.DataFrame({"부호전략 Sharpe": [0.5, np.nan]}, index=["OLS", "Ridge"])
cum = pd.DataFrame({"OLS": [0.0, 1.0, 2.0]}, index=pd.date_range("2020-01-03", periods=3, freq="W-FRI"))
a = RL.record("ml", "옥수수 테스트", cfg, {"WF Sharpe(비용 후)": np.float64(0.42), "WF 최고 모델": "OLS"}, {"metrics": M, "walk_forward_cum": cum}, ["힌트 1"])
time.sleep(1.1)
cfg2 = dict(cfg, 비용bp=10.0)
b = RL.record("ml", "옥수수 비용 10bp", cfg2, {"WF Sharpe(비용 후)": 0.1})
c = RL.record("ml", "옥수수 재실행", cfg, {"WF Sharpe(비용 후)": 0.40})
R = RL.load_registry()
assert len(R) == 3 and set(R.verdict) == {"pending"} and R.iloc[0].run_id in (b, c)
assert RL.config_hash(cfg) != RL.config_hash(cfg2) and R.set_index("run_id").loc[a, "same_config_runs"] == 2
ok("기록 3건 · 최신 먼저 · 같은 설정 지문 묶음(2건)")

RL.set_verdict(a, "bad", "처음 판정"); RL.set_verdict(a, "good", "다시 보니 좋음")
R = RL.load_registry().set_index("run_id")
assert R.loc[a, "verdict"] == "good" and R.loc[a, "note"] == "다시 보니 좋음" and R.loc[b, "verdict"] == "pending"
ok("판정: 추가 기록 중 마지막 줄이 유효")

D = RL.load_run(a)
assert D["settings"]["비용bp"] == 5.0 and D["hints"] == ["힌트 1"] and D["verdict"] == "good"
assert np.isclose(D["tables"]["metrics"].loc["OLS", "부호전략 Sharpe"], 0.5) and np.isnan(D["tables"]["metrics"].loc["Ridge", "부호전략 Sharpe"])
assert len(D["tables"]["walk_forward_cum"]) == 3
diff = RL.settings_diff(D["settings"], RL.load_run(b)["settings"])
assert list(diff.index) == ["비용bp"]
assert (Path(tmp) / "registry.jsonl").exists() and (Path(tmp) / "verdicts.jsonl").exists()
ok("상세: 설정·힌트·표 복원, 설정 차이 = 비용bp 하나")
print("RUNLOG TEST PASSED")
