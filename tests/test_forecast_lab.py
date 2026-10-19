"""예측 실험실 검증: Purged K-fold 누수 차단, DSR·PBO의 크기/검정력, 파운데이션 모델 컨텍스트의 미래정보 차단, LightGBM 학습 표본 purge."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import forecast_lab as FL, validation as VD

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")

# 1) Purged K-fold: 학습 관측치의 목표 구간 [i, i+h)가 검증 블록과 겹치지 않고, embargo 구간 제외
n, h, emb = 1000, 20, 10
for tr, te in VD.purged_kfold(n, 5, h, emb):
    a, b = te.min(), te.max() + 1
    assert not np.any((tr + h > a) & (tr < b)), "목표 겹침 누수"
    assert not np.any((tr >= b) & (tr < b + emb)), "embargo 위반"
ok("Purged K-fold: 학습 표본의 목표 기간이 검증 블록과 겹치지 않고 embargo 구간 제외")

# 2) DSR·PBO: 순수 잡음 전략 100개 중 최고는 DSR 낮음·PBO ≈ 0.5, 진짜 신호 1개가 섞이면 PBO 낮음
rng = np.random.default_rng(0)
M = rng.normal(0, 1, (520, 100))
srs = M.mean(0) / M.std(0, ddof=1)
d = VD.deflated_sharpe(M[:, srs.argmax()], srs)
assert d["PSR (SR>0)"] > 0.9 and d["DSR (SR>SR₀)"] < 0.6, d
pb = VD.pbo_cscv(M, 16, max_combos=1500)
assert 0.3 < pb["PBO"] < 0.7, pb
M2 = M.copy(); M2[:, 0] += 0.25
pb2 = VD.pbo_cscv(M2, 16, max_combos=1500)
assert pb2["PBO"] < 0.1, pb2
ok(f"DSR: 잡음 100개 중 최고 PSR {d['PSR (SR>0)']:.2f} → DSR {d['DSR (SR>SR₀)']:.2f} · PBO 잡음 {pb['PBO']:.2f} / 진짜 신호 {pb2['PBO']:.2f}")

# 3) 파운데이션 모델 컨텍스트: 결정일 d의 컨텍스트 마지막 값 = d의 값 (그 이후 데이터 미포함)
seen = []
def dummy(ctx, hh):
    seen.extend([c[-1] for c in ctx]); return np.zeros((len(ctx), hh, 9))
FL._LOADED["__dummy__"] = dummy
idx = pd.bdate_range("2015-01-01", periods=800)
s = pd.Series(np.arange(800, dtype=float), idx)
dd = FL.fridays(idx, "2016-01-01")
FL.fm_forecast("__dummy__", "test", {"x": s}, {"x": dd}, 5, 100)
assert np.allclose(seen, s.reindex(dd).values) and len(seen) == len(dd)
(FL.CACHE / "__dummy___test.npz").unlink(missing_ok=True)
ok(f"파운데이션 모델 입력: {len(dd)}개 결정일 모두 컨텍스트가 결정일 값에서 끝남 (미래 미포함)")

# 4) LightGBM walk-forward: 그해 예측은 '목표가 연초 전에 확정된' 표본만으로 학습 → 연초 이후 y를 바꿔도 그해 예측 불변
k = 1500
dates = pd.date_range("2000-01-07", periods=k, freq="W-FRI")
mi = pd.MultiIndex.from_arrays([["a"] * k, dates], names=["id", "date"])
X = pd.DataFrame(rng.normal(size=(k, 4)), index=mi, columns=list("abcd"))
y = pd.Series(X["a"].values * 0.5 + rng.normal(size=k), index=mi)
end = pd.Series(dates + pd.Timedelta(days=28), index=mi)
p1 = FL.lgb_walkforward(X, y, end, mi, "reg", 20, start="2012-01-01")
y2 = y.copy(); y2[dates >= "2014-01-01"] *= -5
p2 = FL.lgb_walkforward(X, y2, end, mi, "reg", 20, start="2012-01-01")
sel = (dates >= "2012-01-01") & (dates < "2014-01-01")
assert np.allclose(p1[sel].values, p2[sel].values) and p1[sel].notna().all()
assert not np.allclose(p1[dates >= "2015-01-01"].values, p2[dates >= "2015-01-01"].values)
ok("LightGBM walk-forward: 2014년 이후 목표를 바꿔도 2012~13년 예측 불변 (purge: 확정일 < 연초)")
print("FORECAST LAB TEST PASSED")
