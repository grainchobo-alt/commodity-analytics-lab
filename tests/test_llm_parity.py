"""LLM 도구 ↔ 대시보드 숫자 대조: tools.DASH에 '같음'으로 적힌 도구는 대시보드 탭이 실제로 부르는 함수·기본 위젯값을 여기서 따로 재현해 값이 같은지 확인.
도구 쪽 기본값이 바뀌거나 대시보드 기본값이 바뀌면 여기서 깨짐 → LLM 답변 숫자와 대시보드 화면 숫자가 어긋나는 일을 미리 잡음.
(app.py는 Streamlit이라 import 불가 → 탭 코드의 호출을 그대로 옮겨 적음. 출처 줄은 각 항목 주석)
실행: .venv\\Scripts\\python.exe tests\\test_llm_parity.py   (약 2~3분, BEKK 포함)"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import analytics as A, procurement as PR, risk as RK
from llm import tools as T

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
def close(a, b, tol=1e-6): return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))

# 대시보드 공통 입력 (app.py 115–140: 롤 보정 ROLL=True, 사이드바 빈도 '주간', 시작 2000-01-01, 끝 = 오늘)
cat = A.load_catalog(); UNITS = dict(zip(cat.id, cat.unit))
La, Lr = A.load_levels(roll_adjusted=True), A.load_levels(roll_adjusted=False)
FQ, ANN, START, END = A.FREQ["주간"], A.ANN[A.FREQ["주간"]], pd.Timestamp("2000-01-01").date(), pd.Timestamp.today().date()
panel = lambda ids: A.make_panel(La, list(ids), UNITS, FQ, "logret", START, END)   # noqa: E731  (app.py _panel · panel_or_stop)
checked = set()

# 0) 대조표가 모든 도구를 덮는지
names = {f.__name__ for f in T.READ_TOOLS + T.WRITE_TOOLS}
assert names == set(T.DASH), names ^ set(T.DASH)
ok(f"대조표 DASH = 도구 {len(names)}개 전부 (같음 {sum(v[1].startswith('같') for v in T.DASH.values())}개)")

# ★ 구매 의사결정 (app.py tab_procure 3236–3300: 프리셋 '추천', 가격 위치 창 '품목별 자동', inputs → signals → score)
P0 = PR.PRESETS[PR.BASIC_PRESETS[0]]
o = T.buy_signal(["옥수수", "밀", "대두", "대두박"])
for row in o["table"]:
    i = row["id"] if "id" in row else T.resolve(row["품목"])
    inp = PR.inputs(La, Lr, i, P0["risk"], "price")
    sig = PR.signals(La[i], PR.item_lookback(i, int(P0["lookback"])), inp["rel"], inp["rel_sign"], inp["vol_fc"], inp["crisis"], P0["season"])
    sc = PR.score(sig, P0["w"]); last = sig.dropna(subset=["value"]).index.max()
    b_lo, b_hi = PR.band_4w(La[i], sig, last); px = float(PR.usd_t(Lr, i).loc[:last].iloc[-1])
    assert close(row["매수 점수"], sc.loc[last] * 100, 1e-3), (row["품목"], row["매수 점수"], sc.loc[last] * 100)
    assert close(row["4주 하단 90%"], px * b_lo, 1e-3) and close(row["4주 상단 90%"], px * b_hi, 1e-3)
checked.add("buy_signal")
ok("★ 매수 점수 · 4주 범위 = 대시보드 ★ (" + ", ".join(f"{r['품목']} {r['매수 점수']:.1f}" for r in o["table"]) + ")")

f1 = T.fundamentals(["옥수수", "대두박"])["table"]; F = PR.fundamentals(["yf:ZC=F", "yf:ZM=F"])
F = F.reset_index() if "품목" not in F.columns else F
assert [r.get("재고/사용비율 (%)") for r in f1] == [None if pd.isna(v) else round(float(v), 3) for v in F["재고/사용비율 (%)"]]
checked.add("fundamentals"); ok("★ 펀더멘털 맥락 = procurement.fundamentals")

# ◆ 리스크 VaR · ES (app.py tab_risk 3690–3708: _risk_vm → var_forecasts(σ 'GARCH(1,1)', 평가 2010) → var_now(next_var))
o = T.var_es("옥수수", 0.99, 20, "short", 1e7)
vm = RK.vol_models(RK.returns(La["yf:ZC=F"]))
V = RK.var_forecasts(vm["r"], vm["S"]["GARCH(1,1)"], 0.99, "short", 20, start="2010-01-01")
now = RK.var_now(V, vm["r"], RK.next_var(vm, "GARCH(1,1)"))
for row in o["now"]:
    assert close(row["VaR (%)"], now.loc[row["방법"], "VaR (%)"], 1e-3) and close(row["ES (%)"], now.loc[row["방법"], "ES (%)"], 1e-3), row
bt = RK.backtest_table(V)
assert all(close(r["초과율"], bt.loc[r["방법"], "초과율"], 1e-3) for r in o["backtest"])
checked.add("var_es"); ok(f"◆ 옥수수 20일 99% VaR(FHS) {next(r for r in o['now'] if r['방법'].startswith('FHS'))['VaR (%)']}% = 대시보드 ◆ (5방법 · 초과율)")

# ③ 변동성 동학 (app.py tab_vol 409–422: panel_or_stop([gid]) → garch_fit(r, 'GJR', 't', ANN))
o = T.vol_forecast("옥수수", "GJR")
pt, vol, info, _ = A.garch_fit(panel(["yf:ZC=F"])["yf:ZC=F"], "GJR", "t", ANN)
assert close(o["vol_now_ann_pct"], vol.iloc[-1], 1e-3) and close(o["persistence"], info["persistence"], 1e-3)
checked.add("vol_forecast"); ok(f"③ 옥수수 GJR 현재 연 변동성 {o['vol_now_ann_pct']}% = 대시보드 ③ 변동성 동학")

# ③ DCC (app.py tab_mvgarch 1253–1289: 기본 품목 8개, GJR-t, DCC)
import mvgarch as MG
ids = [i for i in ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "ak:macro_shipping_bdi"] if i in UNITS]
o = T.dcc_correlation()
dc = MG.dcc_fit(panel(ids), False, "GJR", "t", ANN)
assert close(o["avg_corr_now"], dc["avg_corr"].iloc[-1], 1e-3) and close(o["persistence"], dc["persistence"], 1e-3)
pr = {T._pair_key(a, b): s.iloc[-1] for (a, b), s in dc["pairs"].items()}
assert all(close(x["현재"], pr[x["쌍"]], 1e-3) for x in o["pairs"])
checked.add("dcc_correlation"); ok(f"③ DCC 평균 상관 {o['avg_corr_now']} · 쌍 {len(o['pairs'])}개 = 대시보드 ③ DCC")

# ③ BEKK (app.py 1330–1360: Full BEKK → spillover_table). 시간 절약을 위해 2품목으로 양쪽 같게
o = T.bekk_spillover(["옥수수", "대두"])
bk = MG.bekk_fit(panel(["yf:ZC=F", "yf:ZS=F"])[["yf:ZC=F", "yf:ZS=F"]], "Full", ANN)
sp = MG.spillover_table(bk, {i: T.name_of(i) for i in ["yf:ZC=F", "yf:ZS=F"]})
assert [round(x["계수"], 3) for x in o["spillover"]] == [round(float(v), 3) for v in sp["계수"]] and close(o["loglik"], bk["loglik"], 1e-3)
checked.add("bekk_spillover"); ok("③ BEKK 전이 계수 = 대시보드 ③ BEKK (옥수수·대두)")

# ② 레짐 → Kalman TVP, 분석 목적 '설명' (app.py _rv_panel 487 · _rv_reg 857–862: build_design(X, tgt, others, md, 1, 1, True, 0.7))
import regime_mv as RM
X = RM.build_panel(La, RM.COMM_W_DEFAULT, RM.MACRO_DEFAULT, "W")
others = [c for c in X.columns if c != "yf:ZC=F"]
hp_ = dict(thr_var="fred:VIXCLS", ms_k=3, ms_pc=3, tar_boot=199, hmm_states=3, hmm_cov="full", kalman_q=1e-4, mode="explain")
k = A.run_model("Kalman TVP", A.build_design(X, "yf:ZC=F", others, "explain", 1, 1, True, 0.7), RM.ANN["W"], **hp_)
o = T.kalman_beta("옥수수")
assert len(o["betas"]) == len(k["coef"]) and all(close(b["현재 β(필터)"], v, 1e-3) for b, v in zip(o["betas"], k["coef"].reindex(k["coef"].abs().sort_values(ascending=False).index)))
checked.add("kalman_beta"); ok(f"② 칼만 β {len(o['betas'])}개 = 대시보드 ② Kalman TVP (설명)")

# ④ 수익률 예측, 분석 목적 '예측' (app.py tab_ml 1724–1846: 기본 설명변수 · 시차 3 · h 1 · 자기시차 · 학습 0.7 · 비용 5bp · 하이퍼파라미터 위젯 기본값)
tgt = "yf:ZC=F"
feats = [i for i in ["yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "ak:macro_shipping_bdi",
                     "yf:DX-Y.NYB", "fred:VIXCLS", "fred:DGS10", "fred:DTB3", "fred:BAA10Y", "yf:^GSPC"] if i in UNITS and i != tgt]
hp = dict(l1_ratio=[0.1, 0.5, 0.9], ada_gamma=1.0, scad_gamma=3.7, mcp_gamma=3.0, tau=0.05, kalman_q=10 ** -4.0, n_trees=300, max_depth=4,
          seq=20, hidden=32, epochs=40, hmm_states=0, hmm_cov="full", k_max=10, csr_kmax=8, csr_M=2000, bma_iters=20000, hs_iters=2000, mn_theta=0.5,
          mode="predict", thr_var="fred:VIXCLS", ms_k=2, ms_pc=3, tar_boot=199, lgb_leaves=7, lgb_lr=0.03, lgb_l2=5.0, gp_nmax=1500)
D = A.build_design(panel([tgt] + feats), tgt, feats, "predict", 3, 1, True, 0.7); D.cost = 5.0 / 100.0; D.band = 0.0
o = T.ml_models("옥수수", ["Ridge", "Lasso", "LightGBM"])
for row in o["table"]:
    m = A.run_model(row["모델"], D, ANN, **hp)["metrics"]
    assert close(row["OOS R² (vs 학습평균)"], m["OOS R² (vs 학습평균)"], 1e-3) and close(row["방향 적중률"], m["방향 적중률"], 1e-3), row
checked.add("ml_models"); ok("④ Ridge · Lasso · LightGBM 시험 OOS R² · 방향 적중률 = 대시보드 ④ (예측)")

# 롱숏 · Portfolio Customization 비중 규칙 (ls_extra.item_contrib = 롱숏 탭 품목별 손익): 마지막 주 손익 = 전주 비중 × 이번 주 수익률
import ls_extra as LX, ls_strategies as LS
Pw = LS.weekly(La, LS.BASKET, "2001-01-01"); last_wk = Pw.index[-1]
PnL = LX.item_contrib(La)
bak = T.levels
try:
    T.levels = lambda adjusted=True: (La if adjusted else Lr).loc[:last_wk - pd.Timedelta(days=7)]   # 한 주 전까지만 보이게 → 그때의 비중
    w_prev = T.strategy_weights(top=50)["weights_pct"]
finally:
    T.levels = bak
Rw = LS._ret(Pw[[c for c in LS.BASKET if c in Pw]]).iloc[-1].fillna(0).rename(index=T.name_of)   # 손익 합(.sum)은 빈 값을 건너뜀
for k_tool, k_dash in [("MinVar 롱", "MinVar 롱"), ("52주 모멘텀", "52주 모멘텀")]:
    est = sum(v / 100 * Rw[n] for n, v in w_prev[k_tool].items())
    assert abs(est - PnL[k_dash].iloc[-1].sum()) < 0.02, (k_tool, est, PnL[k_dash].iloc[-1].sum())
checked.add("strategy_weights"); ok("롱숏 탭 비중 규칙: 도구의 전주 비중 × 이번 주 수익률 = item_contrib 손익 (MinVar · 모멘텀)")

# 사이드바 데이터 기준일 (app.py 174: yf: 품목 end의 최댓값) · 자료 목록 · 가격 · 실험 결과 · 문서 감성
o = T.data_status()
assert o["table"][0]["마지막 날짜"] == f"{pd.to_datetime(cat.loc[cat.id.str.startswith('yf:'), 'end']).max():%Y-%m-%d}"
assert T.list_items()["n"] == len(cat)
assert close(T.get_prices("옥수수")["stats"]["현재"], Lr["yf:ZC=F"].dropna().iloc[-1], 1e-3)
g = T.get_results("voltgt"); assert g["rows"] == len(pd.read_csv(T.RESULT_FILES["voltgt"], encoding="utf-8-sig"))
S = pd.read_parquet(T.DATA / "mm" / "steo_text.parquet"); v = S.vintage.max()
ts = T.text_sentiment()["sources"]["EIA STEO (에너지 품목, 월간)"]["tone"]
assert close(ts["WTI 원유"][f"{pd.Timestamp(v):%Y-%m-%d}"], S[(S.vintage == v) & (S.id == "yf:CL=F")].tone.iloc[0], 1e-3)
checked |= {"data_status", "list_items", "get_prices", "get_results", "text_sentiment"}
ok("사이드바 기준일 · 자료 목록 · 가격 · 실험 결과 표 · 문서 감성 = 대시보드·저장 파일")

# refresh_data: 사이드바 '데이터 업데이트'(app.py _update_data 150–160)와 같은 단계 · confirm 없으면 실행 안 함
p = T.refresh_data("prices")
assert p["executed"] is False and p["plan"]["steps"] == ["collect.py", "clean.py", "roll_adjust.py", "data_quality.py", "export_excel.py"]
checked.add("refresh_data"); ok("자료 갱신 단계 = 사이드바 버튼 단계, confirm 없이는 계획만")

same = {k for k, v in T.DASH.items() if v[1].startswith("같")}
assert same <= checked, f"대조 테스트 없는 '같음' 도구: {same - checked}"
print(f"\n모두 통과 — '같음' 도구 {len(same)}개 전부 대시보드 값과 일치 확인 ({time.time()-t0:.0f}s)")
