"""Streamlit 앱 헤드리스 테스트: 모든 버튼·모델 상세 경로에서 예외가 없는지 확인."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
import analytics as A

t0 = time.time()
def check(at, label):
    errs = [e.value for e in at.exception]
    assert not errs, f"{label}: {errs[0][:500] if errs else ''}"
    print(f"[{time.time()-t0:6.1f}s] OK  {label}")

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
at.run(); check(at, "초기 렌더")

# ★ 구매 타이밍 · ⑥ 실험 결과
assert at.selectbox(key="pc_preset").options == ["추천 (고도화: 상대가치+계절+MIDAS)", "균형 (국소 저점 = 위치+반등)"]
at.checkbox(key="pc_adv").check().run(); check(at, "구매 타이밍: 고급 프리셋 표시")
for pz in ["보수적 (위치+반등+계절)", "공격적 (가격 위치만, 2배)", "균형 (국소 저점 = 위치+반등)", "추천 (고도화: 상대가치+계절+MIDAS)"]:
    at.selectbox(key="pc_preset").set_value(pz).run(); check(at, f"구매 타이밍: 프리셋 {pz}")
at.checkbox(key="pc_adv").uncheck().run(); check(at, "구매 타이밍: 기본 프리셋 2개로 복귀")
at.number_input(key="pc_weeks").set_value(26).run(); check(at, "구매 타이밍: 26주 기간 백테스트")
at.selectbox(key="pc_detail").set_value("yf:ZM=F").run(); check(at, "구매 타이밍: 대두박 상세")
at.number_input(key="pc_rem").set_value(12000.0).run(); check(at, "구매 타이밍: 매수량 계산기")
at.selectbox(key="pc_y0").set_value(2016).run(); check(at, "구매 타이밍: 검증 시작 2016")
at.number_input(key="pc_carry").set_value(3.0).run(); check(at, "구매 타이밍: 보유비용 3bp/주")
# ◆ 리스크
at.button(key="rk_run").click().run(); check(at, "리스크: 옥수수 8개 변동성 모형 추정")
at.selectbox(key="rk_base").set_value("GARCH(1,1)").run(); check(at, "리스크: DM 기준 GARCH")
at.selectbox(key="rk_zm").set_value("HMM 3상태 λ").run(); check(at, "리스크: HMM 잔차 진단")
at.button(key="rk_mrun").click().run(); check(at, "리스크: 여러 품목 LR·마할라노비스")
at.radio(key="rk_side").set_value("보유 재고 (가격 하락 위험)").run(); check(at, "리스크: 보유 재고 VaR")
at.selectbox(key="rk_h").set_value(20).run(); at.selectbox(key="rk_alpha").set_value(0.95).run(); check(at, "리스크: 20일 95% VaR")
at.selectbox(key="rk_vm").set_value("Kalman TVP-AR(1) + EWMA").run(); check(at, "리스크: Kalman σ VaR")
at.select_slider(key="rk_q").set_value(0.10).run(); check(at, "리스크: 꼬리 의존 10%")
at.radio(key="rk_dir").set_value("보유 재고 (가격 하락이 손실)").run(); check(at, "리스크: 재고 물량 위험")
at.selectbox(key="rk_ph").set_value(60).run(); check(at, "리스크: 60일 물량 VaR")
at.selectbox(key="rk_id").set_value("yf:TIO=F").run()
at.button(key="rk_run").click().run(); check(at, "리스크: 철광석 재추정")
for sec in ["철강 원료·구리 상대가치", "리스크 모형", "모델 선택 메타 전략", "전략 비교 (리스크 버짓)", "모멘텀 기간", "거래 규칙", "변동성 사이징", "구매 타이밍 (프리셋 · 단계별)"]:
    at.radio(key="ex_sec").set_value(sec).run(); check(at, f"실험 결과: {sec}")
at.radio(key="ex_sec").set_value("모델 선택 메타 전략").run()
at.radio(key="ex_ms_fq").set_value("주간").run(); at.selectbox(key="ex_ms_id").set_value("yf:HG=F").run(); check(at, "실험 결과: 메타 전략 주간·구리")
at.radio(key="ex_sec").set_value("구매 타이밍 (프리셋 · 단계별)").run()
at.radio(key="ex_pc_view").set_value("프리셋 · 신호 비교 (13·26주, 품목별)").run(); check(at, "실험 결과: 구매 타이밍 프리셋 비교")

at.button(key="dybtn").click().run(); check(at, "DY 스필오버")
at.button(key="tvpbtn").click().run(); check(at, "TVP-VAR 연결성")
at.button(key="levbtn").click().run(); check(at, "레버리지 비교")
at.button(key="msbtn").click().run(); check(at, "② 레짐: Markov 2레짐")
at.multiselect(key="rg_models").set_value(["Hidden Markov (HMM)", "Markov-switching 회귀", "Kalman TVP", "Threshold 회귀 (TAR)", "STAR (평활 전이)"]).run()
at.button(key="rg_run").click().run(); check(at, "② 레짐: 레짐 회귀 5종 실행")
for m in ["Hidden Markov (HMM)", "Markov-switching 회귀", "Kalman TVP", "Threshold 회귀 (TAR)", "STAR (평활 전이)"]:
    at.selectbox(key="rg_sel").set_value(m).run(); check(at, f"  ② 레짐 상세: {m}")
for kind in ["EGARCH", "GARCH"]:
    at.selectbox(key="g_kind").set_value(kind).run(); check(at, f"{kind} 모형")

# GARCH · DCC · BEKK 탭
at.button(key="mg_gbtn").click().run(); check(at, "단변량 GARCH 요약")
at.button(key="mg_dbtn").click().run(); check(at, "DCC")
at.radio(key="mg_dcc_t").set_value("ADCC (비대칭)").run()
at.button(key="mg_dbtn").click().run(); check(at, "ADCC")
at.multiselect(key="mg_bids").set_value(["yf:ZC=F", "yf:CL=F"]).run()
at.button(key="mg_bbtn").click().run(); check(at, "BEKK Full (2품목) + DCC 비교")
at.selectbox(key="mg_bkind").set_value("Diagonal").run()
at.button(key="mg_bbtn").click().run(); check(at, "BEKK Diagonal")

# 설명 모드: 전 모델
assert at.multiselect(key="models").value == A.CORE_MODELS and len(A.CORE_MODELS) <= 8
at.multiselect(key="models").set_value(A.CORE_MODELS).run()
at.multiselect(key="models_adv").set_value(A.ADV_MODELS).run()
at.button(key="runml").click().run(); check(at, "전 모델 실행 (설명)")
for m in A.MODEL_LIST + ["앙상블 (단순평균)"]:
    at.selectbox(key="mlsel").set_value(m).run(); check(at, f"  상세: {m}")
at.selectbox(key="mlsel").set_value("LightGBM").run()
at.selectbox(key="shap_dep").set_value(at.selectbox(key="shap_dep").options[1]).run(); check(at, "  LightGBM SHAP 의존도")
for m in ["LightGBM", "Gaussian Process", "Ridge"]:
    at.selectbox(key="cf_model").set_value(m).run()
    at.button(key="cf_run").click().run(); check(at, f"  Conformal 구간: {m}")
# 예측 모드
at.radio(key="mode").set_value("예측 (미래 수익률)").run()
at.multiselect(key="models").set_value(["OLS", "Lasso", "Kalman TVP", "Hidden Markov (HMM)"]).run()
at.multiselect(key="models_adv").set_value(["MCP", "Gradient Boosting"]).run()
at.button(key="runml").click().run(); check(at, "예측 모드 실행")
assert any(str(i).startswith("위약 · 모델 7개 중 최댓값") for d in at.dataframe for i in getattr(d.value, "index", [])), "위약 기준선 행 없음"
at.selectbox(key="mlsel").set_value("Hidden Markov (HMM)").run(); check(at, "  상세: HMM (예측 모드)")
at.number_input(key="hor").set_value(4).run()
at.button(key="runml").click().run(); check(at, "예측 모드 h=4 (겹침 제거 지표)")
at.number_input(key="hor").set_value(1).run()
at.multiselect(key="fk").set_value(["모멘텀", "실현변동성", "계절성", "COT 헤징압력"]).run()
at.checkbox(key="wf").check().run()
at.multiselect(key="models").set_value(["OLS", "Ridge", "Lasso", "Hidden Markov (HMM)"]).run()
at.multiselect(key="models_adv").set_value([]).run()
at.button(key="runml").click().run(); check(at, "추가 특성 + Walk-forward")
at.selectbox(key="mlsel").set_value("Ridge").run(); check(at, "  상세: Ridge (특성 포함, 잔차 진단)")
at.checkbox(key="vs").check().run()
at.button(key="runml").click().run(); check(at, "목표 변동성 표준화")
at.checkbox(key="vs").uncheck().run(); at.checkbox(key="wf").uncheck().run()
at.button(key="runml").click().run()
at.multiselect(key="ms_models").set_value(["OLS", "Ridge", "Kalman TVP"]).run()
at.button(key="ms_run").click().run(); check(at, "④ 모델 선택 메타 전략 (OLS·Ridge·Kalman, 주간)")
at.button(key="statbtn").click().run(); check(at, "정상성 진단")

at.button(key="pf_run").click().run(); check(at, "포트폴리오: 모멘텀·롱온리 (주간)")
at.selectbox(key="pf_mom").set_value("CTA 혼합 (EWMA 교차 8/24·16/48·32/96일)").run()
at.selectbox(key="pf_rule").set_value("무거래 구간 (필요할 때만 거래)").run()
at.button(key="pf_run").click().run(); check(at, "포트폴리오: CTA 모멘텀 + 무거래 구간")
at.selectbox(key="pf_rule").set_value("부분 조정 (차이의 κ만큼)").run()
at.selectbox(key="pf_vm").set_value("MIDAS-RV (권장)").run(); at.number_input(key="pf_vt").set_value(0.10).run()
at.button(key="pf_run").click().run(); check(at, "포트폴리오: 부분 조정 + MIDAS 변동성 + 타기팅 10%")
at.selectbox(key="pf_mom").set_value("12-1개월 (표준)").run()
at.selectbox(key="pf_rule").set_value("고정 주기 (점검일마다 목표로)").run()
at.selectbox(key="pf_vm").set_value("과거 창 공분산 (Ledoit-Wolf)").run(); at.number_input(key="pf_vt").set_value(0.0).run()
at.radio(key="pf_mode").set_value("롱숏 (총노출 Σ|w| ≤ L)").run()
at.button(key="pf_run").click().run(); check(at, "포트폴리오: 모멘텀·롱숏")
at.radio(key="pf_sig").set_value("과거 평균 (무정보 기준)").run()
at.button(key="pf_run").click().run(); check(at, "포트폴리오: 과거평균")
at.selectbox(key="pf_reb").set_value("매월 (월말)").run()
at.button(key="pf_run").click().run(); check(at, "포트폴리오: 주간 데이터 + 월간 리밸런싱")
at.radio(key="pf_basis").set_value("Yahoo 연속월물 (롤 보정)").run()
at.button(key="pf_run").click().run(); check(at, "포트폴리오: 연속월물 기준 (비교용)")

at.multiselect(key="pn_models").set_value(["Pooled OLS", "Ridge", "Gradient Boosting"]).run()
at.button(key="pn_run").click().run(); check(at, "패널 모델 (ETF 기준, 주간)")
at.selectbox(key="pn_sel").set_value("Gradient Boosting").run(); check(at, "  패널 상세: Gradient Boosting")
at.button(key="pn_factor").click().run(); check(at, "패널: 단일 특성 팩터")
at.button(key="pn_econ").click().run(); check(at, "패널: 계량 추정 (FM·FE OLS·섹터 기울기)")
at.selectbox(key="pn_wffe").set_value("품목 FE").run(); at.checkbox(key="pn_wfsec").check().run()
at.button(key="pn_run").click().run(); check(at, "패널: walk-forward 품목 FE + 섹터 기울기")
at.selectbox(key="pn_wffe").set_value("없음 (Pooled)").run(); at.checkbox(key="pn_wfsec").uncheck().run()
at.selectbox(key="pn_sel").set_value("Ridge").run()
at.button(key="pn_send").click().run(); check(at, "패널 → 포트폴리오 신호 전송")
at.radio(key="pf_basis").set_value("ETF 기반 선물 초과수익 (실제 롤 비용 반영, 권장)").run()
at.selectbox(key="pf_reb").set_value("매주 (금요일)").run()
at.radio(key="pf_sig").set_value("패널 모델 예측 (④ 탭 '여러 품목'에서 전송)").run()
at.button(key="pf_run").click().run(); check(at, "포트폴리오: 패널 신호 + Newton 최적화")

at.button(key="md_grun").click().run(); check(at, "MIDAS: GARCH-MIDAS (자기 실현분산)")
at.selectbox(key="md_mac").set_value("fred:VIXCLS").run()
at.button(key="md_grun").click().run(); check(at, "MIDAS: GARCH-MIDAS (VIX 요인)")
at.button(key="md_rrun").click().run(); check(at, "MIDAS: 수익 예측 walk-forward (④ 탭)")

at.button(key="favbtn").click().run(); check(at, "FAVAR")
# 빈도 변경 (월간) 후 전체 재렌더
at.sidebar.radio[0].set_value("월간").run(); check(at, "월간 빈도 재렌더")
at.button(key="favbtn").click().run(); check(at, "FAVAR (월간)")
print("APP TEST PASSED")
