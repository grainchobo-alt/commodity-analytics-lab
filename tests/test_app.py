"""Streamlit 앱 헤드리스 테스트: 모든 버튼·모델 상세 경로에서 예외가 없는지 확인."""
import os, sys, tempfile, time
from pathlib import Path
os.environ["RUNLOG_DIR"] = tempfile.mkdtemp()      # 실행 기록은 임시 폴더로 (runs/ 오염 방지)
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
at.selectbox(key="pc_detail").set_value("yf:ZM=F").run(); check(at, "구매 타이밍: 대두박 상세")
at.number_input(key="pc_q_yf:ZC=F").set_value(60000.0).run(); check(at, "구매 타이밍: 카고 물량 변경")
assert not any("재고" in m.label for m in at.metric)                       # 재고 정보 제거
labels = [m.label for m in at.metric][:4]                     # 구매 카드 4개 (다른 탭 지표 제외)
assert all(any(k in l for k in ("적극 매수", "매수 보류", "판단 유보")) for l in labels), labels
at.slider(key="pc_aucmin").set_value(0.50).run(); check(at, "구매 타이밍: 활성화 기준 0.50 (근거 약한 품목 포함)")
at.slider(key="pc_pstar").set_value(0.50).run(); check(at, "구매 타이밍: 미루기 기준 p* 0.50")
at.multiselect(key="pc_ids").set_value(["yf:ZC=F", "yf:RB=F", "yf:LE=F", "yf:ZW=F"]).run(); check(at, "구매 타이밍: 근거 있는 품목(휘발유·생우) 포함")
assert sum("학습 AUC" in c.value and "시험 AUC" in c.value for c in at.caption) >= 4           # 카드마다 확률의 근거(학습·시험 AUC) 표시
assert any("매수 보류" in m.label for m in at.metric[:4])                                     # 기준을 낮추면 근거 있는 품목에서 🔴 매수 보류
at.selectbox(key="pc_hz_item").set_value("yf:RB=F").run(); check(at, "구매 타이밍: 미루는 기간별 AUC 그림 (휘발유)")
assert any("예측 정확도 (AUC)" in e.label for e in at.expander)                            # 품목별 AUC 표 (접기)
assert any("보완 실험" in e.label for e in at.expander)                                    # 가치·추세·레짐 신호 추가 AUC
assert any("설명변수" in e.label for e in at.expander)                                      # 판단 기준(설명변수) 펼치기
assert any(b.key == "data_update" for b in at.sidebar.button)                              # 사이드바 데이터 업데이트 버튼
assert any("거시" in m.value for m in at.markdown)                                         # 거시·금리·해운 표
assert any("펀더멘털 맥락" in e.label for e in at.expander)                                 # 펀더멘털 표 2개는 펼치기
at.checkbox(key="pc_comp").check().run(); check(at, "구매 타이밍: 점수 구성 요소 표시"); at.checkbox(key="pc_comp").uncheck().run()
assert any("t값 > 2" in i.value for i in at.info)                                          # ⑤ 롱숏 전략 표
assert any("비중 결정 방식 비교" in m.value for m in at.markdown)                          # ⑤ ⑦ MinVar·평균-분산 비교
assert any("⑨ 확장 — 칼만 필터" in m.value for m in at.markdown)                          # ⑨ 칼만·헤지펀드식 확장
assert any("사전 선택 검증" in e.label for e in at.expander)                               # ⑨ 학습으로 고르고 시험 확인
assert any("구간 이동 선택 검증" in e.label for e in at.expander)                          # ⑨ 시험 구간 3개 이동 선택
assert any("강세장 덕 vs 전략 덕" in e.label for e in at.expander)                          # 바스켓 대비 베타·알파
at.selectbox(key="ls_ct_per").set_value("전체 2007–").run(); check(at, "롱숏: 기간별 품목 기여도 (전체)")
at.selectbox(key="ls_ab_pick").set_value("따라잡기 혼합 (50/25/25)").run(); check(at, "롱숏: 2년 롤링 알파")
at.radio(key="ls_ra_win").set_value("3년").run(); check(at, "롱숏: 3년 롤링 알파")
assert any("롱숏 전략 2년 롤링 알파" in x.value for x in list(at.warning) + list(at.info) + list(at.caption))   # 리스크 탭 전략 우위 경고
assert any(str(i).startswith("⑩ 혼합 후보") for d in at.dataframe for i in getattr(d.value, "index", []))   # ⑩ 혼합 후보 행
assert any("MinVar 강건성" in e.label for e in at.expander)                               # ⑦ 강건성 점검
assert any("크기 조정(변동성 사이징)" in e.label for e in at.expander)                      # 사이징 방식·창 길이 비교
assert any("흡수비율" in x.value for x in list(at.warning) + list(at.info) + list(at.caption))   # 리스크 탭 상관 급등 경고
assert any("추세" in c.value and "점수 **" in c.value for c in at.caption)                  # ★ 카드 추세 배지
assert any("횡단면상 매력 묶음" in c.value and "점수 **" in c.value for c in at.caption)      # ★ 카드 횡단면 배지
at.selectbox(key="ls_pair").set_value("yf:GC=F|yf:SI=F").run(); check(at, "⑤ 롱숏: 금/은 페어 z")
at.number_input(key="ls_cost").set_value(10.0).run(); check(at, "⑤ 롱숏: 비용 10bp")
at.slider(key="pc_aucmin").set_value(0.55).run(); at.slider(key="pc_pstar").set_value(0.55).run()
at.multiselect(key="pc_ids").set_value(["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F"]).run()
at.selectbox(key="pc_lb_추천 (고도화: 상대가치+계절+MIDAS)").set_value("1년").run(); check(at, "구매 타이밍: 가격 위치 창 1년 고정")
at.selectbox(key="pc_lb_추천 (고도화: 상대가치+계절+MIDAS)").set_value("품목별 자동").run()
at.multiselect(key="pc_ids").set_value(["yf:ZC=F", "yf:BZ=F", "yf:HO=F", "yf:PA=F"]).run(); check(at, "구매 타이밍: 상대가치 없는 품목(경유·팔라듐) 가중치 재배분")
assert any("가중치 재배분" in c.value for c in at.caption)
at.multiselect(key="pc_ids").set_value(["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F"]).run(); check(at, "구매 타이밍: 곡물 4품목 복귀")
# ◆ 리스크
at.button(key="rk_run").click().run(); check(at, "리스크: 옥수수 8개 변동성 모형 추정")
at.selectbox(key="rk_hist").set_value(at.selectbox(key="rk_hist").options[-1]).run(); check(at, "리스크: 꼬리 히스토그램·왜도·첨도")
assert any("통계 지표 — AUC" in e.label for e in at.expander)                              # ⑧ 방법론 Q&A
assert any(t.label == "①+ 요인 모형" for t in at.tabs)
at.radio(key="cs_meth").set_value("계층 (Ward)").run(); check(at, "④+ 횡단면: 계층 군집 (구매 매력도 특성)")
at.radio(key="cs_proj").set_value("주성분 PCA (특성 전체 변동을 요약)").run(); check(at, "④+ 횡단면: PCA 지도")
at.radio(key="cs_fset").set_value("전체").run(); check(at, "④+ 횡단면: 전체 특성")
at.button(key="cs_bt_run").click().run(); check(at, "횡단면: 배지 검증 백테스트")                                    # PCA·DY·TVP-VAR·FAVAR 새 탭
at.button(key="vf_run").click().run(); check(at, "③ 변동성 예측 검증: 옥수수 경주")
at.selectbox(key="vf_base").set_value("GARCH(1,1)").run(); check(at, "③ 변동성 예측 검증: DM 기준 GARCH")
at.selectbox(key="vf_zm").set_value("HMM 3상태 λ").run(); check(at, "③ 변동성 예측 검증: HMM 잔차 진단")
at.button(key="vf_mrun").click().run(); check(at, "③ 변동성 예측 검증: 여러 품목 LR·마할라노비스")
at.button(key="vr_run").click().run(); check(at, "② 변동성 레짐: HMM 3상태 · λ 블록 검정")
at.radio(key="rk_side").set_value("보유 재고 (가격 하락 위험)").run(); check(at, "리스크: 보유 재고 VaR")
at.selectbox(key="rk_h").set_value(20).run(); at.selectbox(key="rk_alpha").set_value(0.95).run(); check(at, "리스크: 20일 95% VaR")
at.selectbox(key="rk_vm").set_value("HMM 3상태 λ").run(); check(at, "리스크: HMM σ VaR")
at.select_slider(key="cm_tdq").set_value(0.10).run(); check(at, "① 꼬리 의존 10%")
at.radio(key="rk_dir").set_value("보유 재고 (가격 하락이 손실)").run(); check(at, "리스크: 재고 물량 위험")
at.selectbox(key="rk_ph").set_value(60).run(); check(at, "리스크: 60일 물량 VaR")
at.selectbox(key="rk_id").set_value("yf:TIO=F").run()
at.button(key="rk_run").click().run(); check(at, "리스크: 철광석 재추정")
for sec in ["📋 한눈에 보기", "철강 원료·구리 상대가치", "전 품목 구매 타이밍 (단일 vs 다변량)", "곡물 상대가치 재검증 (2008~)", "벌크(일괄) 구매 — 문턱 · 가격 위치 창", "미루기 권고 (재고 무관 · 1~4주)", "재고 기반 반복 구매 (학습/시험 · 예측)", "리스크 모형", "모델 선택 메타 전략", "최신 시계열 모델 · 예측 목표", "전략 비교 (리스크 버짓)", "모멘텀 기간", "거래 규칙", "변동성 사이징", "구매 타이밍 (프리셋 · 단계별)"]:
    at.radio(key="ex_sec").set_value(sec).run(); check(at, f"실험 결과: {sec}")
at.radio(key="ex_sec").set_value("전 품목 구매 타이밍 (단일 vs 다변량)").run()
at.radio(key="ex_pa_view").set_value("단일 vs 다변량 요약").run(); check(at, "실험 결과: 전 품목 단일 vs 다변량 요약")
at.radio(key="ex_sec").set_value("모델 선택 메타 전략").run()
at.radio(key="ex_ms_fq").set_value("주간").run(); at.selectbox(key="ex_ms_id").set_value("yf:HG=F").run(); check(at, "실험 결과: 메타 전략 주간·구리")
at.radio(key="ex_sec").set_value("구매 타이밍 (프리셋 · 단계별)").run()
at.radio(key="ex_pc_view").set_value("프리셋 · 신호 비교 (13·26주, 품목별)").run(); check(at, "실험 결과: 구매 타이밍 프리셋 비교")

at.button(key="dybtn").click().run(); check(at, "DY 스필오버")
at.button(key="tvpbtn").click().run(); check(at, "TVP-VAR 연결성")
at.button(key="levbtn").click().run(); check(at, "레버리지 비교")
check(at, "② 레짐: 다변량 HMM (원자재 6 + 거시 8, 주간)")
assert any("지금 국면" in m.label for m in at.metric)
at.button(key="rv_bt_run").click().run(); check(at, "② 레짐: walk-forward 백테스트 (4개 변수 묶음)")
at.radio(key="rv_prof_kind").set_value("변동성 (전체 대비 배수)").run(); check(at, "② 레짐: 국면별 변동성 프로파일")
at.radio(key="rv_k").set_value(2).run(); check(at, "② 레짐: 2국면")
for m in ["Markov-switching 회귀", "Kalman TVP", "Threshold 회귀 (TAR)", "STAR (평활 전이)"]:
    at.selectbox(key="rv_model").set_value(m).run(); at.button(key="rv_reg_run").click().run(); check(at, f"  ② 레짐 회귀 (예측): {m}")
at.radio(key="rv_mode").set_value("설명 (동시점 베타)").run(); at.button(key="rv_reg_run").click().run(); check(at, "  ② 레짐 회귀 (설명): STAR")
at.selectbox(key="rv_model").set_value("Hidden Markov (다변량 레짐)").run()
at.radio(key="rv_fq").set_value("월간 (세계은행, 1990–)").run(); check(at, "② 레짐: 월간 1990– (세계은행 + 거시)")
at.radio(key="rv_fq").set_value("주간 (선물, 2000–)").run()
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
at.multiselect(key="models").set_value(A.CORE_MODELS + A.ADV_MODELS).run()                  # 한 목록에서 전체 선택
at.button(key="runml").click().run(); check(at, "전 모델 실행 (설명)")
for m in A.PREDICT_MODELS + ["앙상블 (단순평균)"]:
    at.selectbox(key="mlsel").set_value(m).run(); check(at, f"  상세: {m}")
at.selectbox(key="mlsel").set_value("LightGBM").run()
at.selectbox(key="shap_dep").set_value(at.selectbox(key="shap_dep").options[1]).run(); check(at, "  LightGBM SHAP 의존도")
for m in ["LightGBM", "Gaussian Process", "Ridge"]:
    at.selectbox(key="cf_model").set_value(m).run()
    at.button(key="cf_run").click().run(); check(at, f"  Conformal 구간: {m}")
# 예측 모드
at.radio(key="mode").set_value("예측 (미래 수익률)").run()
at.multiselect(key="models").set_value(["OLS", "Lasso", "Elastic Net", "PLS (부분최소제곱)", "MCP", "Gradient Boosting"]).run()
at.button(key="runml").click().run(); check(at, "예측 모드 실행")
assert any(str(i).startswith("위약 · 모델 7개 중 최댓값") for d in at.dataframe for i in getattr(d.value, "index", [])), "위약 기준선 행 없음"
at.selectbox(key="mlsel").set_value("PLS (부분최소제곱)").run(); check(at, "  상세: PLS (예측 모드)")
assert not set(A.REGIME_SET) & set(A.CORE_MODELS + A.ADV_MODELS)                 # 레짐·시변 모델은 ② 레짐 탭에서만
at.number_input(key="band").set_value(0.5).run()
at.button(key="runml").click().run(); check(at, "예측 모드: 무거래 구간 c=0.5")
at.number_input(key="band").set_value(0.0).run()
at.number_input(key="hor").set_value(4).run()
at.button(key="runml").click().run(); check(at, "예측 모드 h=4 (겹침 제거 지표)")
at.number_input(key="hor").set_value(1).run()
at.multiselect(key="fk").set_value(["모멘텀", "실현변동성", "계절성", "COT 헤징압력"]).run()
at.checkbox(key="wf").check().run()
at.multiselect(key="models").set_value(["OLS", "Ridge", "Lasso", "Elastic Net"]).run()
at.button(key="runml").click().run(); check(at, "추가 특성 + Walk-forward")
import runlog as RL
RR = RL.load_registry(); assert len(RR) >= 4 and RR.iloc[0]["WF Sharpe(비용 후)"] == RR.iloc[0]["WF Sharpe(비용 후)"], "walk-forward 실행이 기록되지 않음"
at.selectbox(key="ml_verdict").set_value("good").run(); at.text_input(key="ml_note").input("테스트 메모").run()
at.button(key="ml_vsave").click().run(); check(at, "실행 기록: ④ 판정 저장")
assert RL.load_registry().iloc[0]["verdict"] == "good"
at.radio(key="ex_sec").set_value("📂 내 실행 기록").run(); check(at, "⑥ 내 실행 기록")
rid0 = RL.load_registry().iloc[0]["run_id"]
at.selectbox(key="rn_sel").set_value(rid0).run()
at.multiselect(key=f"rv_t_{rid0}").set_value(["yf:ZC=F", "yf:ZS=F"]).run()
at.button(key=f"rv_run_{rid0}").click().run(); check(at, "⑥ 재검증: 같은 설정을 다른 품목에")
assert RL.load_registry().iloc[0]["kind"] == "revalidate" and any("판정:" in m.value for m in at.markdown)
at.multiselect(key="feats").set_value(["yf:ZS=F"]).run()
at.button(key=f"rn_restore_{rid0}").click().run(); check(at, "⑥ 설정 불러오기 → ④")
assert set(at.multiselect(key="feats").value) == set(RL.load_run(rid0)["settings"]["설명변수"]) and at.checkbox(key="wf").value
at.selectbox(key="mlsel").set_value("Ridge").run(); check(at, "  상세: Ridge (특성 포함, 잔차 진단)")
at.checkbox(key="vs").check().run()
at.button(key="runml").click().run(); check(at, "목표 변동성 표준화")
at.checkbox(key="vs").uncheck().run(); at.checkbox(key="wf").uncheck().run()
at.button(key="runml").click().run()
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

assert len(at.multiselect(key="pn_macro").value) >= 1                                      # 패널 거시 변수 (β×Δ)
at.multiselect(key="pn_models").set_value(["Pooled OLS", "Ridge", "Gradient Boosting"]).run()
at.button(key="pn_run").click().run(); check(at, "패널 모델 (ETF 기준, 주간)")
at.selectbox(key="pn_sel").set_value("Gradient Boosting").run(); check(at, "  패널 상세: Gradient Boosting")
at.button(key="pn_factor").click().run(); check(at, "패널: 단일 특성 팩터")
at.button(key="pn_econ").click().run(); check(at, "패널: 계량 추정 (FM·FE OLS·섹터 기울기)")
at.selectbox(key="pn_wffe").set_value("품목 FE").run(); at.checkbox(key="pn_wfsec").check().run()
at.button(key="pn_run").click().run(); check(at, "패널: walk-forward 품목 FE + 섹터 기울기")
at.selectbox(key="pn_wffe").set_value("없음 (Pooled)").run(); at.checkbox(key="pn_wfsec").uncheck().run()
assert at.radio(key="pf_sig").options == ["시계열 모멘텀 (기간 선택)", "과거 평균 (무정보 기준)"]

at.button(key="md_grun").click().run(); check(at, "MIDAS: GARCH-MIDAS (자기 실현분산)")
at.selectbox(key="md_mac").set_value("fred:VIXCLS").run()
at.button(key="md_grun").click().run(); check(at, "MIDAS: GARCH-MIDAS (VIX 요인)")
at.button(key="md_rrun").click().run(); check(at, "MIDAS: 수익 예측 walk-forward (④ 탭)")

at.button(key="favbtn").click().run(); check(at, "FAVAR")
# 빈도 변경 (월간) 후 전체 재렌더
at.sidebar.radio[0].set_value("월간").run(); check(at, "월간 빈도 재렌더")
at.button(key="favbtn").click().run(); check(at, "FAVAR (월간)")
print("APP TEST PASSED")
