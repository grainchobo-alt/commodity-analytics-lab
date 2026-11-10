"""Streamlit 앱 헤드리스 테스트: 모든 버튼·모델 상세 경로에서 예외가 없는지 확인.

묶음(GROUPS) 단위로 나뉘어 있고, 초기 렌더(탭 12개를 차례로 기본값으로 한 번씩 그림)는 항상 실행.
앱은 고른 탭만 계산(v1.65.0)하므로 LazyAT가 위젯을 못 찾으면 그 위젯이 있는 탭으로 옮겨 다시 찾음(AppTest는 탭 상태를 보내지 않아 _test_tab으로 탭을 고름).
  전체 (약 50분):      .venv\\Scripts\\python.exe tests\\test_app.py
  바뀐 곳만 (자동):    .venv\\Scripts\\python.exe tests\\test_app.py --auto [기준, 기본 origin/main]   ← 바뀐 파일 → 묶음 (tests/app_select.py)
  묶음 지정:           .venv\\Scripts\\python.exe tests\\test_app.py --only 구매,리스크
  묶음 목록:           .venv\\Scripts\\python.exe tests\\test_app.py --list
결과 한 줄은 data/last_test.txt (선택 실행이면 '선택: …'과 기준을 함께 기록 → push_check.py가 같은 변경에 대한 통과인지 판정).
"""
import os, sys, tempfile, time
from pathlib import Path
os.environ["RUNLOG_DIR"] = tempfile.mkdtemp()      # 실행 기록은 임시 폴더로 (runs/ 오염 방지)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import app_select as SEL

t0 = time.time()
LAST = ROOT / "data" / "last_test.txt"                 # 마지막 앱 테스트 결과 한 줄 (대화가 바뀌어도 확인용)
NOTE = ""                                              # 선택 실행이면 '· 선택: … · 기준 …'


def _record(status, note=""):
    import subprocess
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "?"
    line = f"{time.strftime('%Y-%m-%d %H:%M')} · {status} · 커밋 {head} · {time.time() - t0:.0f}초 {note} {NOTE}".rstrip()
    LAST.write_text(" ".join(line.split()) + "\n", encoding="utf-8")


def check(at, label):
    errs = [e.value for e in at.exception]
    if errs:
        _record("APP TEST FAILED", f"· 단계: {label}")
    assert not errs, f"{label}: {errs[0][:500] if errs else ''}"
    print(f"[{time.time()-t0:6.1f}s] OK  {label}")


class _Finder:
    """at.radio(key=…) 같은 위젯 찾기. 지금 탭에 없으면 다른 탭으로 옮겨 찾고, 찾은 탭을 기억."""
    def __init__(s, lat, kind):
        s.l, s.k = lat, kind

    def _list(s):
        return getattr(s.l._at, s.k)

    def __call__(s, key=None):
        try:
            return s._list()(key=key)
        except KeyError:
            pass
        order = ([s.l.where[key]] if key in s.l.where else []) + [t for t in s.l.tabs if t != s.l.tab]
        for t in dict.fromkeys(order):
            s.l.go(t)
            try:
                w = s._list()(key=key)
                s.l.where[key] = t
                return w
            except KeyError:
                continue
        raise KeyError(key)

    def __getitem__(s, i):
        return s._list()[i]

    def __len__(s):
        return len(s._list())

    def __iter__(s):
        return iter(s._list())


class LazyAT:
    """AppTest 감싸기: 위젯 접근 시 필요하면 탭 이동 (나머지 속성은 그대로 전달)."""
    KINDS = {"button", "checkbox", "radio", "selectbox", "multiselect", "slider", "select_slider", "number_input", "text_input",
             "text_area", "date_input", "time_input", "toggle", "color_picker"}

    def __init__(s, at):
        s._at, s.where, s.tab = at, {}, None
        s.tabs = []

    def go(s, label):
        s._at.session_state["_test_tab"] = label
        s.tab = label
        s._at.run()
        check(s._at, f"탭 이동: {label}")

    def scan(s, fn):
        """화면 내용 검사: 지금 탭에서 참이면 끝, 아니면 탭을 돌며 찾음 (찾은 탭에 머묾)."""
        if fn(s._at):
            return True
        for t in s.tabs:
            if t != s.tab:
                s.go(t)
                if fn(s._at):
                    return True
        return False

    def __getattr__(s, n):
        if n in LazyAT.KINDS:
            return _Finder(s, n)
        return getattr(s._at, n)


# ============================================================================
# 묶음 — 순서는 예전 한 줄짜리 테스트와 같음 (같은 at을 이어서 씀)
# ============================================================================
def g_buy(at, A):
    """★ 구매 타이밍 · ⑤ 롱숏 화면 확인"""
    assert at.selectbox(key="pc_preset").options == ["추천 (고도화: 상대가치+계절+MIDAS)", "균형 (국소 저점 = 위치+반등)"]
    at.checkbox(key="pc_adv").check().run(); check(at, "구매 타이밍: 고급 프리셋 표시")
    for pz in ["보수적 (위치+반등+계절)", "공격적 (가격 위치만, 2배)", "균형 (국소 저점 = 위치+반등)", "추천 (고도화: 상대가치+계절+MIDAS)"]:
        at.selectbox(key="pc_preset").set_value(pz).run(); check(at, f"구매 타이밍: 프리셋 {pz}")
    at.checkbox(key="pc_adv").uncheck().run(); check(at, "구매 타이밍: 기본 프리셋 2개로 복귀")
    at.selectbox(key="pc_detail").set_value("yf:ZM=F").run(); check(at, "구매 타이밍: 대두박 상세")
    at.number_input(key="pc_q_yf:ZC=F").set_value(60000.0).run(); check(at, "구매 타이밍: 카고 물량 변경")
    assert not at.scan(lambda at: any("재고" in m.label for m in at.metric))                       # 재고 정보 제거
    at.go(at.tabs[0])
    labels = [m.label for m in at.metric][:4]                     # 구매 카드 4개 (다른 탭 지표 제외)
    assert all(any(k in l for k in ("적극 매수", "매수 보류", "판단 유보")) for l in labels), labels
    at.slider(key="pc_aucmin").set_value(0.50).run(); check(at, "구매 타이밍: 활성화 기준 0.50 (근거 약한 품목 포함)")
    at.slider(key="pc_pstar").set_value(0.50).run(); check(at, "구매 타이밍: 미루기 기준 p* 0.50")
    at.multiselect(key="pc_ids").set_value(["yf:ZC=F", "yf:RB=F", "yf:LE=F", "yf:ZW=F"]).run(); check(at, "구매 타이밍: 근거 있는 품목(휘발유·생우) 포함")
    assert at.scan(lambda at: sum("학습 AUC" in c.value and "시험 AUC" in c.value for c in at.caption) >= 4)           # 카드마다 확률의 근거(학습·시험 AUC) 표시
    assert at.scan(lambda at: any("매수 보류" in m.label for m in at.metric[:4]))                                     # 기준을 낮추면 근거 있는 품목에서 🔴 매수 보류
    at.selectbox(key="pc_hz_item").set_value("yf:RB=F").run(); check(at, "구매 타이밍: 미루는 기간별 AUC 그림 (휘발유)")
    assert at.scan(lambda at: any("예측 정확도 (AUC)" in e.label for e in at.expander))                            # 품목별 AUC 표 (접기)
    assert at.scan(lambda at: any("보완 실험" in e.label for e in at.expander))                                    # 가치·추세·레짐 신호 추가 AUC
    assert at.scan(lambda at: any("설명변수" in e.label for e in at.expander))                                      # 판단 기준(설명변수) 펼치기
    assert at.scan(lambda at: any(b.key == "data_update" for b in at.sidebar.button))                              # 사이드바 데이터 업데이트 버튼
    assert at.scan(lambda at: any("거시" in m.value for m in at.markdown))                                         # 거시·금리·해운 표
    assert at.scan(lambda at: any("펀더멘털 맥락" in e.label for e in at.expander))                                 # 펀더멘털 표 2개는 펼치기
    at.checkbox(key="pc_comp").check().run(); check(at, "구매 타이밍: 점수 구성 요소 표시"); at.checkbox(key="pc_comp").uncheck().run()
    assert at.scan(lambda at: any(m.key == "ls_sel" for m in at.multiselect))                                      # ⑤ 롱숏 전략 표·누적 수익 (파란 t>2 박스는 v1.37.0에서 제거)
    assert at.scan(lambda at: any("비중 결정 방식 비교" in m.value for m in at.markdown))                          # ⑤ ⑦ MinVar·평균-분산 비교
    assert at.scan(lambda at: any("⑨ 확장 — 칼만 필터" in m.value for m in at.markdown))                          # ⑨ 칼만·헤지펀드식 확장
    assert at.scan(lambda at: any("사전 선택 검증" in e.label for e in at.expander))                               # ⑨ 학습으로 고르고 시험 확인
    assert at.scan(lambda at: any("구간 이동 선택 검증" in e.label for e in at.expander))                          # ⑨ 시험 구간 3개 이동 선택
    assert at.scan(lambda at: any("강세장 덕 vs 전략 덕" in e.label for e in at.expander))                          # 바스켓 대비 베타·알파
    at.selectbox(key="ls_ct_per").set_value("전체 2007–").run(); check(at, "롱숏: 기간별 품목 기여도 (전체)")
    at.selectbox(key="ls_ab_pick").set_value("따라잡기 혼합 (50/25/25)").run(); check(at, "롱숏: 2년 롤링 알파")
    at.radio(key="ls_ra_win").set_value("3년").run(); check(at, "롱숏: 3년 롤링 알파")
    assert at.scan(lambda at: any("대안 시계열 모델 — Prophet" in m.value for m in at.markdown))                   # ⑤ 포트폴리오 › 대안 시계열 모델 서브탭
    at.radio(key="ls_alt_mode").set_value("순위 롱숏").run(); check(at, "롱숏: 대안 시계열 순위 롱숏")
    assert at.scan(lambda at: any("D 위험할 때 현금 피신 — 규칙" in e.label for e in at.expander))                 # 롱숏 ⑪ 규칙별 상세표
    assert at.radio(key="cu_gcap").value == "곡물 25% (권장)"                                        # ⑤ Customization 품목군 위험 한도 (기본 = 곡물 25%)
    at.radio(key="cu_gcap").set_value("3품목군 25% (곡물 · 에너지 · 축산)").run(); check(at, "⑤ Customization: 품목군 상한 3품목군 25%")
    at.radio(key="cu_gcap").set_value("없음").run(); check(at, "⑤ Customization: 품목군 상한 없음")
    assert at.scan(lambda at: any("내 포트폴리오에 바로 적용" in c.value for c in at.caption))                 # 고른 조합에 바로 적용 (strategy_itemw)
    ms_ = at.multiselect(key="cu_sel_대안 시계열")
    ms_.set_value(ms_.options[:1]).run(); check(at, "⑤ Customization: 대안 시계열 섞으면 품목군 상한은 기본 혼합 고정 결과로")
    assert at.scan(lambda at: any("고정 결과를 보여 줌" in c.value for c in at.caption))
    at.multiselect(key="cu_sel_대안 시계열").set_value([]).run()
    at.radio(key="cu_freq").set_value("일간 (일간 자료로 새로 계산한 재료 — 3축 + 일간 신규 전략)").run(); check(at, "⑤ Customization: 데이터 주기 일간")   # v1.67.0
    assert at.scan(lambda at: any("일간 수익률 기준" in c.value for c in at.caption))
    assert at.scan(lambda at: any("주간 합산 Sharpe" in str(getattr(d.value, "columns", [])) for d in at.dataframe))
    at.selectbox(key="cu_method").set_value("최소분산 (직전 104주 공분산)").run(); check(at, "⑤ Customization: 일간 · 최소분산 비중")
    at.selectbox(key="cu_method").set_value("직접 입력").run()
    ms_d = at.multiselect(key="cu_sel_일간 실험 결과_d")                                                            # v1.68.0: 일간 실험 결과(메타 · 갈아타기 · 현금 피신 …) 드롭박스
    assert len(ms_d.options) > 100, "일간 실험 결과 옵션이 너무 적음"
    ms_d.set_value([o for o in ms_d.options if o.startswith("[B 메타 레이어]")][:1] + [o for o in ms_d.options if o.startswith("[D 현금 피신]")][:1]).run(); check(at, "⑤ Customization: 일간 실험 결과 전략 섞기")
    ov = at.selectbox(key="cu_ov_sig")                                                                          # 일간 신호 오버레이 (DCC · VIX · … · 칼만 · 낙폭 손절)
    for k_ in (0, 1, len(ov.options) - 2, len(ov.options) - 1):
        at.selectbox(key="cu_ov_sig").set_value(ov.options[k_]).run(); check(at, f"⑤ Customization: 일간 신호 오버레이 {ov.options[k_][:14]}")
    at.multiselect(key="cu_sel_일간 실험 결과_d").set_value([]).run()
    assert at.scan(lambda at: any(m.key == "dly_pick" for m in at.selectbox))                                   # F 서브탭: 일간 백테스트 결과
    at.selectbox(key="dly_pick").set_value(at.selectbox(key="dly_pick").options[1]).run(); check(at, "⑤ F 서브탭: 일간 백테스트 결과 묶음 선택")
    at.radio(key="cu_freq").set_value("주간 (고정 결과 — 롱숏 · 대안 시계열 · 멀티모달 재료 포함)").run(); check(at, "⑤ Customization: 데이터 주기 주간으로 복귀")
    assert at.scan(lambda at: any("롱숏 전략 2년 롤링 알파" in x.value for x in list(at.warning) + list(at.info) + list(at.caption)))   # 리스크 탭 전략 우위 경고
    assert at.scan(lambda at: any(str(i).startswith("⑩ 혼합 후보") for d in at.dataframe for i in getattr(d.value, "index", [])))   # ⑩ 혼합 후보 행
    assert at.scan(lambda at: any("MinVar 강건성" in e.label for e in at.expander))                               # ⑦ 강건성 점검
    assert at.scan(lambda at: any("크기 조정(변동성 사이징)" in e.label for e in at.expander))                      # 사이징 방식·창 길이 비교
    assert at.scan(lambda at: any("흡수비율" in x.value for x in list(at.warning) + list(at.info) + list(at.caption)))   # 리스크 탭 상관 급등 경고
    assert at.scan(lambda at: any("추세" in c.value and "점수 **" in c.value for c in at.caption))                  # ★ 카드 추세 배지
    assert at.scan(lambda at: any("횡단면상 매력 묶음" in c.value and "점수 **" in c.value for c in at.caption))      # ★ 카드 횡단면 배지
    at.selectbox(key="ls_pair").set_value("yf:GC=F|yf:SI=F").run(); check(at, "⑤ 롱숏: 금/은 페어 z")
    at.number_input(key="ls_cost").set_value(10.0).run(); check(at, "⑤ 롱숏: 비용 10bp")
    at.slider(key="pc_aucmin").set_value(0.55).run(); at.slider(key="pc_pstar").set_value(0.55).run()
    at.multiselect(key="pc_ids").set_value(["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F"]).run()
    at.selectbox(key="pc_lb_추천 (고도화: 상대가치+계절+MIDAS)").set_value("1년").run(); check(at, "구매 타이밍: 가격 위치 창 1년 고정")
    at.selectbox(key="pc_lb_추천 (고도화: 상대가치+계절+MIDAS)").set_value("품목별 자동").run()
    at.multiselect(key="pc_ids").set_value(["yf:ZC=F", "yf:BZ=F", "yf:HO=F", "yf:PA=F"]).run(); check(at, "구매 타이밍: 상대가치 없는 품목(경유·팔라듐) 가중치 재배분")
    assert at.scan(lambda at: any("가중치 재배분" in c.value for c in at.caption))
    at.multiselect(key="pc_ids").set_value(["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F"]).run(); check(at, "구매 타이밍: 곡물 4품목 복귀")


def g_risk(at, A):
    """◆ 리스크 · ③ 변동성 예측 검증 · ② 변동성 레짐 · ④+ 횡단면 · ① 꼬리 의존"""
    at.button(key="rk_run").click().run(); check(at, "리스크: 옥수수 8개 변동성 모형 추정")
    at.selectbox(key="rk_hist").set_value(at.selectbox(key="rk_hist").options[-1]).run(); check(at, "리스크: 꼬리 히스토그램·왜도·첨도")
    assert at.scan(lambda at: any("통계 지표 — AUC" in e.label for e in at.expander))                              # ⑧ 방법론 Q&A
    assert "①+ 요인 모형" in at.tabs
    at.radio(key="cs_meth").set_value("계층 (Ward)").run(); check(at, "④+ 횡단면: 계층 군집 (구매 매력도 특성)")
    at.radio(key="cs_proj").set_value("주성분 PCA (특성 전체 변동을 요약)").run(); check(at, "④+ 횡단면: PCA 지도")
    at.radio(key="cs_fset").set_value("전체").run(); check(at, "④+ 횡단면: 전체 특성")
    at.button(key="cs_bt_run").click().run(); check(at, "횡단면: 배지 검증 백테스트")
    at.button(key="vf_run").click().run(); check(at, "③ 변동성 예측 검증: 옥수수 경주")
    at.selectbox(key="vf_base").set_value("GARCH(1,1)").run(); check(at, "③ 변동성 예측 검증: DM 기준 GARCH")
    at.selectbox(key="vf_zm").set_value("HMM 3상태 λ").run(); check(at, "③ 변동성 예측 검증: HMM 잔차 진단")
    at.button(key="vf_mrun").click().run(); check(at, "③ 변동성 예측 검증: 여러 품목 LR·마할라노비스")
    at.button(key="vr_run").click().run(); check(at, "② 변동성 레짐: HMM 3상태 · λ 블록 검정")
    at.radio(key="rk_side").set_value("보유 재고 (가격 하락 위험)").run(); check(at, "리스크: 보유 재고 VaR")
    at.selectbox(key="rk_h").set_value(20).run(); at.selectbox(key="rk_alpha").set_value(0.95).run(); check(at, "리스크: 20일 95% VaR")
    at.selectbox(key="rk_vm").set_value("HMM 3상태 λ").run(); check(at, "리스크: HMM σ VaR")
    at.select_slider(key="cm_tdq").set_value(0.10).run(); check(at, "① 꼬리 의존 10%")
    assert at.scan(lambda at: any(m.key == "pv_sector" for m in at.selectbox))                                      # v1.69.0 ◆ 리스크 › 포트폴리오 VaR 서브탭 (기본 혼합 지금 비중)
    assert at.scan(lambda at: any("섹터별 VaR 기여" in m.value for m in at.markdown))
    at.selectbox(key="pv_sector").set_value(at.selectbox(key="pv_sector").options[-1]).run(); check(at, "포트폴리오 VaR: 다른 섹터 선택")
    at.radio(key="pv_mode").set_value("cut").run(); check(at, "포트폴리오 VaR: 그 섹터만 줄임")
    at.radio(key="pv_mode").set_value("redistribute").run(); check(at, "포트폴리오 VaR: 나머지에 재배분")
    at.selectbox(key="pv_h").set_value(5).run(); at.selectbox(key="pv_alpha").set_value(0.95).run(); check(at, "포트폴리오 VaR: 5일 95%")
    at.radio(key="pv_init").set_value("직접 고른 품목 (기본 5%씩)").run(); check(at, "포트폴리오 VaR: 직접 고른 품목")
    at.selectbox(key="pv_sigma").set_value("GARCH(1,1) (품목당 약 10초)").run(); check(at, "포트폴리오 VaR: GARCH σ")
    at.radio(key="rk_dir").set_value("보유 재고 (가격 하락이 손실)").run(); check(at, "리스크: 재고 물량 위험")
    at.selectbox(key="rk_ph").set_value(60).run(); check(at, "리스크: 60일 물량 VaR")
    at.selectbox(key="rk_id").set_value("yf:TIO=F").run()
    at.button(key="rk_run").click().run(); check(at, "리스크: 철광석 재추정")


def g_exp(at, A):
    """⑥ 실험 결과 — 섹션 전부"""
    for sec in ["📋 한눈에 보기", "🧪 3축 신호 · 위험 한도 실험 (2026-11)", "철강 원료·구리 상대가치", "전 품목 구매 타이밍 (단일 vs 다변량)", "곡물 상대가치 재검증 (2008~)", "벌크(일괄) 구매 — 문턱 · 가격 위치 창", "미루기 권고 (재고 무관 · 1~4주)", "재고 기반 반복 구매 (학습/시험 · 예측)", "리스크 모형", "모델 선택 메타 전략", "최신 시계열 모델 · 예측 목표", "전략 비교 (리스크 버짓)", "모멘텀 기간", "거래 규칙", "변동성 사이징", "구매 타이밍 (프리셋 · 단계별)"]:
        at.radio(key="ex_sec").set_value(sec).run(); check(at, f"실험 결과: {sec}")
    at.radio(key="ex_sec").set_value("전 품목 구매 타이밍 (단일 vs 다변량)").run()
    at.radio(key="ex_pa_view").set_value("단일 vs 다변량 요약").run(); check(at, "실험 결과: 전 품목 단일 vs 다변량 요약")
    at.radio(key="ex_sec").set_value("모델 선택 메타 전략").run()
    at.radio(key="ex_ms_fq").set_value("주간").run(); at.selectbox(key="ex_ms_id").set_value("yf:HG=F").run(); check(at, "실험 결과: 메타 전략 주간·구리")
    at.radio(key="ex_sec").set_value("구매 타이밍 (프리셋 · 단계별)").run()
    at.radio(key="ex_pc_view").set_value("프리셋 · 신호 비교 (13·26주, 품목별)").run(); check(at, "실험 결과: 구매 타이밍 프리셋 비교")


def g_factor_regime(at, A):
    """①+ 요인 (DY · TVP-VAR) · ③ 변동성 동학 · ② 레짐"""
    at.button(key="dybtn").click().run(); check(at, "DY 스필오버")
    at.button(key="tvpbtn").click().run(); check(at, "TVP-VAR 연결성")
    at.button(key="levbtn").click().run(); check(at, "레버리지 비교")
    check(at, "② 레짐: 다변량 HMM (원자재 6 + 거시 8, 주간)")
    assert at.scan(lambda at: any("지금 국면" in m.label for m in at.metric))
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


def g_mvgarch(at, A):
    """③ GARCH · DCC · BEKK"""
    at.button(key="mg_gbtn").click().run(); check(at, "단변량 GARCH 요약")
    at.button(key="mg_dbtn").click().run(); check(at, "DCC")
    at.radio(key="mg_dcc_t").set_value("ADCC (비대칭)").run()
    at.button(key="mg_dbtn").click().run(); check(at, "ADCC")
    at.multiselect(key="mg_bids").set_value(["yf:ZC=F", "yf:CL=F"]).run()
    at.button(key="mg_bbtn").click().run(); check(at, "BEKK Full (2품목) + DCC 비교")
    at.selectbox(key="mg_bkind").set_value("Diagonal").run()
    at.button(key="mg_bbtn").click().run(); check(at, "BEKK Diagonal")


def g_ml(at, A):
    """④ 회귀·ML (설명 · 예측 · walk-forward · 대안 모델) · ⑥ 내 실행 기록"""
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
    at.radio(key="mode").set_value("예측 (미래 수익률)").run()
    at.multiselect(key="models").set_value(["OLS", "Lasso", "Elastic Net", "PLS (부분최소제곱)", "MCP", "Gradient Boosting"]).run()
    at.button(key="runml").click().run(); check(at, "예측 모드 실행")
    assert at.scan(lambda at: any(str(i).startswith("위약 · 모델 7개 중 최댓값") for d in at.dataframe for i in getattr(d.value, "index", []))), "위약 기준선 행 없음"
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
    at.checkbox(key="vs").uncheck().run()
    zs = next(k for k in at.multiselect(key="alt_models").options if k.startswith("Chronos-Bolt"))
    at.multiselect(key="alt_models").set_value([zs]).run()
    at.button(key="runml").click().run(); check(at, "zero-shot 대안 모델 + Walk-forward (재학습 없이 같은 예측 재사용)")
    assert zs in at.session_state["ml_wf"], "zero-shot 모델이 walk-forward 결과에 없음"
    at.multiselect(key="alt_models").set_value([]).run(); at.checkbox(key="wf").uncheck().run()
    at.button(key="runml").click().run()
    at.button(key="statbtn").click().run(); check(at, "정상성 진단")


def g_portfolio(at, A):
    """⑤ Portfolio Customization → 품목 단위 평균-분산"""
    at.radio(key="cu_mode").set_value("품목 단위 평균-분산 최적화 (Newton, 기존 바스켓 포트폴리오)").run()   # v1.34.0부터 Customization 안의 모드
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
    assert at.radio(key="pf_sig").options == ["시계열 모멘텀 (기간 선택)", "과거 평균 (무정보 기준)"]


def g_panel(at, A):
    """패널 모델"""
    assert len(at.multiselect(key="pn_macro").value) >= 1                                      # 패널 거시 변수 (β×Δ)
    at.multiselect(key="pn_models").set_value(["Pooled OLS", "Ridge", "Gradient Boosting"]).run()
    at.button(key="pn_run").click().run(); check(at, "패널 모델 (ETF 기준, 주간)")
    at.selectbox(key="pn_sel").set_value("Gradient Boosting").run(); check(at, "  패널 상세: Gradient Boosting")
    at.button(key="pn_factor").click().run(); check(at, "패널: 단일 특성 팩터")
    at.button(key="pn_econ").click().run(); check(at, "패널: 계량 추정 (FM·FE OLS·섹터 기울기)")
    at.selectbox(key="pn_wffe").set_value("품목 FE").run(); at.checkbox(key="pn_wfsec").check().run()
    at.button(key="pn_run").click().run(); check(at, "패널: walk-forward 품목 FE + 섹터 기울기")
    at.selectbox(key="pn_wffe").set_value("없음 (Pooled)").run(); at.checkbox(key="pn_wfsec").uncheck().run()


def g_midas(at, A):
    """MIDAS · GARCH-MIDAS"""
    at.button(key="md_grun").click().run(); check(at, "MIDAS: GARCH-MIDAS (자기 실현분산)")
    at.selectbox(key="md_mac").set_value("fred:VIXCLS").run()
    at.button(key="md_grun").click().run(); check(at, "MIDAS: GARCH-MIDAS (VIX 요인)")
    at.button(key="md_rrun").click().run(); check(at, "MIDAS: 수익 예측 walk-forward (④ 탭)")


def g_favar(at, A):
    """FAVAR · 월간 빈도 전체 재렌더"""
    at.button(key="favbtn").click().run(); check(at, "FAVAR")
    at.sidebar.radio[0].set_value("월간").run(); check(at, "월간 빈도 재렌더")
    for t_ in at.tabs:                                                                          # 고른 탭만 계산하므로 탭마다 월간으로 다시 그림
        at.go(t_); check(at, f"월간 빈도 재렌더: {t_}")
    at.go(at.tabs[0])
    at.button(key="favbtn").click().run(); check(at, "FAVAR (월간)")
    at.sidebar.radio[0].set_value("주간").run()


def g_settings(at, A):
    """⑨ 설정: 배경색 바꿔도 오류 없음 (사용자 설정 파일은 테스트 후 원래대로 복원)"""
    ui_f = ROOT / "data" / "ui_settings.json"; ui_bak = ui_f.read_bytes() if ui_f.exists() else None
    try:
        at.selectbox(key="ui_bg").set_value("짙은 남색 (다크)").run(); check(at, "⑨ 설정: 다크 배경")
        at.selectbox(key="ui_bg").set_value("직접 고르기").run(); check(at, "⑨ 설정: 직접 고르기")
        at.button(key="ui_reset").click().run(); check(at, "⑨ 설정: 기본값 되돌리기")
    finally:
        if ui_bak is None:
            ui_f.unlink(missing_ok=True)
        else:
            ui_f.write_bytes(ui_bak)


GROUPS = {"구매": g_buy, "리스크": g_risk, "실험": g_exp, "요인·레짐": g_factor_regime, "다변량GARCH": g_mvgarch, "ML": g_ml,
          "포트폴리오": g_portfolio, "패널": g_panel, "MIDAS": g_midas, "FAVAR·월간": g_favar, "설정": g_settings}
assert set(GROUPS) == set(SEL.GROUP_FUNCS), "tests/app_select.py GROUP_FUNCS와 묶음 이름이 같아야 함"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    if "--list" in a:
        for k, f in GROUPS.items():
            print(f"{k:<10} {f.__doc__.strip()}  ← {', '.join(SEL.GROUP_FUNCS[k])}")
        sys.exit()
    pick = list(GROUPS)
    if "--only" in a:
        pick = [x.strip() for x in a[a.index("--only") + 1].split(",") if x.strip()]
        bad = [x for x in pick if x not in GROUPS]
        assert not bad, f"없는 묶음 {bad} — --list로 확인"
        NOTE = f"· 선택: {','.join(pick)} · 지정"
    elif "--auto" in a:
        i = a.index("--auto"); base = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else "origin/main"
        req, why = SEL.required(base)
        print("자동 선택 근거:\n  " + "\n  ".join(why or ["앱 관련 변경 없음"]))
        if req != SEL.ALL:
            pick = [k for k in GROUPS if k in req]
            NOTE = f"· 선택: {','.join(pick) or '초기 렌더만'} · 기준 {base}"
    print(f"실행 묶음: {', '.join(pick) or '초기 렌더만'}" + ("" if NOTE else " (전체)"))
    from streamlit.testing.v1 import AppTest
    import analytics as A
    at = LazyAT(AppTest.from_file(str(ROOT / "app.py"), default_timeout=600))
    at._at.run(); check(at, "앱 첫 실행")
    import ast, re
    at.tabs = ast.literal_eval(re.search(r"^TAB_LABELS = (\[.*\])$", (ROOT / "app.py").read_text(encoding="utf-8"), re.M).group(1))
    assert len(at.tabs) == 12 and at.tabs[0].startswith("★"), at.tabs
    for t_ in at.tabs:
        at.go(t_); check(at, f"초기 렌더: {t_}")
    at.go(at.tabs[0])
    for k in GROUPS:
        if k in pick:
            print(f"── {k}: {GROUPS[k].__doc__.strip()}")
            GROUPS[k](at, A)
    _record("APP TEST PASSED")
    print("APP TEST PASSED" + (f" {NOTE}" if NOTE else ""))
