"""구매 타이밍 (중간 트레이더의 매입 의사결정) — '정해진 기간 안에 필요한 물량을 언제·얼마나 살까'.

v1.7.0부터 대시보드는 벌크(카고 전량 일괄) 구매 판단: backtest_lump — 점수 ≥ τ 첫 주에 전량, 없으면 마감 주 (아래 분할 매수 규칙은 실험용으로 유지).
평가 기준은 Sharpe가 아니라 '균등 분할 매수(TWAP) 대비 평균 매입가 절감(bp)' (+ 선택: 재판매 시점까지 보유비용).
매수 점수 (0~1, 높을수록 지금 사기 좋음) = 가중 평균:
  value  : 1 − (최근 lookback일 가격 분위)                  → 쌀수록 ↑
  turn   : Φ(최근 10일 수익 z)                             → 단기 반등 = 국소 저점 통과 확인 (떨어지는 칼날 회피)
  relval : 대체재·가공마진 대비 상대가치 (1년 분위)          → 옥수수↔밀, 대두↔크러시 마진, 대두박·대두유↔대두,
                                                            철광석↔DCE 수입 차익, 점결탄↔철근 고로 마진, 구리↔금
  season : Φ(−과거 연도들의 같은 달 상대가격 / σ)          → 계절 패턴 (올해 이전 연도만; 전체·최근 10년·최근 가중 선택)
  trend  : Φ(12-1개월 모멘텀 z)                            → 정보용 (실험상 구매 타이밍에는 해로워 기본 가중치 0)
속도 규칙: 기본 = 남은 물량/남은 주, 배율 = clip(1 + k·확신·(2·점수 − 1), 0, m_max), 마감 주 전량.
  확신 = 1 − 0.5·위험,  위험 = EWMA 변동성 분위 | MIDAS 변동성 예측 분위 | HMM 위기 레짐 필터 확률 → 위험할수록 균등 매수에 가깝게.
신호는 롤 보정 가격(만기 교체 갭 제거), 경제적 가격 관계(크러시·대체비율)는 실제 근월물 시세, 매입가는 실제 근월물 시세,
체결 = 결정일(금) 다음 거래일 종가. MIDAS·HMM은 walk-forward(그 시점 이전 데이터로만 추정·필터) → 미래정보 없음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

import risk as RK

BUY_IDS = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀 (CBOT SRW)", "yf:KE=F": "밀 (KC HRW)", "yf:ZS=F": "대두",
           "yf:ZM=F": "대두박", "yf:ZL=F": "대두유",
           "yf:TIO=F": "철광석 (SGX 62%)", "ak:JM0": "점결탄 (DCE)", "yf:HG=F": "구리 (COMEX)",
           # v1.6.0 확장: ④ 수익률 예측의 나머지 원자재 (상대가치는 experiments.procurement_all 검증 후 채택분만 사용)
           "yf:ZO=F": "귀리", "yf:ZR=F": "쌀 (조곡)",
           "yf:CL=F": "WTI 원유", "yf:BZ=F": "브렌트 원유", "yf:HO=F": "난방유 (경유)", "yf:RB=F": "RBOB 휘발유", "yf:NG=F": "천연가스 (HH)",
           "yf:GC=F": "금", "yf:SI=F": "은", "yf:PL=F": "백금", "yf:PA=F": "팔라듐", "yf:ALI=F": "알루미늄 (COMEX)",
           "yf:CC=F": "코코아", "yf:KC=F": "커피", "yf:CT=F": "면화", "yf:SB=F": "원당 #11", "yf:OJ=F": "오렌지주스",
           "yf:LE=F": "생우", "yf:GF=F": "비육 송아지", "yf:HE=F": "돈육",
           "ak:A0": "대두 (DCE)", "ak:C0": "옥수수 (DCE)", "ak:M0": "대두박 (DCE)", "ak:Y0": "대두유 (DCE)",
           "ak:I0": "철광석 (DCE)", "ak:J0": "코크스 (DCE)", "ak:RB0": "철근 (SHFE)"}
# 원 단위 → 미국 달러/메트릭톤 (위안 표시 품목은 CNY_IDS: 그날 달러/위안 환율로 환산). 에너지는 대표 밀도로 근사 환산
# (원유 7.33 bbl/t, 경유 7.45 bbl/t, 휘발유 8.45 bbl/t, LNG 52 MMBtu/t) — 표시용이며 신호·백테스트 결과(bp)에는 영향 없음
_LB = 2204.62 / 100
TO_USD_T = {"yf:ZC=F": 39.368 / 100, "yf:ZW=F": 36.744 / 100, "yf:KE=F": 36.744 / 100, "yf:ZS=F": 36.744 / 100,
            "yf:ZM=F": 1.10231, "yf:ZL=F": _LB, "yf:TIO=F": 1.0, "yf:HG=F": 2204.62,
            "yf:ZO=F": 68.894 / 100, "yf:ZR=F": 22.0462, "yf:CL=F": 7.33, "yf:BZ=F": 7.33, "yf:HO=F": 7.45 * 42, "yf:RB=F": 8.45 * 42,
            "yf:NG=F": 52.0, "yf:GC=F": 1.0, "yf:SI=F": 1.0, "yf:PL=F": 1.0, "yf:PA=F": 1.0, "yf:ALI=F": 1.0, "yf:CC=F": 1.0,
            "yf:KC=F": _LB, "yf:CT=F": _LB, "yf:SB=F": _LB, "yf:OJ=F": _LB, "yf:LE=F": _LB, "yf:GF=F": _LB, "yf:HE=F": _LB}
GRAIN_IDS = ["yf:ZC=F", "yf:ZW=F", "yf:KE=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F"]   # 2008~ 검증·프리셋 결정에 쓴 품목
METAL_IDS = ["yf:TIO=F", "ak:JM0", "yf:HG=F"]                                    # 2013~ 별도 검증 (experiments.procurement_metals)
PX_UNIT = {k: "$/oz" for k in ["yf:GC=F", "yf:SI=F", "yf:PL=F", "yf:PA=F"]}   # 귀금속은 트로이온스 표시 (나머지 $/t)


def unit_of(i: str) -> str:
    return PX_UNIT.get(i, "$/t")


CNY_IDS = {"ak:JM0", "ak:A0", "ak:C0", "ak:M0", "ak:Y0", "ak:I0", "ak:J0", "ak:RB0"}
FX_CNY = "yf:CNY=X"
DEFAULT_W = {"value": 0.6, "turn": 0.4, "relval": 0.0, "season": 0.0, "trend": 0.0}   # 균형 = 국소 저점 (싸면서 반등 시작)
# 프리셋: 가중치·공격성 k·위험 판단·가격 위치 창·계절성 방식. 2008~2026, 6품목 × 13주 구매 기간 단계별 검증 결과:
#  - 추세는 구매자에게 해로움(평균 −20~−50bp) → 0
#  - 상대가치(실제 시세 관계)만 전·후반 모두 + (후반 +12bp, t≈2.1) — 롤보정 상대성과 버전은 효과 없음 → 경제적 관계가 핵심
#  - MIDAS 위험 판단·6개월 창·최근가중 계절은 소폭 개선 + 꼬리 위험 감소
# 재검증 (v1.6.2, experiments.procurement_grains_recheck — 같은 2008~ 기간, 경계 2017, 엄격 기준 rel_adoption):
#  - 이 '추천' 구성(6개월 창·MIDAS·계절) 안에서 곡물 현재 정의는 6개 모두 미통과. 6품목 평균 개선 +3.9bp(전반 −2 / 후반 +7~12)는
#    위약(무관한 가격비) 평균 +2bp·자기 가격 1년 위치 대조군 +5bp와 구별되지 않음 → 상대가치 40%의 근거는 '약함'으로 정정.
#  - 가중치는 그대로 둠: 후반(2017~) 평균은 + 이고, 대안(대두 CBOT/DCE만 통과, 5개 중 사후 선택·t≈1.8)도 확실하지 않음 — 변경은 사용자 결정.
_REC_W = {"value": 0.27, "turn": 0.18, "relval": 0.40, "season": 0.15, "trend": 0.0}
PRESETS = {"추천 (고도화: 상대가치+계절+MIDAS)": dict(w=_REC_W, k=1.0, risk="midas", lookback=126, season="ewm"),
           "균형 (국소 저점 = 위치+반등)": dict(w=DEFAULT_W, k=1.0, risk="ewma", lookback=252, season="all"),
           "보수적 (위치+반등+계절)": dict(w={"value": 0.45, "turn": 0.30, "season": 0.25, "relval": 0.0, "trend": 0.0}, k=1.0,
                                     risk="ewma", lookback=252, season="all"),
           "공격적 (가격 위치만, 2배)": dict(w={"value": 1.0, "turn": 0.0, "season": 0.0, "relval": 0.0, "trend": 0.0}, k=2.0,
                                       risk="ewma", lookback=252, season="all")}
BASIC_PRESETS = ["추천 (고도화: 상대가치+계절+MIDAS)", "균형 (국소 저점 = 위치+반등)"]   # 기본 노출. 보수적·공격적 = 검증상 열위 → 고급 옵션
COMP_LBL = {"value": "가격 위치 (쌀수록↑)", "turn": "단기 반등 (저점 확인)", "relval": "상대가치 (대체재 대비 쌀수록↑)",
            "season": "계절성 (저가 시즌↑)", "trend": "12개월 추세 (참고)"}
RISK_MODES = {"EWMA 변동성 분위": "ewma", "MIDAS 변동성 예측 분위": "midas", "HMM 위기 레짐 확률": "hmm"}
SEASON_MODES = {"전체 과거 연도": "all", "최근 10년": "10y", "최근 가중 (반감기 5년)": "ewm"}
# 상대가치 정의: 실제 근월물 시세 기반 경제적 관계. 부호 +1 = 값이 클수록 '내 품목이 상대적으로 쌈'
REL_DEF = {"yf:ZC=F": ("옥수수/밀 가격비 (낮을수록 옥수수 쌈)", -1), "yf:ZW=F": ("밀/옥수수 가격비 (낮을수록 밀 쌈)", -1),
           "yf:KE=F": ("KC밀/옥수수 가격비", -1), "yf:ZS=F": ("크러시 마진 (높을수록 대두 쌈)", +1),
           "yf:ZM=F": ("대두박 가치/대두 (낮을수록 대두박 쌈)", -1), "yf:ZL=F": ("대두유 가치/대두 (낮을수록 대두유 쌈)", -1),
           "yf:TIO=F": ("SGX/DCE 수입 차익 (낮을수록 SGX 철광석 쌈)", -1),
           "ak:JM0": ("철근 고로 마진 (높을수록 점결탄 쌈)", +1),
           "yf:HG=F": ("구리/금 가격비 (낮을수록 구리 쌈, 효과 약함)", -1)}
# 철강 원료 관계 (2013~2026, 13주 구매 · 추천 프리셋, 전반 2013~19.6 / 후반 2019.7~ 모두 + 인 정의만 채택):
#  - 철광석: SGX 달러 가격 vs DCE 위안 가격(환율·증치세 환산) — 중국 항구 수요 대비 해상 가격이 쌀 때 ↑  (+27bp, 전 +30 / 후 +24)
#  - 점결탄: 철근 − 1.6·철광석 − 0.5·코크스 (톤당 고로 마진) — 마진이 높으면 제철소 재고 확충 전 ↑     (+23bp, 전 +31 / 후 +17)
#  - 구리: 대체재 알루미늄 비율은 후반에 음(−) → 기각, 구리/금은 전·후반 + 이나 작음(+5bp, t≈1) — 참고 수준
#  - 탈락: 철광석 고로 마진·철광석/철근·철광석/점결탄, 점결탄 코크스 마진(효과 미미)


def cn_vat(idx: pd.DatetimeIndex) -> np.ndarray:
    """중국 증치세율 (DCE 가격은 세금 포함): 17% → 16% (2018-05-01) → 13% (2019-04-01)."""
    return np.where(idx < "2018-05-01", 1.17, np.where(idx < "2019-04-01", 1.16, 1.13))


def usd_t(Lr: pd.DataFrame, i: str) -> pd.Series:
    """실제 근월물 시세 → 달러/톤 (위안 품목은 그날까지의 달러/위안 환율, 세금 포함 그대로)."""
    s = Lr[i].dropna()
    if i in CNY_IDS:
        if FX_CNY not in Lr:
            return s * np.nan
        return s / Lr[FX_CNY].ffill().reindex(s.index)
    return s * TO_USD_T.get(i, 1.0)


# ============================================================================
# 입력 준비: 상대가치 관계 · MIDAS 변동성 · HMM 위기 확률 (모두 그 시점까지의 정보)
# ============================================================================
def rel_series(Lr: pd.DataFrame, i: str, La: pd.DataFrame | None = None, mode: str = "price") -> pd.Series | None:
    """mode='price': 실제 근월물 시세로 경제적 관계 (크러시 마진·가격비). mode='adj': 롤 보정 가격의 상대 성과(로그 비율)."""
    S = La if (mode == "adj" and La is not None) else Lr
    g = lambda k: S[k].dropna() if k in S else None
    ZC, ZW, ZS, ZM, ZL, KE = (g(k) for k in ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:KE=F"])
    TIO, I0, J0, JM, RB, HG, GC = (g(k) for k in ["yf:TIO=F", "ak:I0", "ak:J0", "ak:JM0", "ak:RB0", "yf:HG=F", "yf:GC=F"])
    try:
        if i == "yf:ZC=F":
            x = np.log(ZC) - np.log(ZW)
        elif i == "yf:ZW=F":
            x = np.log(ZW) - np.log(ZC)
        elif i == "yf:KE=F":
            x = np.log(KE) - np.log(ZC)
        elif i == "yf:ZS=F":
            x = (ZM * 0.022 * 100 + ZL * 11 - ZS) if mode == "price" else -(np.log(ZS) - np.log(ZM * 0.022 * 100 + ZL * 11))
        elif i == "yf:ZM=F":
            x = np.log(ZM * 0.022 * 100) - np.log(ZS)
        elif i == "yf:ZL=F":
            x = np.log(ZL * 11) - np.log(ZS)
        elif i == "yf:TIO=F":
            if mode == "price":
                fx = Lr[FX_CNY].ffill().reindex(I0.index)
                x = np.log(TIO) - np.log(I0 / fx / cn_vat(I0.index))
            else:
                x = -(np.log(TIO) - np.log(I0))
        elif i == "ak:JM0":
            x = (RB - 1.6 * I0 - 0.5 * J0) if mode == "price" else -(np.log(JM) - np.log(RB))
        elif i == "yf:HG=F":
            x = np.log(HG) - np.log(GC) if mode == "price" else -(np.log(HG) - np.log(GC))
        else:
            return None
    except (TypeError, KeyError):
        return None
    x = x.dropna()
    return x[x.index.dayofweek < 5]


# ---- 확장 품목 · 다변량 상대가치 후보 -----------------------------------------------------------------
# ("r", 상대 id, 부호): log(내 품목 $/t) − log(상대 $/t) — 낮을수록(부호 −1) 내 품목이 상대적으로 쌈. 위안 품목은 환율·증치세 환산.
# ("f", 함수(Lr), 부호): 정제마진 등 가격 차이(스프레드). 부호 +1 = 클수록 내 품목이 쌈.
def _crack321(Lr):   # 3-2-1 정제마진 ($/bbl) = (2·휘발유 + 1·경유 − 3·원유) / 3 — 높을수록 원유 수요(정제) 강함 → 제품 대비 원유가 쌈
    return (2 * Lr["yf:RB=F"] * 42 + Lr["yf:HO=F"] * 42 - 3 * Lr["yf:CL=F"]) / 3


def _crack321b(Lr):
    return (2 * Lr["yf:RB=F"] * 42 + Lr["yf:HO=F"] * 42 - 3 * Lr["yf:BZ=F"]) / 3


REL_CANDS = {
    "yf:ZC=F": [("옥수수/밀 (사료 대체)", "r", "yf:ZW=F", -1), ("옥수수/대두 (재배면적 경쟁)", "r", "yf:ZS=F", -1),
                ("옥수수/원유 (에탄올)", "r", "yf:CL=F", -1), ("CBOT/DCE 옥수수 (수입 패리티)", "r", "ak:C0", -1)],
    "yf:ZW=F": [("밀/옥수수 (사료 대체)", "r", "yf:ZC=F", -1), ("SRW/HRW 밀", "r", "yf:KE=F", -1), ("밀/쌀 (식용 곡물)", "r", "yf:ZR=F", -1)],
    "yf:KE=F": [("HRW밀/옥수수", "r", "yf:ZC=F", -1), ("HRW/SRW 밀", "r", "yf:ZW=F", -1)],
    "yf:ZS=F": [("크러시 마진 (현재)", "cur", None, +1), ("대두/옥수수 (재배면적 경쟁)", "r", "yf:ZC=F", -1),
                ("CBOT/DCE 대두 (수입 패리티)", "r", "ak:A0", -1)],
    "yf:ZM=F": [("대두박 가치/대두 (현재)", "cur", None, -1), ("대두박/옥수수 (단백질 vs 에너지 사료)", "r", "yf:ZC=F", -1),
                ("CBOT/DCE 대두박", "r", "ak:M0", -1)],
    "yf:ZL=F": [("대두유 가치/대두 (현재)", "cur", None, -1), ("대두유/난방유 (바이오디젤)", "r", "yf:HO=F", -1),
                ("CBOT/DCE 대두유", "r", "ak:Y0", -1)],
    "yf:ZO=F": [("귀리/옥수수 (사료 대체)", "r", "yf:ZC=F", -1), ("귀리/밀", "r", "yf:ZW=F", -1)],
    "yf:ZR=F": [("쌀/밀 (식용 곡물)", "r", "yf:ZW=F", -1), ("쌀/옥수수", "r", "yf:ZC=F", -1)],
    "yf:CL=F": [("WTI/브렌트", "r", "yf:BZ=F", -1), ("3-2-1 정제마진 (높을수록 원유 쌈)", "f", _crack321, +1), ("원유/천연가스", "r", "yf:NG=F", -1)],
    "yf:BZ=F": [("브렌트/WTI", "r", "yf:CL=F", -1), ("3-2-1 정제마진 (브렌트)", "f", _crack321b, +1),
                ("경유 크랙 (브렌트, 높을수록 원유 쌈)", "f", lambda L: L["yf:HO=F"] * 42 - L["yf:BZ=F"], +1), ("브렌트/천연가스", "r", "yf:NG=F", -1)],
    "yf:HO=F": [("경유 크랙 (경유 − 원유, 낮을수록 경유 쌈)", "f", lambda L: L["yf:HO=F"] * 42 - L["yf:CL=F"], -1),
                ("경유/원유", "r", "yf:CL=F", -1), ("경유/휘발유", "r", "yf:RB=F", -1), ("경유/브렌트", "r", "yf:BZ=F", -1),
                ("경유/천연가스 (난방 연료 대체)", "r", "yf:NG=F", -1)],
    "yf:RB=F": [("휘발유 크랙 (휘발유 − 원유)", "f", lambda L: L["yf:RB=F"] * 42 - L["yf:CL=F"], -1),
                ("휘발유/원유", "r", "yf:CL=F", -1), ("휘발유/경유", "r", "yf:HO=F", -1)],
    "yf:NG=F": [("가스/원유 (열량 대비)", "r", "yf:CL=F", -1), ("가스/난방유", "r", "yf:HO=F", -1)],
    "yf:GC=F": [("금/은", "r", "yf:SI=F", -1), ("금/백금", "r", "yf:PL=F", -1), ("금/구리", "r", "yf:HG=F", -1),
                ("금/원유", "r", "yf:CL=F", -1), ("금/S&P500", "r", "yf:^GSPC", -1)],
    "yf:SI=F": [("은/금", "r", "yf:GC=F", -1), ("은/구리 (산업 수요)", "r", "yf:HG=F", -1)],
    "yf:PL=F": [("백금/팔라듐 (촉매 대체)", "r", "yf:PA=F", -1), ("백금/금", "r", "yf:GC=F", -1)],
    "yf:PA=F": [("팔라듐/백금 (촉매 대체)", "r", "yf:PL=F", -1), ("팔라듐/금", "r", "yf:GC=F", -1), ("팔라듐/구리 (산업 수요)", "r", "yf:HG=F", -1)],
    "yf:ALI=F": [("알루미늄/구리 (대체재)", "r", "yf:HG=F", -1), ("알루미늄/원유 (에너지 원가)", "r", "yf:CL=F", -1)],
    "yf:HG=F": [("구리/금 (현재)", "r", "yf:GC=F", -1), ("구리/알루미늄", "r", "yf:ALI=F", -1), ("구리/은", "r", "yf:SI=F", -1)],
    "yf:CC=F": [("코코아/원당 (초콜릿 원료)", "r", "yf:SB=F", -1), ("코코아/커피", "r", "yf:KC=F", -1), ("코코아/대두유 (유지 대체)", "r", "yf:ZL=F", -1)],
    "yf:KC=F": [("커피/코코아", "r", "yf:CC=F", -1), ("커피/원당", "r", "yf:SB=F", -1)],
    "yf:CT=F": [("면화/원유 (폴리에스터 대체)", "r", "yf:CL=F", -1), ("면화/옥수수 (재배면적)", "r", "yf:ZC=F", -1),
                ("면화/대두 (재배면적)", "r", "yf:ZS=F", -1)],
    "yf:SB=F": [("원당/원유 (브라질 에탄올 패리티)", "r", "yf:CL=F", -1), ("원당/옥수수 (HFCS)", "r", "yf:ZC=F", -1)],
    "yf:OJ=F": [("오렌지주스/원당", "r", "yf:SB=F", -1), ("오렌지주스/커피 (음료)", "r", "yf:KC=F", -1)],
    "yf:LE=F": [("생우/비육 송아지 (비육 마진)", "r", "yf:GF=F", -1), ("생우/돈육 (육류 대체)", "r", "yf:HE=F", -1), ("생우/옥수수 (사료비)", "r", "yf:ZC=F", -1)],
    "yf:GF=F": [("비육 송아지/생우", "r", "yf:LE=F", -1), ("비육 송아지/옥수수 (사료비 ↑ → 송아지 ↓)", "r", "yf:ZC=F", -1),
                ("비육 마진 (생우 − 송아지 − 사료, 높을수록 송아지 쌈)", "f",
                 lambda L: 12.5 * L["yf:LE=F"] - 7.5 * L["yf:GF=F"] - 50 * L["yf:ZC=F"] / 100, +1)],
    "yf:HE=F": [("돈육/옥수수 (hog-corn ratio)", "r", "yf:ZC=F", -1), ("돈육/대두박", "r", "yf:ZM=F", -1), ("돈육/생우", "r", "yf:LE=F", -1)],
    "ak:A0": [("DCE/CBOT 대두 (수입 패리티)", "r", "yf:ZS=F", -1), ("DCE 대두/옥수수", "r", "ak:C0", -1)],
    "ak:C0": [("DCE/CBOT 옥수수 (수입 패리티)", "r", "yf:ZC=F", -1), ("DCE 옥수수/대두박", "r", "ak:M0", -1), ("DCE 옥수수/대두", "r", "ak:A0", -1)],
    "ak:M0": [("DCE 대두박/대두", "r", "ak:A0", -1), ("DCE/CBOT 대두박", "r", "yf:ZM=F", -1), ("DCE 대두박/옥수수", "r", "ak:C0", -1)],
    "ak:Y0": [("DCE 대두유/대두", "r", "ak:A0", -1), ("DCE/CBOT 대두유", "r", "yf:ZL=F", -1)],
    "ak:I0": [("DCE/SGX 철광석 (수입 패리티)", "r", "yf:TIO=F", -1), ("철광석/철근", "r", "ak:RB0", -1),
              ("철근 고로 마진 (높을수록 철광석 쌈)", "f", lambda L: L["ak:RB0"] - 1.6 * L["ak:I0"] - 0.5 * L["ak:J0"], +1)],
    "ak:J0": [("코크스/점결탄 (코킹 마진, 낮을수록 코크스 쌈)", "r", "ak:JM0", -1), ("코크스/철근", "r", "ak:RB0", -1)],
    "ak:RB0": [("철근 고로 마진 (낮을수록 철근 쌈)", "f", lambda L: L["ak:RB0"] - 1.6 * L["ak:I0"] - 0.5 * L["ak:J0"], -1),
               ("철근/철광석", "r", "ak:I0", -1)],
}
MULTI_AVG, MULTI_COINT = "다변량: 후보 분위 평균", "다변량: 공적분 잔차 (적정가 대비)"


def _usd_ex_vat(Lr: pd.DataFrame, k: str) -> pd.Series:
    s = usd_t(Lr, k)
    return s / cn_vat(s.index) if k in CNY_IDS else s


def cand_series(Lr: pd.DataFrame, i: str, spec, La: pd.DataFrame | None = None) -> pd.Series | None:
    """후보 상대가치 1개 → 시계열 (실제 근월물 시세, 영업일)."""
    _, kind, arg, _sg = spec
    try:
        if kind == "cur":
            return rel_series(Lr, i, La, "price")
        if kind == "r":
            if arg not in Lr or i not in Lr:
                return None
            a, b = _usd_ex_vat(Lr, i), _usd_ex_vat(Lr, arg).reindex(Lr.index).ffill(limit=5)
            x = np.log(a) - np.log(b.reindex(a.index))
        else:
            x = arg(Lr)
    except (KeyError, TypeError):
        return None
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    return x[x.index.dayofweek < 5] if len(x) else None


def rel_multi_avg(Lr: pd.DataFrame, i: str, La=None, window: int = 252) -> pd.Series | None:
    """여러 상대가치 후보를 각각 1년 분위(부호 반영, 1 = 쌈)로 만든 뒤 평균 → 0~1 점수 (rel_sign=0 으로 그대로 사용)."""
    P = []
    for spec in REL_CANDS.get(i, []):
        x = cand_series(Lr, i, spec, La)
        if x is None or len(x) < window:
            continue
        pct = x.rolling(window, min_periods=window // 2).rank(pct=True)
        P.append(pct if spec[3] > 0 else 1 - pct)
    if len(P) < 2:
        return None
    return pd.concat(P, axis=1).mean(axis=1, skipna=True).dropna()


def rel_multi_coint(Lr: pd.DataFrame, i: str, window: int = 756, min_obs: int = 504) -> pd.Series | None:
    """다변량 적정가: log(내 품목) = a + Σ b_k log(대체재·원료 k) 를 매월 초 직전 window일로 추정(OLS) →
    그 달의 잔차 = 실제 − 적정가 (클수록 비쌈, rel_sign −1). 계수는 과거 자료로만 추정 → 미래정보 없음."""
    legs = [spec[2] for spec in REL_CANDS.get(i, []) if spec[1] == "r" and spec[2] in Lr]
    if len(legs) < 2 or i not in Lr:
        return None
    y = np.log(_usd_ex_vat(Lr, i))
    X = pd.concat({k: np.log(_usd_ex_vat(Lr, k)).reindex(Lr.index).ffill(limit=5) for k in legs}, axis=1).reindex(y.index)
    Z = pd.concat([y.rename("__y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    Z = Z[Z.index.dayofweek < 5]
    if len(Z) < min_obs + 21:
        return None
    out = pd.Series(np.nan, index=Z.index)
    months = Z.index.to_period("M")
    for m in months.unique():
        cur = months == m
        t0 = np.flatnonzero(cur)[0]
        past = Z.iloc[max(0, t0 - window):t0]
        if len(past) < min_obs:
            continue
        A_ = np.column_stack([np.ones(len(past)), past[legs].values])
        b = np.linalg.lstsq(A_, past["__y"].values, rcond=None)[0]
        Zc = Z[cur]
        out[cur] = Zc["__y"].values - np.column_stack([np.ones(len(Zc)), Zc[legs].values]) @ b
    return out.dropna()


def rel_option(Lr: pd.DataFrame, i: str, name: str, La=None) -> tuple[pd.Series | None, int]:
    """상대가치 선택지 이름 → (시계열, 부호). 부호 0 = 이미 0~1 점수."""
    if name == MULTI_AVG:
        return rel_multi_avg(Lr, i, La), 0
    if name == MULTI_COINT:
        return rel_multi_coint(Lr, i), -1
    for spec in REL_CANDS.get(i, []):
        if spec[0] == name:
            return cand_series(Lr, i, spec, La), spec[3]
    return None, 1


# 확장 품목의 채택 상대가치 (experiments.procurement_all + procurement_placebo, 2026-10-20, v1.6.1 엄격 기준 = experiments.rel_adoption):
# '추천' 프리셋에서 상대가치만 바꿔 ① 13·26주 모두 전반·후반이 '상대가치 없음'보다 좋고 ② 평균 개선 ≥ 5bp ③ 무관한 품목과의 가격비(위약)
# 개선의 90% 분위 초과 ④ 대조군(자기 가격 1년 위치)보다 큰 선택지 중 최대 개선. 나머지 확장 품목은 상대가치 없음 (가중치 자동 재배분).
# 위약 검증: 무관한 가격비도 31%가 ①을 통과 — '내 품목/아무 품목' 비율이 내 가격의 1년 위치를 담기 때문 → ③④ 없이는 채택 불가.
#   (돈육/옥수수 +66bp는 대조군 +53bp·위약 90% +66bp로 대부분 자기 가격 효과 → 기각. WTI는 위약 90%가 +22bp로 높아 기각)
# 곡물 6·철광석·점결탄·구리는 기존 정의 유지. 곡물은 원래 기간(2008~)에서도 엄격 기준 미통과 (v1.6.2 재검증, 위 _REC_W 주석 참고).
REL_CHOSEN = {"yf:NG=F": "가스/원유 (열량 대비)", "yf:LE=F": "생우/비육 송아지 (비육 마진)", "ak:M0": "DCE 대두박/옥수수",
              "yf:ALI=F": "알루미늄/구리 (대체재)", "yf:ZR=F": "쌀/밀 (식용 곡물)", "yf:BZ=F": "경유 크랙 (브렌트, 높을수록 원유 쌈)",
              "yf:KC=F": MULTI_AVG, "ak:C0": "DCE 옥수수/대두", "yf:GC=F": MULTI_COINT, "ak:A0": "DCE/CBOT 대두 (수입 패리티)"}
for _i, _nm in REL_CHOSEN.items():
    _sg = {MULTI_AVG: 0, MULTI_COINT: -1}.get(_nm, next((c[3] for c in REL_CANDS.get(_i, []) if c[0] == _nm), -1))
    REL_DEF[_i] = (_nm + (" (높을수록 쌈)" if _sg > 0 else " (낮을수록 쌈)" if _sg < 0 else " (0~1 점수, 높을수록 쌈)"), _sg)


def hmm_crisis_prob(p_adj: pd.Series, min_years: int = 5, seed: int = 0) -> pd.Series:
    """2-레짐 가우시안 HMM (일간 수익률). 매년 초 그 이전 데이터로만 재추정 → 그해 각 날짜의 '필터' 확률
    P(고변동 레짐 | r₁..r_t) (평활 확률은 미래를 쓰므로 사용 안 함). 전방 필터는 risk.hmm_filter와 공용."""
    from hmmlearn.hmm import GaussianHMM
    r = RK.returns(p_adj)
    x = r.values; idx = r.index; yrs = idx.year.values
    out = np.full(len(r), np.nan)
    for y in sorted(set(yrs)):
        tr = yrs < y
        if tr.sum() < 252 * min_years:
            continue
        m = GaussianHMM(2, "diag", n_iter=100, random_state=seed).fit(x[tr][:, None])
        seg = np.flatnonzero(yrs <= y)
        cur = yrs[seg] == y
        out[seg[cur]] = RK.hmm_filter(x[seg], m)[cur, -1]                 # 마지막 열 = 고변동 레짐
    return pd.Series(out, index=idx)


def midas_var(p_adj: pd.Series, h: int = 5) -> pd.Series:
    import midas as MD
    return MD.daily_vol_forecasts(RK.returns(p_adj), h=h)["MIDAS-RV"]


def inputs(La: pd.DataFrame, Lr: pd.DataFrame, i: str, risk: str = "ewma", rel_mode: str = "price") -> dict:
    if i in REL_CHOSEN and rel_mode == "price":
        rel, sg = rel_option(Lr, i, REL_CHOSEN[i], La)
    else:
        rel, sg = rel_series(Lr, i, La, rel_mode), (REL_DEF.get(i, ("", 1))[1] if rel_mode == "price" else 1)
    return dict(rel=rel, rel_sign=sg,
                vol_fc=midas_var(La[i]) if risk == "midas" else None, crisis=hmm_crisis_prob(La[i]) if risk == "hmm" else None)


# ============================================================================
# 신호 · 점수 · 속도
# ============================================================================
def _season(lp: pd.Series, idx: pd.DatetimeIndex, mode: str = "all", hl: float = 5.0) -> np.ndarray:
    m = lp.resample("ME").mean()
    rel = m - m.groupby(m.index.year).transform("mean")
    sea = np.full(len(idx), np.nan)
    years = idx.year.values; months = idx.month.values
    for y in np.unique(years):
        past = rel[rel.index.year < y]
        if mode == "10y":
            past = past[past.index.year >= y - 10]
        if past.index.year.nunique() < 5:
            continue
        if mode == "ewm":
            wt = 0.5 ** ((y - 1 - past.index.year.values) / hl)
            g = pd.Series(past.values * wt).groupby(past.index.month.values).sum() / pd.Series(wt).groupby(past.index.month.values).sum()
        else:
            g = past.groupby(past.index.month).mean()
        sd = past.std() or 1.0
        sel = years == y
        sea[sel] = norm.cdf(-g.reindex(months[sel]).fillna(0).values / sd)
    return sea


def signals(p_adj: pd.Series, lookback: int = 252, rel: pd.Series | None = None, rel_sign: int = 1,
            vol_fc: pd.Series | None = None, crisis: pd.Series | None = None, season: str = "all") -> pd.DataFrame:
    """일간 롤 보정 가격 → 시점별 구성 점수(0~1)·위험 지표. 모든 값은 그 시점까지의 데이터만 사용."""
    p = p_adj.dropna(); p = p[p > 0]
    p = p[p.index.dayofweek < 5]
    lp = np.log(p); r = lp.diff()
    out = pd.DataFrame(index=p.index)
    out["value"] = 1 - lp.rolling(lookback, min_periods=lookback // 2).rank(pct=True)
    vol_d = r.rolling(63, min_periods=20).std()
    mom = lp.shift(21) - lp.shift(252)
    out["trend"] = norm.cdf(mom / (vol_d * np.sqrt(231)))
    out["turn"] = norm.cdf((lp - lp.shift(10)) / (vol_d * np.sqrt(10)))
    out["season"] = _season(lp, p.index, season)
    if rel is not None and len(rel):
        rr = rel.reindex(p.index).ffill(limit=5)
        if rel_sign == 0:                                              # 이미 0~1 점수 (다변량 평균 분위)
            out["relval"] = rr.clip(0, 1)
        else:
            pct = rr.rolling(252, min_periods=126).rank(pct=True)
            out["relval"] = pct if rel_sign > 0 else 1 - pct
    else:
        out["relval"] = np.nan
    ew = RK.ewma_std(r)                                               # RiskMetrics λ=0.94
    out["vol_ann"] = ew * np.sqrt(252)
    out["vol_pct"] = ew.rolling(756, min_periods=126).rank(pct=True)
    if vol_fc is not None:
        vf = np.sqrt(vol_fc.reindex(p.index).ffill(limit=5))
        out["risk"] = vf.rolling(756, min_periods=126).rank(pct=True)
        out["vol_ann"] = (vf / 100 * np.sqrt(252)).fillna(out["vol_ann"])
    elif crisis is not None:
        out["risk"] = crisis.reindex(p.index).ffill(limit=5)
    else:
        out["risk"] = out["vol_pct"]
    return out


def score(sig: pd.DataFrame, w: dict | None = None) -> pd.Series:
    w = w or DEFAULT_W
    cols = [k for k in w if k in sig and w[k] > 0]
    S = sig[cols]
    ww = pd.Series({k: w[k] for k in cols})
    num = (S * ww).sum(axis=1, min_count=1); den = S.notna().mul(ww).sum(axis=1)
    return (num / den.replace(0, np.nan)).clip(0, 1)


def eff_weights(sig_row: pd.Series, w: dict) -> dict:
    """그 시점에 실제 쓰인 가중치: 값이 없는 구성요소(예: 상대가치 관계가 없는 품목)는 빠지고 나머지 가중치가 비례 재배분 (score와 동일 규칙)."""
    act = {c: v for c, v in w.items() if v > 0 and c in sig_row and np.isfinite(sig_row[c])}
    tot = sum(act.values())
    return {c: v / tot for c, v in act.items()} if tot else {}


def reason(sig_row: pd.Series, w: dict, top: int = 3) -> str:
    """점수에 가장 크게 작용한 구성요소 → '가격 위치 82 (유리) · 상대가치 30 (불리)' (가중치 × |점수 − 50| 순)."""
    ew = eff_weights(sig_row, w)
    short_ = {"value": "가격 위치", "turn": "단기 반등", "relval": "상대가치", "season": "계절성", "trend": "추세"}
    items = sorted(ew, key=lambda c: -ew[c] * abs(sig_row[c] - 0.5))[:top]
    tag = lambda v: "유리" if v >= 0.65 else ("불리" if v <= 0.35 else "보통")
    return " · ".join(f"{short_[c]} {sig_row[c] * 100:.0f} ({tag(sig_row[c])})" for c in items)


def pace(score_now: float, risk: float, remaining: float, weeks_left: int, k: float = 1.0, m_max: float = 3.0) -> float:
    """이번 주 매수량 (남은 물량 단위). 마감 주는 전량. risk ∈ [0,1] (클수록 보수적)."""
    if weeks_left <= 1:
        return remaining
    base = remaining / weeks_left
    if not np.isfinite(score_now):
        return base
    conf = 1 - 0.5 * (risk if np.isfinite(risk) else 0.5)
    mult = float(np.clip(1 + k * conf * (2 * score_now - 1), 0, m_max))
    return min(remaining, base * mult)


# ============================================================================
# 백테스트
# ============================================================================
def backtest(p_act: pd.Series, p_adj: pd.Series, weeks: int = 13, step: int = 4, start="2008-01-01", k: float = 1.0,
             m_max: float = 3.0, w: dict | None = None, lookback: int = 252, inp: dict | None = None, season: str = "all",
             carry_bp_week: float = 0.0, score_override: pd.Series | None = None) -> pd.DataFrame:
    """구매 기간(weeks주)을 step주 간격으로 반복 시작. 매주 금요일(결정일) 점수 → 다음 거래일 실제 종가에 매수.
    carry_bp_week: 기간 말(재판매·인도 시점)까지 보유하는 주당 비용(금융+보관, bp) — 일찍 살수록 비용 ↑, 모든 방식에 동일 적용.
    반환: 기간별 실효 평균 매입가 — 규칙 / 균등(TWAP) / 기간 초 일괄 / 가격 위치만 / 사후 최저(이론 상한) + 시작 시점 위험 지표."""
    inp = inp or {}
    sig = signals(p_adj, lookback, inp.get("rel"), inp.get("rel_sign", 1), inp.get("vol_fc"), inp.get("crisis"), season)
    sc = score(sig, w) if score_override is None else score_override.reindex(sig.index)   # score_override: 메타 라벨링 등 외부 조정 점수
    sc_v = score(sig, {"value": 1.0})
    pa = p_act.dropna(); pa = pa[pa > 0]
    days = pa.index.intersection(sig.index)
    fri = pd.Series(days, index=days).groupby(days.to_period("W-FRI")).max()
    fri = pd.DatetimeIndex(fri.values)
    fri = fri[fri >= pd.Timestamp(start)]
    pos = days.get_indexer(fri)
    exe = [days[i + 1] if i + 1 < len(days) else None for i in pos]
    hold = 1 + carry_bp_week / 1e4 * (weeks - 1 - np.arange(weeks))      # 주차별 보유비용 배수
    rows = []
    for s0 in range(0, len(fri) - weeks - 1, step):
        win = range(s0, s0 + weeks)
        if exe[s0 + weeks - 1] is None:
            break
        px = np.array([pa[exe[j]] for j in win]) * hold
        res = {}
        for nm, s_ in [("규칙", sc), ("가격 위치만", sc_v)]:
            rem, q = 1.0, np.zeros(weeks)
            for t, j in enumerate(win):
                d = fri[j]
                q[t] = pace(float(s_.get(d, np.nan)), float(sig["risk"].get(d, np.nan)), rem, weeks - t, k, m_max)
                rem -= q[t]
            res[nm] = float(q @ px)
        rows.append(dict(start=exe[s0], end=exe[s0 + weeks - 1], 규칙=res["규칙"], 가격위치만=res["가격 위치만"],
                         균등=float(px.mean()), 일괄=float(px[0]), 사후최저=float(px.min()),
                         시작위험=float(sig["vol_pct"].get(fri[s0], np.nan))))
    B = pd.DataFrame(rows).set_index("start")
    for c in ["규칙", "가격위치만", "일괄", "사후최저"]:
        B[f"절감_{c}"] = (B["균등"] - B[c]) / B["균등"] * 1e4            # bp, +면 균등보다 싸게 삼
    return B


# ============================================================================
# 벌크(일괄) 구매: 한 카고(3~6만 톤)를 한 번에 — '지금 살까, 기다릴까' (v1.7.0)
# ============================================================================
LUMP_TAU = 0.60          # 기본 매수 문턱 (experiments.procurement_lump: 곡물·확장 품목 공통으로 두 하위기간 모두 안정적인 값)


# 가격 위치 창 품목별 기본값 (procurement_lump, τ=0.6, 8·13주 평균): 1년 창이 6개월보다 전·후반 모두 낫고 +10bp 이상인 품목만 1년.
# (돈육·귀리는 오히려 6개월이 훨씬 나음 — '자기 가격 1년 위치' 상대가치 대조군 결과와 별개). 두 후보 중 고른 것이라 개선 폭은 다소 낙관적.
LOOKBACK_1Y = {"yf:ZM=F", "yf:ZW=F", "yf:KE=F", "yf:CT=F", "yf:PA=F", "yf:SB=F", "yf:GC=F", "ak:M0", "ak:J0"}


def item_lookback(i: str, default: int = 126) -> int:
    """'품목별 자동' 가격 위치 창: LOOKBACK_1Y 품목은 1년(252일), 나머지는 프리셋 값."""
    return 252 if i in LOOKBACK_1Y else default


def _fridays(pa: pd.Series, sig_index: pd.DatetimeIndex, start):
    days = pa.index.intersection(sig_index)
    fri = pd.Series(days, index=days).groupby(days.to_period("W-FRI")).max()
    fri = pd.DatetimeIndex(fri.values)
    fri = fri[fri >= pd.Timestamp(start)]
    pos = days.get_indexer(fri)
    exe = [days[i + 1] if i + 1 < len(days) else None for i in pos]
    return fri, exe


def backtest_lump(p_act: pd.Series, p_adj: pd.Series, weeks: int = 13, step: int = 4, start="2008-01-01", tau: float = LUMP_TAU,
                  w: dict | None = None, lookback: int = 252, inp: dict | None = None, season: str = "all", carry_bp_week: float = 0.0,
                  score_override: pd.Series | None = None) -> pd.DataFrame:
    """구매 창(weeks주) 안에서 전량을 한 번에 산다. 규칙: 매주 금요일 점수 ≥ tau 인 첫 주에 전량 매수(다음 거래일 종가), 끝까지 없으면 마지막 주.
    비교: 평균(창 안 무작위 하루에 산 기대 가격 = 매주 가격 평균) · 즉시(창 첫 주) · 마감(마지막 주) · 사후최저(이론 상한).
    carry_bp_week: 마감 시점까지의 주당 보유비용 — 일찍 살수록 비용 ↑ (모든 방식 동일 적용)."""
    inp = inp or {}
    sig = signals(p_adj, lookback, inp.get("rel"), inp.get("rel_sign", 1), inp.get("vol_fc"), inp.get("crisis"), season)
    sc = score(sig, w) if score_override is None else score_override.reindex(sig.index)
    pa = p_act.dropna(); pa = pa[pa > 0]
    fri, exe = _fridays(pa, sig.index, start)
    hold = 1 + carry_bp_week / 1e4 * (weeks - 1 - np.arange(weeks))
    rows = []
    for s0 in range(0, len(fri) - weeks - 1, step):
        if exe[s0 + weeks - 1] is None:
            break
        px = np.array([pa[exe[j]] for j in range(s0, s0 + weeks)]) * hold
        scs = np.array([float(sc.get(fri[j], np.nan)) for j in range(s0, s0 + weeks)])
        hit = np.flatnonzero(np.nan_to_num(scs, nan=-1) >= tau)
        k = int(hit[0]) if len(hit) else weeks - 1
        rows.append(dict(start=exe[s0], 규칙=px[k], 매수주=k + 1, 신호매수=bool(len(hit)), 평균=px.mean(), 즉시=px[0], 마감=px[-1], 사후최저=px.min(),
                         시작점수=scs[0]))
    B = pd.DataFrame(rows).set_index("start")
    for c in ["규칙", "즉시", "마감", "사후최저"]:
        B[f"절감_{c}"] = (B["평균"] - B[c]) / B["평균"] * 1e4            # bp, +면 '창 안 무작위 하루'보다 싸게 삼
    B["규칙_vs_즉시"] = (B["즉시"] - B["규칙"]) / B["즉시"] * 1e4          # +면 '오늘 바로 사기'보다 기다린 규칙이 쌈
    return B


def wait_stats(p_act: pd.Series, sc: pd.Series, horizon_w: int = 4, start="2008-01-01") -> pd.DataFrame:
    """과거 금요일마다: 다음 거래일에 바로 샀을 때 vs 향후 horizon_w주 안의 가격. 점수 구간별 요약 (실현이 끝난 과거만):
    '더 싼 날이 왔을 확률'(최저가 < 오늘 −1%), 향후 최저가까지 평균 하락(%), horizon_w주 뒤 가격 변화 중앙값(%)."""
    pa = p_act.dropna(); pa = pa[pa > 0]
    fri, exe = _fridays(pa, sc.dropna().index, start)
    lp = np.log(pa)
    rows = []
    for d, e in zip(fri, exe):
        if e is None:
            continue
        fut = lp.loc[e:].iloc[1:horizon_w * 5 + 1]
        if len(fut) < horizon_w * 5:
            break
        rows.append(dict(date=d, 점수=float(sc.get(d, np.nan)), 최저하락=(np.exp(fut.min() - lp[e]) - 1) * 100,
                         기말변화=(np.exp(fut.iloc[-1] - lp[e]) - 1) * 100))
    W_ = pd.DataFrame(rows).dropna()
    W_["구간"] = pd.cut(W_.점수, [-0.01, 0.35, 0.5, 0.65, 1.01], labels=["35 미만 (비쌈)", "35~50", "50~65", "65 이상 (쌈)"])
    g = W_.groupby("구간", observed=True)
    return pd.DataFrame({"표본 수": g.size(), "더 싼 날 올 확률 (−1% 이상)": g.최저하락.apply(lambda x: (x < -1).mean()),
                         "향후 최저까지 평균 (%)": g.최저하락.mean(), f"{horizon_w}주 뒤 변화 중앙값 (%)": g.기말변화.median()})


# ============================================================================
# 재고 기반 반복 구매 (v1.8.0): 카고(파나막스 3~6만 톤)를 평균 G주마다 1척, 저장 상한 C척 —
# '지금 가격이 충분히 괜찮으면(점수 ≥ τ) 이번 주 1카고, 아니면 대기, 재고가 바닥나기 직전이면 무조건 1카고'
# 기준: 정해진 간격(G주마다) 구매 · 매주 균등 분할(TWAP). 예측(1~2주 뒤 하락 확률)은 '살 수 있는 주'에 1주 미루기에만 사용.
# ============================================================================
INV_TAU = 0.45           # experiments.procurement_inventory: 학습 기간(2008~2016) 25품목 평균 정기 대비 최대(+37bp)인 τ → 시험(2017~) +10bp (균등 대비 +13bp)
#   곡물 6품목만 보면 시험 기간 효과 소멸(정기 대비 −10~+2bp), 확장 품목은 +13~+21bp. 1~2주 가격 방향 예측 AUC ≈ 0.51 (사실상 없음);
#   '하락 확률 > 0.6이면 1주 미루기'는 학습 +2.5bp → 시험 +4.7bp로 소폭 개선이나 AUC가 낮아 신뢰 낮음 → 기본 꺼짐. '다음 주 신호 충족' 예측 AUC ≈ 0.96은 점수의 주간 자기상관 때문(가격 예측이 아님).
INV_G = 4                # 평균 구매 간격 (주) = 소비 속도 1카고 / G주
INV_CAP = 2.0            # 저장 상한 (카고 수)


def weekly_frame(p_act: pd.Series, sig: pd.DataFrame, sc: pd.Series, start="2001-01-01") -> pd.DataFrame:
    """금요일 결정 · 다음 거래일 체결 가격 + 그날의 점수·구성요소 (모두 그 시점까지의 정보)."""
    pa = p_act.dropna(); pa = pa[pa > 0]
    fri, exe = _fridays(pa, sig.index, start)
    ok = [e is not None for e in exe]
    W_ = pd.DataFrame({"exe": [e for e, o in zip(exe, ok) if o]}, index=fri[ok])
    W_["px"] = pa.reindex(W_["exe"]).values
    W_["score"] = sc.reindex(W_.index).values
    for c in ["value", "turn", "relval", "season", "trend", "vol_pct"]:
        W_[c] = sig[c].reindex(W_.index).values if c in sig else np.nan
    return W_


def fc_features(W_: pd.DataFrame) -> pd.DataFrame:
    lp = np.log(W_["px"])
    X = pd.DataFrame(index=W_.index)
    X["score"] = W_["score"]; X["d_score"] = W_["score"].diff()
    for c in ["value", "turn", "relval", "season", "vol_pct"]:
        X[c] = W_[c]
    X["r1"] = lp.diff(); X["r4"] = lp.diff(4); X["r13"] = lp.diff(13)
    X["vol4"] = lp.diff().rolling(13, min_periods=6).std()
    return X


def fc_walkforward(W_: pd.DataFrame, horizon: int = 1, tau: float = INV_TAU, first_test_year: int = 2008, min_train: int = 260) -> pd.DataFrame:
    """1~2주 앞 방향 예측 (walk-forward, 매년 1월 그 이전 자료로만 로지스틱 회귀 재학습, 목표 겹침만큼 학습 끝 purge):
    p_down = P(horizon주 뒤 체결가 < 이번 주 체결가), p_sig = P(다음 주 점수 ≥ τ). 반환 = OOS 예측 (학습 연도 이전은 NaN)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = fc_features(W_)
    lp = np.log(W_["px"])
    y_down = (lp.shift(-horizon) < lp).astype(float).where(lp.shift(-horizon).notna())
    y_sig = (W_["score"].shift(-1) >= tau).astype(float).where(W_["score"].shift(-1).notna())
    out = pd.DataFrame(index=W_.index, columns=["p_down", "p_sig"], dtype=float)
    for yr in sorted(set(W_.index.year)):
        if yr < first_test_year:
            continue
        te = W_.index.year == yr
        before_all = np.flatnonzero(W_.index < pd.Timestamp(f"{yr}-01-01"))
        Xb = X.iloc[before_all]
        cols = [c for c in X.columns if Xb[c].notna().mean() > 0.5]                 # 특성 선택·결측 대체도 학습 구간 자료로만
        if not cols:
            continue
        med = Xb[cols].median()
        Xf = X[cols].fillna(med)
        xok = (X[cols].notna().sum(axis=1) >= len(cols) - 2).values
        for nm, y in [("p_down", y_down), ("p_sig", y_sig)]:
            h_ = horizon if nm == "p_down" else 1
            before = before_all[:-h_] if h_ and len(before_all) > h_ else before_all   # 목표가 그해로 넘어가는 마지막 h주 제거 (purge)
            tr_idx = before[np.isfinite(y.values[before]) & xok[before]]
            if len(tr_idx) < min_train or len(set(y.iloc[tr_idx])) < 2:
                continue
            sc_ = StandardScaler().fit(Xf.iloc[tr_idx])
            m = LogisticRegression(C=0.5, max_iter=500).fit(sc_.transform(Xf.iloc[tr_idx]), y.iloc[tr_idx])
            out.loc[te, nm] = m.predict_proba(sc_.transform(Xf[te]))[:, 1]
    out["y_down"] = y_down; out["y_sig"] = y_sig
    return out


def backtest_inventory(W_: pd.DataFrame, tau: float = INV_TAU, G: int = INV_G, cap: float = INV_CAP, start="2008-01-01", end=None,
                       fc: pd.DataFrame | None = None, p_wait: float | None = None, inv0: float = 1.0, carry_bp_week: float = 0.0) -> dict:
    """재고 시뮬레이션 (카고 단위). 매주: 재고 −1/G(소비) → 재고 < 1/G (다음 주 바닥) 이면 강제 1카고 →
    아니면 점수 ≥ τ 이고 재고+1 ≤ cap 이면 1카고 (단, fc 와 p_wait 가 있으면 '1주 뒤 하락 확률 > p_wait 이고 1주 더 버틸 재고'일 때 미룸).
    기준: 정기 = G주마다 1카고(같은 주기 시작), 균등 = 매주 1/G카고. 결과: 평균 매입가·카고 수·절감 bp·연도별 차이."""
    Wv = W_[(W_.index >= pd.Timestamp(start)) & ((W_.index <= pd.Timestamp(end)) if end is not None else True)].dropna(subset=["px"])
    inv, buys, forced, deferred, inv_path = float(inv0), [], 0, 0, []
    pw = fc["p_down"].reindex(Wv.index) if (fc is not None and p_wait is not None) else None
    for k, (d, r) in enumerate(Wv.iterrows()):
        inv -= 1.0 / G
        sc = r["score"]
        if inv < 1.0 / G:
            buys.append((d, r["px"], "강제")); inv += 1; forced += 1
        elif np.isfinite(sc) and sc >= tau and inv + 1 <= cap + 1e-9:
            if pw is not None and np.isfinite(pw.get(d, np.nan)) and pw[d] > p_wait and inv - 1.0 / G >= 1.0 / G:
                deferred += 1
            else:
                buys.append((d, r["px"], "신호")); inv += 1
        inv_path.append(inv)
    fx_inv = float(inv0) - (np.arange(1, len(Wv) + 1)) / G + (np.arange(len(Wv)) // G + 1)   # 정기: 매 G주 첫 주에 1척
    Bdf = pd.DataFrame(buys, columns=["date", "px", "사유"]).set_index("date")
    fixed = Wv["px"].iloc[::G]
    twap = Wv["px"].mean()
    rule = Bdf["px"].mean() if len(Bdf) else np.nan
    yr = pd.DataFrame({"규칙": Bdf["px"].groupby(Bdf.index.year).mean(), "정기": fixed.groupby(fixed.index.year).mean(),
                       "균등": Wv["px"].groupby(Wv.index.year).mean()})
    yr["vs정기bp"] = (yr["정기"] - yr["규칙"]) / yr["정기"] * 1e4; yr["vs균등bp"] = (yr["균등"] - yr["규칙"]) / yr["균등"] * 1e4
    inv_r, inv_f = float(np.mean(inv_path)) if inv_path else np.nan, float(np.mean(fx_inv)) if len(fx_inv) else np.nan
    hold_bp = carry_bp_week * (inv_r - inv_f) * len(Wv) / max(1, len(Bdf))      # 카고 1척당 추가 보유비용 (bp, 규칙이 재고를 더 들고 있으면 +)
    vs_f = (fixed.mean() - rule) / fixed.mean() * 1e4
    return dict(buys=Bdf, 규칙=rule, 정기=fixed.mean(), 균등=twap, vs정기bp=vs_f, 평균재고=inv_r, 정기평균재고=inv_f, 보유비용bp=hold_bp,
                vs정기_보유후bp=vs_f - hold_bp,
                vs균등bp=(twap - rule) / twap * 1e4, 카고=len(Bdf), 정기카고=len(fixed), 강제=forced, 미룸=deferred,
                신호비율=(Bdf["사유"] == "신호").mean() if len(Bdf) else np.nan, years=yr, 주수=len(Wv))


# ============================================================================
# 미루기 권고 (v1.9.0, 재고 무관): 매주 '지금 1척 살까, 미룰까'
#   점수 ≥ τ (가격 매력) → 지금 매수 / 아니고 P(2주 안에 지금보다 쌈) ≥ p* → 미룸 / 그 외 → 지금 매수
#   미룸: 'fixed' = 정확히 max_wait주 뒤 매수, 'adaptive' = 매주 재평가(매력적이 되거나 확률 < p* 이면 매수), 최대 max_wait주
#   기준 = 판단 없이 즉시 매수. 위약 = 같은 비율을 무작위로 미뤘을 때.
# ============================================================================
DEFER_WAIT = 2


def backtest_defer(W_: pd.DataFrame, fc: pd.DataFrame, tau: float | None = None, p_star: float = 0.6, max_wait: int = DEFER_WAIT,
                   mode: str = "adaptive", start="2008-01-01", end=None) -> pd.DataFrame:
    """주마다 한 번의 구매 판단. 반환 (판단 주별): 미룸 여부·기다린 주·즉시가·체결가·절감 bp (+ = 미뤄서 싸게 삼, 즉시 매수 주는 0).
    fc = fc_walkforward(W_, horizon=max_wait) — p_down = P(max_wait주 뒤 체결가 < 이번 주 체결가) (walk-forward)."""
    px = W_["px"].values; sc = W_["score"].values
    pdn = fc["p_down"].reindex(W_.index).values
    idx = W_.index
    lo = idx.searchsorted(pd.Timestamp(start)); hi = len(idx) - max_wait if end is None else min(len(idx) - max_wait, idx.searchsorted(pd.Timestamp(end), "right"))

    def buy_now(t):
        attractive = tau is not None and np.isfinite(sc[t]) and sc[t] >= tau
        return attractive or not (np.isfinite(pdn[t]) and pdn[t] >= p_star)

    rows = []
    for t in range(lo, hi):
        if not np.isfinite(px[t]) or not np.isfinite(pdn[t]):
            continue
        if buy_now(t):
            w = 0
        elif mode == "fixed":
            w = max_wait
        else:
            w = max_wait
            for k in range(1, max_wait):
                if buy_now(t + k):
                    w = k; break
        rows.append((idx[t], w > 0, w, px[t], px[t + w], (px[t] - px[t + w]) / px[t] * 1e4, pdn[t], sc[t],
                     (px[t] - px[t + max_wait]) / px[t] * 1e4))
    return pd.DataFrame(rows, columns=["date", "미룸", "대기주", "즉시가", "체결가", "절감bp", "p_down", "점수", "고정대기절감bp"]).set_index("date")


def defer_summary(D: pd.DataFrame, n_placebo: int = 500, seed: int = 0) -> dict:
    """미루기 성과 + 위약(같은 비율을 무작위 주에 고정 max_wait주 미룸) 비교. 절감은 '판단 1회당 평균'(즉시 매수 주 = 0 포함)."""
    if not len(D):
        return {}
    d = D["미룸"].values; s = D["절감bp"].values; g = D["고정대기절감bp"].values
    n, k = len(D), int(d.sum())
    rng = np.random.default_rng(seed)
    pl = np.array([g[rng.choice(n, k, replace=False)].sum() / n for _ in range(n_placebo)]) if 0 < k < n else np.zeros(1)
    yr = D.groupby(D.index.year)["절감bp"].mean()
    return {"판단 수": n, "미룸 비율": k / n, "판단당 평균 절감 (bp)": float(s.mean()),
            "미룬 경우 평균 (bp)": float(s[d].mean()) if k else np.nan, "미룬 경우 적중률": float((s[d] > 0).mean()) if k else np.nan,
            "위약 평균 (bp)": float(pl.mean()), "위약 90% (bp)": float(np.quantile(pl, 0.9)), "위약 초과 확률": float((pl < s.mean()).mean()),
            "무조건 2주 미룸 (bp)": float(g.mean()),
            "연도 t": float(yr.mean() / yr.std() * np.sqrt(len(yr))) if len(yr) > 2 and yr.std() > 0 else np.nan}


def summarize(B: pd.DataFrame, weeks: int, step: int, halflife_years: float | None = None) -> dict:
    nov = B.iloc[::max(1, weeks // step)]                                # 겹치지 않는 기간만으로 t값
    out = {}
    if halflife_years:
        age = np.asarray((B.index.max() - B.index).days, float) / 365.25
        wt = pd.Series(0.5 ** (age / halflife_years), index=B.index)
    for c in ["규칙", "가격위치만", "일괄", "사후최저"]:
        x, xn = B[f"절감_{c}"], nov[f"절감_{c}"]
        out[c] = {"평균 절감 (bp)": x.mean(), "중앙값 (bp)": x.median(), "균등보다 싸게 산 비율": (x > 0).mean(),
                  "하위 5% (bp)": x.quantile(0.05), "t값 (비중첩)": xn.mean() / xn.std() * np.sqrt(len(xn)) if xn.std() > 0 else np.nan,
                  "기간 수": len(x)}
        if halflife_years:
            out[c]["최근 가중 평균 (bp)"] = float((x * wt).sum() / wt.sum())
    return out


def band_4w(p_adj: pd.Series, sig: pd.DataFrame, last, h: int = 20) -> tuple[float, float]:
    """향후 h영업일 가격의 90% 범위 배수 (하단, 상단). 정규 ±1.645σ 대신 '표준화 h일 수익률의 경험적 5%·95% 분위'
    (그 시점까지 실현이 끝난 표본만) → 꼬리가 두꺼운 품목(철광석 등)의 범위가 실제 적중률 90%에 가깝게 넓어짐."""
    sd = sig["vol_ann"] / np.sqrt(252) * 100
    qlo, qhi = RK.hday_band(RK.returns(p_adj), sd, h, asof=last)
    s = float(sd.loc[last]) * np.sqrt(h) / 100
    return float(np.exp(qlo * s)), float(np.exp(qhi * s))


def today_table(La: pd.DataFrame, Lr: pd.DataFrame, ids: list[str], w: dict | None = None, k: float = 1.0, m_max: float = 3.0,
                weeks_left: int = 13, risk: str = "ewma", season: str = "all", lookback: int = 252, rel_mode: str = "price") -> pd.DataFrame:
    rows = []
    for i in ids:
        if i not in La or i not in Lr:
            continue
        inp = inputs(La, Lr, i, risk, rel_mode)
        sig = signals(La[i], lookback, inp["rel"], inp["rel_sign"], inp["vol_fc"], inp["crisis"], season)
        sc = score(sig, w)
        last = sig.dropna(subset=["value"]).index.max()
        pa = usd_t(Lr, i)
        px = float(pa.loc[:last].iloc[-1])
        p1 = pa.loc[last - pd.DateOffset(years=1):last]
        rk = float(sig["risk"].loc[last])
        mult = pace(float(sc.loc[last]), rk, 1.0, weeks_left, k, m_max) / (1.0 / weeks_left)
        b_lo, b_hi = band_4w(La[i], sig, last)
        va = float(sig["vol_ann"].loc[last])
        rows.append({"품목": BUY_IDS.get(i, i), "id": i, "기준일": last.date(), "단위": unit_of(i), "가격": px,
                     "1년 최저": p1.min(), "1년 최고": p1.max(),
                     "매수 점수": float(sc.loc[last]) * 100, **{COMP_LBL[c]: float(sig[c].loc[last]) * 100 for c in COMP_LBL},
                     "위험 지표": rk * 100, "변동성 분위": float(sig["vol_pct"].loc[last]) * 100, "연 변동성": va,
                     "4주 하단 90%": px * b_lo, "4주 상단 90%": px * b_hi,
                     "권장 속도 (균등 대비 배)": mult, "상대가치 지표": REL_DEF.get(i, ("없음 (검증 통과 관계 없음)",))[0]})
    return pd.DataFrame(rows)


# ============================================================================
# 펀더멘털 맥락 (점수에는 넣지 않음 — 기관 리포트·COT는 발표 후 예측력이 재현되지 않았음, 참고 정보)
# ============================================================================
WASDE_MAP = {"yf:ZC=F": "corn", "yf:ZS=F": "soybean", "yf:ZM=F": "soymeal", "yf:ZL=F": "soyoil", "yf:ZW=F": "wheat", "yf:KE=F": "wheat"}
COT_NAME = {"yf:ZC=F": "Corn", "yf:ZS=F": "Soybeans", "yf:ZM=F": "Soybean Meal", "yf:ZL=F": "Soybean Oil", "yf:ZW=F": "Wheat SRW", "yf:KE=F": "Wheat HRW",
            "yf:HG=F": "Copper"}


def fundamentals(ids: list[str]) -> pd.DataFrame:
    from pathlib import Path
    root = Path(__file__).parent / "data"
    W = pd.read_parquet(root / "reports" / "wasde_features.parquet") if (root / "reports" / "wasde_features.parquet").exists() else None
    C = pd.read_parquet(root / "cot.parquet") if (root / "cot.parquet").exists() else None
    rows = []
    for i in ids:
        r = {"품목": BUY_IDS.get(i, i)}
        if W is not None and i in WASDE_MAP:
            g = W[W.commodity == WASDE_MAP[i]].sort_values("release")
            if len(g):
                last = g.iloc[-1]
                hist = g[(g.release.dt.month == last.release.month) & (g.release < last.release)]["stu"]
                r.update({"WASDE 발표일": last.release.date(), "연산연도": last.my, "재고/사용비율 (%)": last.stu,
                          "전월 대비 (%p)": last.get("d_stu", np.nan),
                          "같은 달 과거 분위": float((hist < last.stu).mean()) if len(hist) else np.nan,
                          "USDA 가격 전망 수정 (%)": last.get("d_price", np.nan)})
        if C is not None and i in COT_NAME:
            c = C[C.name == COT_NAME[i]].sort_values("date")
            if len(c):
                mm = c.mm_net / c.open_interest_all * 100
                z = (mm - mm.rolling(156, min_periods=52).mean()) / mm.rolling(156, min_periods=52).std()
                r.update({"COT 기준일": c.date.iloc[-1].date() if hasattr(c.date.iloc[-1], "date") else c.date.iloc[-1],
                          "펀드 순포지션 (% OI)": float(mm.iloc[-1]), "펀드 포지션 z (3년)": float(z.iloc[-1])})
        rows.append(r)
    return pd.DataFrame(rows).set_index("품목")
