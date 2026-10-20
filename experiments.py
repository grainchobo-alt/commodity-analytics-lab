"""비교 실험 모음 — ⑥ 실험 결과 탭이 읽는 CSV를 만들고, 실행 기록(시각·소요시간·설정)을 data/experiments_meta.json에 남김.
모든 실험은 백테스트 엔진(portfolio.backtest·procurement.backtest)을 그대로 사용 → 미래정보 차단·비용 가정 동일."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
META = DATA / "experiments_meta.json"
KEEP = ["Sharpe (비용 후)", "Sharpe (비용 전)", "연환산 수익률(CAGR)", "연환산 변동성", "최대낙폭(MDD)", "연간 회전율(편도 합)",
        "연간 비용 부담", "연간 리밸런싱 횟수"]
STRATS = ["최적화(Newton)", "리스크 패리티(Newton)", "리스크 버짓(신호 기울기)", "동일가중 1/N"]


def _data():
    import analytics as A
    import portfolio as P
    cat = A.load_catalog()
    U = dict(zip(cat.id, cat.unit))
    La = A.load_levels(roll_adjusted=True); Lr = A.load_levels(roll_adjusted=False)
    return La, Lr, U, [i for i in P.DEFAULT_UNIVERSE if i in La]


def _save(name: str, df: pd.DataFrame, t0: float, settings: dict):
    df.to_csv(DATA / EXPERIMENTS[name]["file"], index=False, encoding="utf-8-sig")
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    meta[name] = dict(run_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), seconds=round(time.time() - t0), settings=settings)
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def _rows(res: dict, base: dict) -> list[dict]:
    out = []
    for k in STRATS:
        if k in res:
            out.append(dict(base, 전략=k, **{c: res[k]["stats"].get(c, np.nan) for c in KEEP}))
    return out


def momentum_lookback(progress=None, start="2010-01-01"):
    import portfolio as P
    t0 = time.time(); La, Lr, U, ids = _data(); rows = []
    jobs = [(m, s) for m in ["long_short", "long_only"] for s in P.MOM_SPECS]
    for j, (mode, spec) in enumerate(jobs):
        r = P.backtest(La, Lr, ids, U, "W-FRI", start=start, signal="momentum", mode=mode, w_max=0.25, mom_spec=spec)["results"]
        rows += _rows(r, dict(모드=mode, 모멘텀=spec))
        if progress: progress((j + 1) / len(jobs))
    _save("momentum", pd.DataFrame(rows), t0, dict(데이터="주간", 리밸런싱="주간", 기준="ETF", 시작=start, 종목상한=0.25))


def trade_rules(progress=None, start="2010-01-01"):
    import portfolio as P
    t0 = time.time(); La, Lr, U, ids = _data(); rows = []
    cases = [("고정 매일", dict(rebalance="B")), ("고정 매주", dict(rebalance="W-FRI")), ("고정 매월", dict(rebalance="ME")),
             ("무거래 5% (최대 21일)", dict(rebalance="B", trade_rule="band", band=0.05, max_hold=21)),
             ("무거래 10% (최대 21일)", dict(rebalance="B", trade_rule="band", band=0.10, max_hold=21)),
             ("무거래 20% (최대 63일)", dict(rebalance="B", trade_rule="band", band=0.20, max_hold=63)),
             ("부분조정 κ=0.3", dict(rebalance="B", trade_rule="partial", kappa=0.3)),
             ("부분조정 κ=0.1", dict(rebalance="B", trade_rule="partial", kappa=0.1))]
    for j, (nm, kw) in enumerate(cases):
        r = P.backtest(La, Lr, ids, U, "B", start=start, signal="momentum", mode="long_short", w_max=0.25, **kw)["results"]
        rows += _rows(r, dict(규칙=nm))
        if progress: progress((j + 1) / len(cases))
    _save("rules", pd.DataFrame(rows), t0, dict(데이터="일간", 신호="모멘텀 12-1", 제약="롱숏", 기준="ETF", 시작=start))


def vol_sizing(progress=None, start="2010-01-01"):
    import portfolio as P
    t0 = time.time(); La, Lr, U, ids = _data(); rows = []
    jobs = [(m, v, t) for m in ["long_short", "long_only"] for v in ["lw", "EWMA", "HAR-RV", "MIDAS-RV"] for t in [None, 0.10]]
    for j, (mode, vm, vt) in enumerate(jobs):
        r = P.backtest(La, Lr, ids, U, "B", start=start, signal="momentum", mode=mode, w_max=0.25, rebalance="W-FRI",
                       vol_model=vm, vol_target=vt, max_lev=3.0)["results"]
        for k in STRATS:
            ret = r[k]["ret"]; rv = ret.rolling(63).std() * np.sqrt(252)
            rows.append(dict(모드=mode, 변동성모형={"lw": "과거 창(LW)"}.get(vm, vm), 목표=("10%" if vt else "없음"), 전략=k,
                             **{c: r[k]["stats"].get(c, np.nan) for c in KEEP}, 변동성의변동성=float(rv.std()),
                             목표이탈=float((rv - 0.10).abs().mean()) if vt else np.nan))
        if progress: progress((j + 1) / len(jobs))
    _save("vol", pd.DataFrame(rows), t0, dict(데이터="일간", 리밸런싱="주간", 신호="모멘텀 12-1", 기준="ETF", 시작=start, 최대배율=3.0))


def risk_budget(progress=None, start="2010-01-01"):
    import portfolio as P
    t0 = time.time(); La, Lr, U, ids = _data(); rows = []
    sets = {"주간·모멘텀·롱숏": dict(freq="W-FRI", signal="momentum", mode="long_short"),
            "주간·모멘텀·롱온리": dict(freq="W-FRI", signal="momentum", mode="long_only"),
            "주간·Ridge·롱숏": dict(freq="W-FRI", signal="ridge", mode="long_short"),
            "일간·모멘텀·롱숏·무거래20%": dict(freq="B", signal="momentum", mode="long_short", rebalance="B", trade_rule="band", band=0.2, max_hold=63)}
    jobs = [(nm, t) for nm in sets for t in [0.0, 0.25, 0.5, 0.75, 1.0]]
    for j, (nm, tilt) in enumerate(jobs):
        kw = dict(sets[nm]); fq = kw.pop("freq")
        r = P.backtest(La, Lr, ids, U, fq, start=start, w_max=0.25, rb_tilt=tilt, **kw)["results"]
        rows += _rows(r, dict(설정=nm, τ=tilt))
        if "DBC 보유 (투자가능 지수)" in r:
            rows.append(dict(설정=nm, τ=tilt, 전략="DBC 보유", **{c: r["DBC 보유 (투자가능 지수)"]["stats"].get(c, np.nan) for c in KEEP}))
        if progress: progress((j + 1) / len(jobs))
    _save("rb", pd.DataFrame(rows), t0, dict(기준="ETF", 시작=start, 종목상한=0.25))


def procurement(progress=None, start="2008-01-01"):
    import procurement as PR
    t0 = time.time(); La, Lr, U, _ = _data(); rows = []
    combos = dict(PR.PRESETS)
    combos.update({"추세만 (참고)": dict(w={"trend": 1.0}, k=1.0, risk="ewma", lookback=252, season="all"),
                   "가격 위치 (k=1)": dict(w={"value": 1.0}, k=1.0, risk="ewma", lookback=252, season="all")})
    cache = {}
    ids = ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:KE=F", "yf:ZL=F"]
    jobs = [(i, nm, wk) for i in ids for nm in combos for wk in (13, 26)]
    for j, (i, nm, wk) in enumerate(jobs):
        P_ = combos[nm]
        if (i, P_["risk"]) not in cache:
            cache[(i, P_["risk"])] = PR.inputs(La, Lr, i, P_["risk"], "price")
        B = PR.backtest(Lr[i], La[i], wk, 4, start=start, w=P_["w"], k=P_["k"], lookback=P_["lookback"],
                        inp=cache[(i, P_["risk"])], season=P_["season"])
        for per, msk in [("전체", B.index >= start), ("2008-2016", B.index < "2017-01-01"), ("2017-현재", B.index >= "2017-01-01")]:
            x = B.loc[msk, "절감_규칙"]
            rows.append(dict(품목=PR.BUY_IDS[i], 신호=nm, 구매기간주=wk, 기간=per, 평균절감bp=x.mean(), 중앙값bp=x.median(),
                             균등보다싸게=(x > 0).mean(), 하위5bp=x.quantile(0.05), 사후최저bp=B.loc[msk, "절감_사후최저"].mean(),
                             일괄매수bp=B.loc[msk, "절감_일괄"].mean(), 기간수=int(msk.sum())))
        if progress: progress((j + 1) / len(jobs))
    _save("procure", pd.DataFrame(rows), t0, dict(시작=start, 시작간격="4주", 체결="결정일(금) 다음 거래일 실제 종가"))


def procurement_steps(progress=None, start="2008-01-01", weeks: int = 13):
    """구매 신호 고도화 단계별 검증: 기준(균형 프리셋) 대비 한 가지씩 바꿨을 때 균등 매수 대비 절감이 개선되는가."""
    import procurement as PR
    t0 = time.time(); La, Lr, U, _ = _data(); rows = []
    base_w = dict(PR.DEFAULT_W)
    V = [  # (단계, 변형 이름, 가중치, k, 위험, 상대가치 모드, 계절, lookback, 보유비용)
        ("0 기준", "균형 (위치+반등), EWMA 위험", base_w, 1.0, "ewma", "price", "all", 252, 0.0),
        ("1 위험 판단", "MIDAS 변동성 예측 위험", base_w, 1.0, "midas", "price", "all", 252, 0.0),
        ("1 위험 판단", "HMM 위기 확률 위험", base_w, 1.0, "hmm", "price", "all", 252, 0.0),
        ("1 위험 판단", "공격적 k=2 + EWMA", base_w, 2.0, "ewma", "price", "all", 252, 0.0),
        ("1 위험 판단", "공격적 k=2 + HMM", base_w, 2.0, "hmm", "price", "all", 252, 0.0),
        ("2 상대가치", "+ 상대가치 25% (실제 시세 관계)", {"value": .45, "turn": .30, "relval": .25}, 1.0, "ewma", "price", "all", 252, 0.0),
        ("2 상대가치", "+ 상대가치 25% (롤보정 상대성과)", {"value": .45, "turn": .30, "relval": .25}, 1.0, "ewma", "adj", "all", 252, 0.0),
        ("2 상대가치", "상대가치만 (단독 효과)", {"relval": 1.0}, 1.0, "ewma", "price", "all", 252, 0.0),
        ("2 상대가치", "+ 상대가치 50%", {"value": .3, "turn": .2, "relval": .5}, 1.0, "ewma", "price", "all", 252, 0.0),
        ("4 계절성·창", "+ 계절 25% (전체 연도)", {"value": .45, "turn": .30, "season": .25}, 1.0, "ewma", "price", "all", 252, 0.0),
        ("4 계절성·창", "+ 계절 25% (최근 10년)", {"value": .45, "turn": .30, "season": .25}, 1.0, "ewma", "price", "10y", 252, 0.0),
        ("4 계절성·창", "+ 계절 25% (최근 가중)", {"value": .45, "turn": .30, "season": .25}, 1.0, "ewma", "price", "ewm", 252, 0.0),
        ("4 계절성·창", "가격 위치 창 6개월", base_w, 1.0, "ewma", "price", "all", 126, 0.0),
        ("4 계절성·창", "가격 위치 창 2년", base_w, 1.0, "ewma", "price", "all", 504, 0.0),
        ("4 보유비용", "기준 + 보유비용 3bp/주", base_w, 1.0, "ewma", "price", "all", 252, 3.0),
    ]
    ids = PR.GRAIN_IDS                                                   # 2008~ 표본이 있는 곡물 6품목 (금속은 procurement_metals)
    cache = {}
    jobs = [(i, v) for v in V for i in ids]
    for j, (i, (stg, nm, w, k, risk, relm, sea, lb, carry)) in enumerate(jobs):
        key = (i, risk, relm)
        if key not in cache:
            cache[key] = PR.inputs(La, Lr, i, risk, relm)
        B = PR.backtest(Lr[i], La[i], weeks, 4, start=start, w=w, k=k, lookback=lb, inp=cache[key], season=sea, carry_bp_week=carry)
        for dt, r in B.iterrows():
            rows.append(dict(단계=stg, 변형=nm, 품목=PR.BUY_IDS[i], start=dt, 절감bp=r["절감_규칙"], 일괄bp=r["절감_일괄"],
                             시작위험=r["시작위험"]))
        if progress: progress((j + 1) / len(jobs))
    D = pd.DataFrame(rows)
    D.to_parquet(DATA / "procurement_steps_raw.parquet")
    out = []
    for (stg, nm), g in D.groupby(["단계", "변형"], sort=False):
        per = g.groupby("start")["절감bp"].mean()                        # 기간별 품목 평균
        nov = per.iloc[::max(1, weeks // 4)]
        rec = per[per.index >= "2017-01-01"]; old = per[per.index < "2017-01-01"]
        hv = g[g.시작위험 >= 0.7].groupby("start")["절감bp"].mean(); lv = g[g.시작위험 < 0.7].groupby("start")["절감bp"].mean()
        out.append(dict(단계=stg, 변형=nm, 평균절감bp=g.절감bp.mean(), 균등보다싸게=(g.절감bp > 0).mean(), 하위5bp=g.절감bp.quantile(0.05),
                        t값_비중첩=nov.mean() / nov.std() * np.sqrt(len(nov)), 전반_2008_16=old.mean(), 후반_2017_현재=rec.mean(),
                        고변동국면=hv.mean(), 저변동국면=lv.mean()))
    _save("procure_steps", pd.DataFrame(out), t0, dict(시작=start, 구매기간=f"{weeks}주", 품목=len(ids), 시작간격="4주"))


def _metal_candidates(Lr: pd.DataFrame) -> dict:
    """철강 원료·구리의 상대가치 후보 (실제 근월물 시세의 경제적 관계). 값: (관계 시계열, 부호 +1=클수록 내 품목이 쌈)."""
    import procurement as PR
    g = lambda k: Lr[k].dropna()
    I0, J0, JM, RB = g("ak:I0"), g("ak:J0"), g("ak:JM0"), g("ak:RB0")
    return {"yf:TIO=F": {"SGX/DCE 수입 차익 (채택)": (PR.rel_series(Lr, "yf:TIO=F"), -1),
                         "철근 고로 마진": (RB - 1.6 * I0 - 0.5 * J0, +1),
                         "철광석/철근 가격비": (np.log(I0) - np.log(RB), -1),
                         "철광석/점결탄 가격비": (np.log(I0) - np.log(JM), -1)},
            "ak:JM0": {"철근 고로 마진 (채택)": (PR.rel_series(Lr, "ak:JM0"), +1),
                       "코크스 마진": (J0 - 1.33 * JM, +1),
                       "점결탄/코크스 가격비": (np.log(JM) - np.log(J0), -1)},
            "yf:HG=F": {"구리/금 가격비 (채택)": (PR.rel_series(Lr, "yf:HG=F"), -1),
                        "구리/알루미늄 가격비": (np.log(g("yf:HG=F") * 2204.62) - np.log(g("yf:ALI=F")), -1)}}


def procurement_metals(progress=None, split="2019-07-01"):
    """철광석·점결탄·구리: '추천' 프리셋에서 상대가치 정의만 바꿔 비교 → 전반·후반 모두 + 인 정의만 채택.
    데이터가 2010~2013년부터라 곡물(2008~)과 따로, 시작 2013(점결탄 2015) · 전반/후반 경계 split."""
    import procurement as PR
    t0 = time.time(); La, Lr, U, _ = _data(); rows = []
    P_ = PR.PRESETS["추천 (고도화: 상대가치+계절+MIDAS)"]
    C = _metal_candidates(Lr)
    jobs = [(i, nm, wk) for i in PR.METAL_IDS for nm in ["상대가치 없음", *C[i]] for wk in (8, 13, 26)]
    vf = {}
    for j, (i, nm, wk) in enumerate(jobs):
        if i not in vf:
            vf[i] = PR.midas_var(La[i])
        rel, sg = C[i].get(nm, (None, 1))
        if rel is not None:
            rel = rel.dropna(); rel = rel[rel.index.dayofweek < 5]
        st_ = "2015-01-01" if i == "ak:JM0" else "2013-01-01"
        B = PR.backtest(Lr[i], La[i], wk, 4, start=st_, w=P_["w"], k=P_["k"], lookback=P_["lookback"],
                        inp=dict(rel=rel, rel_sign=sg, vol_fc=vf[i]), season=P_["season"])
        x = B["절감_규칙"]; nov = x.iloc[::max(1, wk // 4)]
        rows.append(dict(품목=PR.BUY_IDS[i], 상대가치=nm, 구매기간주=wk, 평균절감bp=x.mean(), 중앙값bp=x.median(),
                         t값_비중첩=nov.mean() / nov.std() * np.sqrt(len(nov)), 전반bp=x[x.index < split].mean(), 후반bp=x[x.index >= split].mean(),
                         균등보다싸게=(x > 0).mean(), 하위5bp=x.quantile(0.05), 기간수=len(x)))
        if progress: progress((j + 1) / len(jobs))
    _save("procure_metals", pd.DataFrame(rows), t0, dict(프리셋="추천", 시작="2013 (점결탄 2015)", 전후반경계=split, 시작간격="4주"))


def procurement_all(progress=None, weeks=(13, 26)):
    """전 품목 구매 타이밍: '추천' 프리셋에서 상대가치만 바꿈 — 없음 / 후보 단일 관계들 / 다변량 2종(후보 분위 평균 · 공적분 잔차).
    품목마다 평가 시작 = max(2008, 모든 관계 자료가 4년 쌓인 시점) → 같은 품목 안의 선택지는 같은 기간으로 비교.
    전반·후반 = 그 품목 평가 기간의 중간 날짜로 2등분. 철광석(SGX)·점결탄은 procurement_metals 에서 별도 검증."""
    import procurement as PR
    t0 = time.time(); La, Lr, U, _ = _data(); rows = []
    P_ = PR.PRESETS["추천 (고도화: 상대가치+계절+MIDAS)"]
    ids = [i for i in PR.BUY_IDS if i in La and i in Lr and i not in ("yf:TIO=F", "ak:JM0")]
    for j, i in enumerate(ids):
        opts = ["상대가치 없음"] + [c[0] for c in PR.REL_CANDS.get(i, [])]
        if sum(c[1] == "r" for c in PR.REL_CANDS.get(i, [])) >= 2:
            opts += [PR.MULTI_AVG, PR.MULTI_COINT]
        rels = {o: PR.rel_option(Lr, i, o, La) for o in opts[1:]}
        own = np.log(Lr[i].dropna()); rels[OWN_CTRL] = (own[own.index.dayofweek < 5], -1)   # 대조군: 대체재 정보 없이 자기 가격만
        rels = {o: (r_.dropna(), sg) for o, (r_, sg) in rels.items() if r_ is not None and len(r_.dropna()) > 500}
        firsts = [La[i].first_valid_index() + pd.DateOffset(years=4)] + [r_.index[0] + pd.DateOffset(years=2) for r_, _ in rels.values()]
        st_ = max([pd.Timestamp("2008-01-01")] + firsts)
        if st_ > pd.Timestamp.today() - pd.DateOffset(years=6):
            if progress: progress((j + 1) / len(ids))
            continue
        vf = PR.midas_var(La[i])
        for o in ["상대가치 없음"] + list(rels):
            rel, sg = rels.get(o, (None, 1))
            for wk in weeks:
                B = PR.backtest(Lr[i], La[i], wk, 4, start=st_, w=P_["w"], k=P_["k"], lookback=P_["lookback"],
                                inp=dict(rel=rel, rel_sign=sg, vol_fc=vf), season=P_["season"])
                x = B["절감_규칙"]; nov = x.iloc[::max(1, wk // 4)]
                mid = x.index[0] + (x.index[-1] - x.index[0]) / 2
                rows.append(dict(품목=PR.BUY_IDS[i], id=i, 상대가치=o, 종류=("없음" if o == "상대가치 없음" else "대조군" if o == OWN_CTRL else
                                                                       "다변량" if o.startswith("다변량") else "단일"),
                                 구매기간주=wk, 평균절감bp=x.mean(), t값_비중첩=nov.mean() / nov.std() * np.sqrt(len(nov)),
                                 전반bp=x[x.index < mid].mean(), 후반bp=x[x.index >= mid].mean(), 하위5bp=x.quantile(0.05),
                                 균등보다싸게=(x > 0).mean(), 평가시작=str(st_.date()), 경계=str(mid.date()), 기간수=len(x)))
        if progress: progress((j + 1) / len(ids))
    df = pd.DataFrame(rows)
    base = df[df.종류 == "없음"].set_index(["id", "구매기간주"])
    for c in ["평균절감bp", "전반bp", "후반bp"]:
        df[f"개선_{c[:-2]}"] = df[c].values - base[c].reindex(pd.MultiIndex.from_frame(df[["id", "구매기간주"]])).values
    _save("procure_all", df, t0, dict(프리셋="추천 (상대가치 정의만 변경)", 구매기간=", ".join(f"{w}주" for w in weeks), 시작간격="4주",
                                       평가시작="max(2008, 자료 4년 + 관계 2년)", 전후반="품목별 평가 기간 중간"))


# 위약용 '경제적 연결이 없는' 상대 품목 풀 (각 품목과 다른 섹터에서 고름)
_SECTOR = {"yf:ZC=F": "곡물", "yf:ZW=F": "곡물", "yf:KE=F": "곡물", "yf:ZS=F": "곡물", "yf:ZM=F": "곡물", "yf:ZL=F": "곡물", "yf:ZO=F": "곡물", "yf:ZR=F": "곡물",
           "yf:CL=F": "에너지", "yf:BZ=F": "에너지", "yf:HO=F": "에너지", "yf:RB=F": "에너지", "yf:NG=F": "에너지",
           "yf:GC=F": "금속", "yf:SI=F": "금속", "yf:PL=F": "금속", "yf:PA=F": "금속", "yf:ALI=F": "금속", "yf:HG=F": "금속", "yf:TIO=F": "금속",
           "yf:CC=F": "소프트", "yf:KC=F": "소프트", "yf:CT=F": "소프트", "yf:SB=F": "소프트", "yf:OJ=F": "소프트",
           "yf:LE=F": "축산", "yf:GF=F": "축산", "yf:HE=F": "축산",
           "ak:A0": "곡물", "ak:C0": "곡물", "ak:M0": "곡물", "ak:Y0": "곡물", "ak:I0": "금속", "ak:J0": "금속", "ak:RB0": "금속", "ak:JM0": "금속"}
OWN_CTRL = "대조군: 자기 가격 1년 위치"


def rel_adoption(df_all: pd.DataFrame, df_pl: pd.DataFrame | None = None, min_gain: float = 5.0) -> pd.DataFrame:
    """상대가치 선택지별 판정표. 개선 = '상대가치 없음' 대비 bp (13·26주 평균). 엄격 채택 조건 (모두 충족):
    ① 13·26주 모두 전반·후반 개선 > 0  ② 개선 ≥ min_gain  ③ 그 품목 위약(무관한 품목과의 가격비) 개선의 90% 분위 초과
    ④ 대조군(자기 가격 1년 위치) 개선 초과 — 대체재 정보 없이 자기 가격만으로도 나오는 효과는 상대가치로 인정하지 않음."""
    d = df_all[df_all.종류 != "없음"]
    g = d.groupby(["품목", "id", "상대가치", "종류"]).agg(전반=("개선_전반", "min"), 후반=("개선_후반", "min"), 개선=("개선_평균절감", "mean")).reset_index()
    g["두기간통과"] = (g.전반 > 0) & (g.후반 > 0)
    own = g[g.종류 == "대조군"].set_index("id")["개선"]
    g["대조군개선"] = g.id.map(own)
    if df_pl is not None and len(df_pl):
        pl = df_pl[df_pl.종류 == "위약"].groupby(["id", "선택지"]).개선_평균절감.mean().groupby("id").quantile(0.9)
        g["위약90"] = g.id.map(pl)
    else:
        g["위약90"] = np.nan
    g["엄격채택"] = (g.종류.isin(["단일", "다변량"]) & g.두기간통과 & (g.개선 >= min_gain) & (g.개선 > g.위약90.fillna(np.inf))
                    & (g.개선 > g.대조군개선.fillna(-np.inf)))
    return g


def procurement_placebo(progress=None, n_placebo=10, weeks=(13, 26), seed=0):
    """'상대가치 없음' 판정 검증용 위약: 품목마다 경제적 연결이 없는 다른 섹터 품목과의 가격비(무작위 n개)를 상대가치로 넣어 같은 백테스트.
    → (1) 무의미한 관계도 우연히 '통과'(13·26주 모두 전·후반 개선)하는 비율 = 우연 통과율, (2) 위약 개선의 분포(95% 분위) —
    실제 후보가 이 기준을 넘는지로 '통과' 판정의 신뢰도를 봄. 평가 기간·경계는 procurement_all 과 동일. 부호 반전(−1 → +1)도 진단으로 기록."""
    import procurement as PR
    t0 = time.time(); La, Lr, U, _ = _data(); rows = []
    P_ = PR.PRESETS["추천 (고도화: 상대가치+계절+MIDAS)"]
    base = pd.read_csv(DATA / EXPERIMENTS["procure_all"]["file"], encoding="utf-8-sig")
    win = base.groupby("id")[["평가시작", "경계"]].first()
    rng = np.random.default_rng(seed)
    ids = [i for i in win.index if i in La and i in _SECTOR]
    for j, i in enumerate(ids):
        st_, mid = pd.Timestamp(win.loc[i, "평가시작"]), pd.Timestamp(win.loc[i, "경계"])
        pool = [k for k, sec in _SECTOR.items() if sec != _SECTOR[i] and k in Lr and Lr[k].first_valid_index() < st_ - pd.DateOffset(years=2)]
        pick = list(rng.choice(pool, min(n_placebo, len(pool)), replace=False))
        vf = PR.midas_var(La[i])
        opts = [("없음", None, 1, "없음")] + [(f"위약: /{k}", k, -1, "위약") for k in pick] + \
               [(c[0], None, -c[3], "부호 반전") for c in PR.REL_CANDS.get(i, []) if c[3] != 0]
        for nm, k, sg, kind in opts:
            if kind == "위약":
                rel = PR.cand_series(Lr, i, ("", "r", k, -1))
            elif kind == "부호 반전":
                rel = PR.rel_option(Lr, i, nm, La)[0]
            else:
                rel = None
            for wk in weeks:
                B = PR.backtest(Lr[i], La[i], wk, 4, start=st_, w=P_["w"], k=P_["k"], lookback=P_["lookback"],
                                inp=dict(rel=rel, rel_sign=sg, vol_fc=vf), season=P_["season"])
                x = B["절감_규칙"]
                rows.append(dict(품목=PR.BUY_IDS[i], id=i, 선택지=nm, 종류=kind, 구매기간주=wk, 평균절감bp=x.mean(),
                                 전반bp=x[x.index < mid].mean(), 후반bp=x[x.index >= mid].mean()))
        if progress: progress((j + 1) / len(ids))
    df = pd.DataFrame(rows)
    b0 = df[df.종류 == "없음"].set_index(["id", "구매기간주"])
    for c in ["평균절감bp", "전반bp", "후반bp"]:
        df[f"개선_{c[:-2]}"] = df[c].values - b0[c].reindex(pd.MultiIndex.from_frame(df[["id", "구매기간주"]])).values
    _save("procure_placebo", df, t0, dict(위약="다른 섹터 품목과의 가격비 (품목당 무작위 %d개)" % n_placebo, 구매기간=", ".join(f"{w}주" for w in weeks),
                                         평가기간="procurement_all 과 동일"))


RISK_IDS = ["yf:ZC=F", "yf:ZW=F", "yf:KE=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:TIO=F", "ak:JM0", "yf:HG=F", "yf:CL=F", "yf:NG=F", "yf:GC=F"]


def risk_models(progress=None, start="2010-01-01"):
    """◆ 리스크: 품목별 (1) 변동성 모형 경주 — QLIKE·DM(기준 RiskMetrics)·MCS, (2) VaR 백테스트 — 1일 99%·20일 95%, 롱·숏(매입 예정),
    (3) 구매 타이밍 4주 범위 적중률 — 정규 vs 경험적 분위. 모두 walk-forward, 평가 start~."""
    import risk as RK
    import procurement as PR
    t0 = time.time(); La, Lr, U, _ = _data(); rows = []
    ids = [i for i in RISK_IDS if i in La]
    for j, i in enumerate(ids):
        nm = PR.BUY_IDS.get(i, {"yf:CL=F": "WTI 원유", "yf:NG=F": "천연가스", "yf:GC=F": "금"}.get(i, i))
        r = RK.returns(La[i]); o = RK.vol_models(r); S = o["S"]
        Lo = RK.qlike(r, S)[S.index >= start].dropna()
        dm = RK.dm_table(Lo, "RiskMetrics λ=0.94"); pv = RK.mcs(Lo, B=500)
        for m in S.columns:
            rows.append(dict(구분="변동성", 품목=nm, 항목=m, QLIKE=Lo[m].mean(), DM_t=dm["DM t"].get(m, 0.0), MCS_p=pv[m],
                             평균반감기일=float(np.nanmean(RK.hl_days(o["lam"][m][o["lam"].index >= start]))) if m in o["lam"] else np.nan))
        for side in ["long", "short"]:
            for a_, h in [(0.99, 1), (0.95, 20)]:
                V = RK.var_forecasts(r, S["GARCH(1,1)"], a_, side, h, start=start)
                bt = RK.backtest_table(V)
                for m, b in bt.iterrows():
                    rows.append(dict(구분="VaR", 품목=nm, 항목=m, 포지션={"long": "보유 (하락 위험)", "short": "매입 예정 (상승 위험)"}[side],
                                     기준=f"{h}일 {a_:.0%}", 초과율=b["초과율"], 기대초과율=1 - a_, Kupiec_p=b["Kupiec p (적중률)"],
                                     Christoffersen_p=b["Christoffersen p (독립성)"], ES_p=b["ES 검정 p (과소추정)"]))
        if i in PR.BUY_IDS:
            inp = PR.inputs(La, Lr, i, "midas")
            sig = PR.signals(La[i], 126, inp["rel"], inp["rel_sign"], inp["vol_fc"], None, "ewm")
            cv = RK.band_coverage(r, sig["vol_ann"] / np.sqrt(252) * 100, 20, "2008-01-01")
            rows.append(dict(구분="4주 범위", 품목=nm, 항목="MIDAS σ (추천 프리셋)", 정규적중=cv["정규 적중률"], 경험적적중=cv["경험적 적중률"],
                             하단이탈=cv["하단 이탈률 (경험적)"], 상단이탈=cv["상단 이탈률 (경험적)"], n=cv["n"]))
        if progress: progress((j + 1) / len(ids))
    _save("risk", pd.DataFrame(rows), t0, dict(평가시작=start, 품목=len(ids), VaR_σ="GARCH(1,1)", MCS="T_max, 블록 10일, B=500"))


META_IDS = ["yf:ZC=F", "yf:ZS=F", "yf:ZW=F", "yf:ZM=F", "yf:ZL=F", "yf:HG=F"]
META_FEATS = ["yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "ak:macro_shipping_bdi",
              "yf:DX-Y.NYB", "fred:VIXCLS", "fred:DGS10", "yf:^GSPC"]          # ④ 탭 기본 설명변수
META_CORR = ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:ZL=F", "yf:CL=F", "yf:HG=F", "yf:GC=F"]


def model_selection(progress=None, oos_start="2004-01-01", eval_start="2008-01-01", B: int = 1000):
    """④ 모델 선택 메타 전략: 후보 6모델의 walk-forward OOS 경로 → (a) 직전 6개월 최소손실 (b) 지수가중 역MSE 결합
    (c) 레짐(HMM·실현변동성·평균 상관) 조건부 선택 (d) 동일가중 (e) 과거평균. 일간(재추정 126일)·주간(26주), 비용 5bp, 평가 2008~.
    위약 = 롱 비중·전환 빈도 일치 무작위 부호, 스누핑 보정 = White RC · Hansen SPA (전 13개 전략 동시)."""
    import analytics as A
    import metasel as MS
    t0 = time.time(); La, Lr, U, _ = _data()
    Rd = A.make_panel(La, [i for i in META_CORR if i in La], U, "B", "logret", "2000-01-01")
    rows, curves = [], []
    jobs = [(i, fq) for i in META_IDS if i in La for fq in ["B", "W-FRI"]]
    for j, (i, fq) in enumerate(jobs):
        feats = [f for f in META_FEATS if f != i and f in La]
        R = A.make_panel(La, [i] + feats, U, fq, "logret", "2000-01-01")
        D = A.build_design(R, i, feats, "predict", 3, 1, True, 0.7, xlag=1 if fq == "B" else 0)
        D.cost = 0.05
        o = MS.run(D, La, Rd, i, A.PPY[fq], 126 if fq == "B" else 26, oos_start=oos_start, eval_start=eval_start, B=B)
        fql = {"B": "일간", "W-FRI": "주간"}[fq]
        for s, r in o["table"].iterrows():
            d_ = r.to_dict(); rows.append(dict(구분="성과", 품목=i, 빈도=fql, 전략=s, 유형=d_.pop("구분"), **d_))
        pm = o["placebo_max"]
        rows.append(dict(구분="최댓값 위약", 품목=i, 빈도=fql, 전략=pm["최고전략"], **{"부호전략 Sharpe (비용 후)": pm["관측최댓값"],
                         "위약 중앙값": pm["위약최댓값중앙"], "위약 95% 분위": pm["위약최댓값95"], "위약 p (단일)": pm["p"]}))
        for _, r in o["spa"].iterrows():
            rows.append(dict(구분="SPA", 품목=i, 빈도=fql, 전략=r["최고 전략"], 기준=r["귀무가설 기준"], **{"최고 평균차": r["최고 평균차"],
                             "White RC p": r["White RC p"], "Hansen SPA p": r["Hansen SPA p"], "전략 수": r["전략 수"]}))
        C = MS.cum_net(o["preds"], o["y"], D.horizon, D.cost)
        C = C.stack().rename("누적").reset_index(); C.columns = ["date", "전략", "누적"]
        C["품목"] = i; C["빈도"] = fql
        curves.append(C)
        if progress: progress((j + 1) / len(jobs))
    pd.concat(curves).to_parquet(DATA / "model_selection_cum.parquet")
    _save("metasel", pd.DataFrame(rows), t0, dict(후보="OLS·Ridge·Lasso·GB·LightGBM·Kalman", 비용="5bp", OOS시작=oos_start,
                                                   평가시작=eval_start, 선택창="6개월", 반감기="6개월", 위약B=B))


def forecast_lab(progress=None):
    """최신 시계열 파운데이션 모델(Chronos-Bolt·Chronos-2·TimesFM 2.5·TiRex) vs 기준 모델 — 목표별(방향·변동성·스프레드·순위·구매)
    walk-forward 비교 + DSR·PBO·CV 누수 시연 + 메타 라벨링. 모델 예측은 data/fm/ 캐시 (첫 실행 약 1~2시간, 이후 수 분)."""
    import forecast_lab as FL
    t0 = time.time()
    steps = iter(range(1, 10 ** 6))
    out = FL.run(log=(lambda m: progress(min(0.95, next(steps) / 120)) if progress else print))
    df = pd.concat([v.assign(구분=k) for k, v in out.items()], ignore_index=True)
    _save("forecast_lab", df, t0, dict(평가시작=FL.EVAL_START, 결정일="매주 금요일 (TiRex 4주마다)", 컨텍스트=f"{FL.CTX}영업일", 예측기간=f"{FL.H}영업일",
                                     모델=", ".join(FL.FM_NAMES)))


EXPERIMENTS = {
    "procure": dict(title="구매 타이밍 — 신호·공격성별 평균 매입가 절감", file="procurement_compare.csv", fn=procurement, eta="약 1분",
                    cols=["품목", "신호", "구매기간주", "기간", "평균절감bp"]),
    "procure_steps": dict(title="구매 신호 고도화 — 단계별 추가 효과 검증", file="procurement_steps_compare.csv", fn=procurement_steps,
                          eta="약 5~10분", cols=["단계", "변형", "평균절감bp", "하위5bp"]),
    "procure_metals": dict(title="철강 원료·구리 — 상대가치 정의 비교", file="procurement_metals_compare.csv", fn=procurement_metals,
                           eta="약 1분", cols=["품목", "상대가치", "구매기간주", "전반bp", "후반bp"]),
    "procure_all": dict(title="전 품목 구매 타이밍 — 단일 vs 다변량 상대가치", file="procurement_all_compare.csv", fn=procurement_all,
                        eta="약 3~6분", cols=["품목", "상대가치", "구매기간주", "전반bp", "후반bp"]),
    "procure_placebo": dict(title="구매 상대가치 위약 검증 — 무의미한 관계의 우연 통과율", file="procurement_placebo_compare.csv",
                            fn=procurement_placebo, eta="약 5~10분", cols=["품목", "선택지", "종류", "구매기간주"]),
    "metasel": dict(title="모델 선택 메타 전략 — 시간·레짐별 ML 모델 선택 (위약·SPA 보정)", file="model_selection_compare.csv",
                    fn=model_selection, eta="약 8~12분", cols=["구분", "품목", "빈도", "전략"]),
    "risk": dict(title="리스크 — 변동성 모형 경주 · VaR 백테스트 · 4주 범위 적중률", file="risk_models_compare.csv", fn=risk_models,
                 eta="약 3~5분", cols=["구분", "품목", "항목"]),
    "forecast_lab": dict(title="최신 시계열 모델 · 목표별 예측력 · 과적합 판정 · 메타 라벨링", file="forecast_lab_compare.csv", fn=forecast_lab,
                         eta="첫 실행 1~2시간 (캐시 후 수 분)", cols=["구분", "모델"]),
    "momentum": dict(title="모멘텀 기간 비교", file="momentum_lookback_compare.csv", fn=momentum_lookback, eta="약 5~8분",
                     cols=["모드", "모멘텀", "전략", "Sharpe (비용 후)", "연간 회전율(편도 합)"]),
    "rules": dict(title="거래 규칙 비교 (고정 주기 · 무거래 구간 · 부분 조정)", file="rebalance_rule_compare.csv", fn=trade_rules, eta="약 7~10분",
                  cols=["규칙", "전략", "Sharpe (비용 후)", "연간 회전율(편도 합)", "연환산 변동성"]),
    "vol": dict(title="변동성 예측 → 포지션 크기 (변동성 타기팅)", file="vol_sizing_compare.csv", fn=vol_sizing, eta="약 5분",
                cols=["모드", "변동성모형", "목표", "전략", "Sharpe (비용 후)", "연환산 변동성", "변동성의변동성"]),
    "rb": dict(title="전략 비교 — 리스크 버짓(신호 기울기 τ) vs 리스크 패리티 vs 최적화", file="risk_budget_tilt_compare.csv", fn=risk_budget, eta="약 10분",
               cols=["설정", "τ", "전략", "Sharpe (비용 후)"]),
}


def load(name: str) -> tuple[pd.DataFrame | None, dict]:
    f = DATA / EXPERIMENTS[name]["file"]
    meta = json.loads(META.read_text(encoding="utf-8")).get(name, {}) if META.exists() else {}
    if not f.exists():
        return None, meta
    df = pd.read_csv(f, encoding="utf-8-sig")
    if not set(EXPERIMENTS[name].get("cols", [])) <= set(df.columns):     # 이전 형식(구버전 스크립트) → 재실행 필요
        return None, dict(meta, stale=True)
    if not meta:
        meta = dict(run_at=pd.Timestamp(f.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M") + " (파일 시각)")
    return df, meta


if __name__ == "__main__":
    import sys
    for n in (sys.argv[1:] or list(EXPERIMENTS)):
        print(n, end=" ", flush=True); t = time.time(); EXPERIMENTS[n]["fn"](); print(f"{time.time() - t:.0f}s")
