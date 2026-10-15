"""data/*.parquet -> dashboard.html (자체 완결형 HTML, Plotly CDN)"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import analytics as A

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
D = ROOT / "data"
daily = pd.read_parquet(A._daily_path())   # 정제본 우선
monthly = pd.read_parquet(D / "monthly.parquet")
cot = pd.read_parquet(D / "cot.parquet") if (D / "cot.parquet").exists() else pd.DataFrame()
status = pd.read_csv(D / "status.csv")

ASOF = daily[(daily.source == "Yahoo") & (daily.group != "FX")].date.max()
DASH_START = pd.Timestamp("1986-01-01")   # 대시보드 차트용 (금리·달러는 1960~70년대부터 있으나 원자재 비교 구간으로 제한)
GROUP_ORDER = ["Energy", "Metals", "Iron Ore / Coal", "Grains", "Softs", "Livestock", "Freight", "China (DCE)", "Macro", "FX", "FRED daily"]

# ---- Kalman 로컬 선형 추세 (placeholder: 사용자의 레짐 모델로 교체 예정) -------------
def kalman_trend(s: pd.Series, n=750):
    """log price -> local linear trend 평활. return (level_price, slope_z)"""
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    y = np.log(s[s > 0].iloc[-n:])
    if len(y) < 120: return None, None
    try:
        # MLE는 관측 노이즈를 ~0으로 추정해 가격에 그대로 붙으므로 분산을 고정:
        # 'smooth trend' (level 교란 없음) + irregular = 0.5*var(r), trend = 1e-3*var(r)
        v = float(np.var(np.diff(y.values)))
        m = UnobservedComponents(y.values, level="strend")
        res = m.smooth([0.5 * v, 1e-3 * v])
        lvl = res.smoothed_state[0]; slope = res.smoothed_state[1]
        z = slope / (np.std(slope) + 1e-12)
        return pd.Series(np.exp(lvl), index=y.index), pd.Series(z, index=y.index)
    except Exception:
        return None, None

def pct(a, b): return None if (b in (0, None) or a is None or pd.isna(a) or pd.isna(b)) else (a / b - 1) * 100

# 롤오버 갭 보정 가격 (roll_adjust.py) — 보정 적용 품목만
_adj = pd.read_parquet(A._daily_path(True))
_rs = pd.read_csv(D / "roll_summary.csv") if (D / "roll_summary.csv").exists() else pd.DataFrame(columns=["id", "보정적용"])
ADJ_IDS = set(_rs.loc[_rs["보정적용"] == "✔", "id"])
ADJ = {i: g.sort_values("date").drop_duplicates("date", keep="last").set_index(pd.to_datetime(g.sort_values("date").drop_duplicates("date", keep="last").date))["value"]
       for i, g in _adj[_adj.id.isin(ADJ_IDS)].groupby("id")}

series_out, rows = {}, []
for sid, g in daily.groupby("id", sort=False):
    g = g.sort_values("date").drop_duplicates("date", keep="last")
    s = pd.Series(g.value.values, index=pd.to_datetime(g.date))
    meta = g.iloc[0]
    # 수익률·변동성·추세는 롤오버 갭 보정 가격(sa), 가격 수준·스파크라인은 실제 시세(s)
    sa = ADJ.get(sid, s).reindex(s.index)
    lvl, z = kalman_trend(sa) if meta.group not in ("FX", "Macro") else (None, None)
    if lvl is not None:   # 보정 가격의 추세를 실제 시세 수준으로 환산해 차트에 겹쳐 표시
        lvl = lvl * (s / sa).reindex(lvl.index)
    ret = A.log_return(sa, meta.unit) / 100 if not A.is_rate(meta.unit) else A.log_return(sa, meta.unit)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    volpct = float((vol20.dropna().iloc[-750:] <= vol20.iloc[-1]).mean() * 100) if vol20.notna().sum() > 60 else None
    last = float(s.iloc[-1]); lastd = s.index[-1]; la = float(sa.iloc[-1])
    ytd_base = sa[sa.index < pd.Timestamp(lastd.year, 1, 1)]
    y1 = s.iloc[-252:]
    regime = None
    if z is not None:
        zz = float(z.iloc[-1]); regime = "up" if zz > 0.5 else "down" if zz < -0.5 else "flat"
    rows.append(dict(id=sid, name=meta["name"], group=meta.group, unit=meta.unit, source=meta.source,
                     last=last, date=lastd.strftime("%Y-%m-%d"), stale=bool(lastd < ASOF - pd.Timedelta(days=4)),
                     d1=pct(la, sa.iloc[-2]), d5=pct(la, sa.iloc[-6]) if len(sa) > 6 else None,
                     m1=pct(la, sa.iloc[-22]) if len(sa) > 22 else None,
                     ytd=pct(la, ytd_base.iloc[-1]) if len(ytd_base) else None,
                     roll_adj=sid in ADJ_IDS,
                     p1y=float((y1 <= last).mean() * 100), regime=regime,
                     zslope=None if z is None else round(float(z.iloc[-1]), 2), volpct=volpct,
                     spark=[round(float(v), 4) for v in s.iloc[-90:].values]))
    # 용량 절감: 날짜는 시작일 + 일수 오프셋, Kalman은 값 배열을 끝에 정렬(앞부분 null)
    s = s[s.index >= DASH_START]
    t0 = s.index[0]
    kk = None
    if lvl is not None:
        # (WTI 2020-04 음수가격처럼 로그 불가 시점은 lvl에 없으므로 날짜 기준으로 정렬)
        kk = [None if pd.isna(v) else float(f"{v:.5g}") for v in lvl.reindex(s.index).values]
    series_out[sid] = dict(n=meta["name"], u=meta.unit, g=meta.group, t0=t0.strftime("%Y-%m-%d"),
                           o=[int(x) for x in (s.index - t0).days], v=[float(f"{v:.5g}") for v in s.values], k=kk)
    print("done", sid)

rows.sort(key=lambda r: (GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else 99, r["name"]))

mon = {}
for sid, g in monthly.groupby("id", sort=False):
    g = g.sort_values("date")
    meta = g.iloc[0]
    mon[sid] = dict(n=meta["name"], u=meta.unit, s=meta.source,
                    d=[d.strftime("%Y-%m") for d in g.date], v=[round(float(v), 4) for v in g.value])

cot_out = {}
if len(cot):
    for nm, g in cot.groupby("name"):
        g = g.sort_values("date")
        cot_out[nm] = dict(d=[d.strftime("%Y-%m-%d") for d in g.date], mm=[int(v) for v in g.mm_net.fillna(0)],
                           comm=[int(v) for v in g.comm_net.fillna(0)], oi=[int(v) for v in g.open_interest_all.fillna(0)])

# ---- 변동성: 단변량 GARCH(일간, 최근 10년) + DCC(주간, 2005~) ------------------------------
def _f(v, n=4):
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def build_vol():
    import mvgarch as M
    cat = A.load_catalog(); U = dict(zip(cat.id, cat.unit)); NM = dict(zip(cat.id, cat.name)); L = A.load_levels()
    garch_ids = [r["id"] for r in rows if r["group"] not in ("FX", "Macro", "FRED daily") and not r["stale"]]
    g_rows = []
    start10 = ASOF - pd.DateOffset(years=10)
    for i in garch_ids:
        try:
            Rd = A.make_panel(L, [i], U, "B", "logret", start10)
            gt, _ = M.garch_table(Rd, 252, "GJR", "t", NM)
            r = gt.iloc[0]
            g_rows.append(dict(id=i, name=NM[i], group=GROUP.get(i, ""), cur=_f(r["현재 GARCH 변동성"], 2), lr=_f(r["장기 변동성"], 2),
                               f22=_f(r["22기 예측"], 2), rv20=_f(r["최근 20기 실현"], 2), pct=_f(r["현재 백분위(%)"], 1),
                               persist=_f(r["지속성"], 3), half=_f(r["반감기"], 1), gamma=_f(r["γ(비대칭)"], 4)))
        except Exception as ex:
            print("GARCH skip", i, ex)
    key = [i for i in ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "ak:macro_shipping_bdi"] if i in U]
    Rw = A.make_panel(L, key, U, "W-FRI", "logret", "2005-01-01")
    dc = M.dcc_fit(Rw, False, "GJR", "t", 52)
    t0 = Rw.index[0]
    dcc = dict(t0=t0.strftime("%Y-%m-%d"), o=[int(x) for x in (Rw.index - t0).days], ids=key, names=[NM[i] for i in key],
               pairs={f"{a}|{b}": [round(float(v), 4) for v in s.values] for (a, b), s in dc["pairs"].items()},
               avg=[round(float(v), 4) for v in dc["avg_corr"].values],
               a=_f(dc["params"].iloc[0, 0]), b=_f(dc["params"].iloc[1, 0]))
    return dict(garch=g_rows, dcc=dcc)


GROUP = {r["id"]: r["group"] for r in rows}
vol = build_vol()
print(f"변동성: GARCH {len(vol['garch'])}개, DCC {len(vol['dcc']['ids'])}개 품목 (a={vol['dcc']['a']}, b={vol['dcc']['b']})")

payload = dict(asof=ASOF.strftime("%Y-%m-%d"), generated=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
               rows=rows, series=series_out, monthly=mon, cot=cot_out, status=status.fillna("").to_dict("records"), vol=vol)
html = (ROOT / "dashboard_template.html").read_text(encoding="utf-8").replace(
    "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str))
(ROOT / "dashboard.html").write_text(html, encoding="utf-8")
print("wrote dashboard.html", round(len(html) / 1e6, 2), "MB; asof", payload["asof"])
