"""상관 묶음별 포트폴리오 백테스트 — 상관 히트맵의 빨간 블록(곡물·에너지·금속)을 묶어 투자하면 어떻게 되나.

묶음: ① 식량(곡물·유지) 단독 ② 식량 + 에너지 ③ 금속 단독 ④ 식량 + 에너지 + 금속  (+ 비교용 19품목 전체)
식량에는 대련(DCE) 팜유 주력 연속물(akshare 'P0', 2007-10~)을 넣은 판과 뺀 판을 함께 계산.
전략: (a) 동일가중 롱 — 연 10% 목표 변동성 (b) 대시보드 혼합 = MinVar 50 + 52주 모멘텀 25 + 가치 25 (ls_extra.item_contrib와 같은 조리법).
비용 = 편도 5bp. 기간 2008~2026-10 (팜유 시작에 맞춤), 학습 ~2016 / 시험 2017~.

사용: .venv\\Scripts\\python.exe corr_group_backtest.py   → data/corr_group_compare.csv
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analytics as A  # noqa: E402
import ls_strategies as LS  # noqa: E402
from ls_extra import xs_weights  # noqa: E402

FOOD = ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F"]
ENERGY = ["yf:CL=F", "yf:RB=F", "yf:HO=F", "yf:NG=F"]
METAL = ["yf:GC=F", "yf:SI=F", "yf:PL=F", "yf:HG=F"]
PALM = "ak:P0"
START, SPLIT, END, TGT, COST = "2008-01-01", "2017-01-01", "2026-10-26", 10.0, 5.0


def palm_daily() -> pd.Series:
    """대련 팜유 주력 연속물 종가(위안/톤). data/ak_P0.parquet에 저장해 두고 재사용."""
    f = ROOT / "data" / "ak_P0.parquet"
    if not f.exists():
        import akshare as ak
        d = ak.futures_zh_daily_sina(symbol="P0")
        pd.DataFrame({"date": pd.to_datetime(d["date"]), "close": d["close"].astype(float)}).to_parquet(f)
    d = pd.read_parquet(f)
    return d.set_index("date")["close"].rename(PALM)


def mix_book(P: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """동일가중 롱 · 혼합(MinVar 50 + 모멘텀 25 + 가치 25), 둘 다 목표 변동성 10%, 비용 차감 주간 수익(%)."""
    R = LS._ret(P); lp = np.log(P)
    cap = max(0.25, 1.0 / P.shape[1])                       # 품목 4개 이하면 상한 0.25로는 합 1이 안 됨
    W_mv = LS._vol_target(LS.alloc_weights_cap(P, cap), R, TGT)
    W_mo = LS._vol_target(LS.tsmom_weights(P, 52, port_tgt=None), R, TGT)
    W_va = LS._vol_target(xs_weights(-(lp - lp.rolling(260, min_periods=156).mean())), R, TGT)
    raw = 0.5 * W_mv + 0.25 * W_mo + 0.25 * W_va
    gross = (raw.shift(1) * R).sum(1)
    k = (TGT / (gross.rolling(26, min_periods=13).std().shift(1) * np.sqrt(52))).clip(upper=4.0).fillna(1.0)
    mix = LS._book(raw.mul(k, axis=0), R, COST)
    W_eq = pd.DataFrame(1.0 / P.shape[1], index=P.index, columns=P.columns).where(R.notna(), 0.0)
    eq = LS._book(LS._vol_target(W_eq, R, TGT), R, COST)
    return eq, mix


def stats(r: pd.Series) -> dict:
    out = {}
    for tag, s in (("전체", r), ("학습 ~2016", r[:SPLIT]), ("시험 2017~", r[SPLIT:])):
        m = LS.metrics(s); out[f"{tag} Sharpe"] = m["Sharpe"]
        if tag == "전체":
            out.update({"연수익 (%)": m["연수익 (%)"], "연변동성 (%)": m["연변동성 (%)"], "최대낙폭 (%)": m["최대낙폭 (%)"]})
    return out


def main():
    sys.stdout.reconfigure(errors="replace")
    L = A.load_levels(roll_adjusted=True).loc[:END]
    L = L.join(palm_daily(), how="left")
    groups = {}
    for palm in (False, True):
        f = FOOD + ([PALM] if palm else []); tag = " (+팜유)" if palm else ""
        groups[f"① 식량 단독{tag}"] = f
        groups[f"② 식량 + 에너지{tag}"] = f + ENERGY
        groups[f"④ 식량 + 에너지 + 금속{tag}"] = f + ENERGY + METAL
    groups["③ 금속 단독"] = METAL
    groups["(참고) 에너지 단독"] = ENERGY
    groups["(참고) 대시보드 19품목 전체"] = LS.BASKET
    rows, series = [], {}
    for name, ids in groups.items():
        P = LS.weekly(L, ids, "2001-01-01")[ids].ffill(limit=2)
        R = LS._ret(P).loc[START:]
        C = R.corr().values; n = len(ids)
        avg_c = (C.sum() - n) / (n * (n - 1))
        eq, mix = mix_book(P)
        for strat, r in (("동일가중 롱", eq), ("혼합 50·25·25", mix)):
            r = r.loc[START:]
            series[f"{name} | {strat}"] = r
            rows.append({"묶음": name, "품목 수": n, "평균 상관": round(float(avg_c), 2), "전략": strat, **{k: round(v, 2) for k, v in stats(r).items()}})
        print("done", name, flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(ROOT / "data" / "corr_group_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(series).cumsum().to_parquet(ROOT / "data" / "corr_group_cum.parquet")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(T.to_string(index=False))
    # 팜유 주간 수익 극단값(연속물 교체 이음새 점검)
    pw = np.log(LS.weekly(L, [PALM], START)[PALM]).diff().dropna() * 100
    print("팜유 주간 수익 |r|>10% 횟수:", int((pw.abs() > 10).sum()), " 최대:", round(float(pw.abs().max()), 1))


if __name__ == "__main__":
    main()
