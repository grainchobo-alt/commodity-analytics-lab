"""LLM 도구층 검증: 모든 도구가 JSON으로 직렬화되는 작은 결과 + asof·method·source, 대시보드 함수와 같은 숫자, 잘못된 입력은 분명한 오류."""
import json, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np
import analytics as A, procurement as PR, risk as RK
from llm import tools as T

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")

def chk(o, keys=()):
    s = json.dumps(o, ensure_ascii=False, allow_nan=False)          # NaN·numpy가 남아 있으면 여기서 실패
    assert {"asof", "method", "source"} <= set(o) and all(k in o for k in keys), list(o)
    return len(s)

assert T.resolve("옥수수") == T.resolve("yf:ZC=F") == "yf:ZC=F" and T.resolve("대두박") == "yf:ZM=F"
try:
    T.resolve("없는품목"); raise AssertionError
except ValueError as e:
    assert "찾지 못함" in str(e)
ok("품목 이름 → id (한글 · id), 없는 품목은 분명한 오류")

n = chk(T.list_items("Grains"), ["items"]); assert T.list_items()["n"] >= 60
o = T.get_prices("옥수수", "2025-01-01"); chk(o, ["stats", "recent"])
assert abs(o["stats"]["현재"] - float(A.load_levels(roll_adjusted=False)["yf:ZC=F"].dropna().iloc[-1])) < 1e-2 and len(o["recent"]) <= T.MAX_ROWS
ok(f"조회: 품목 목록 · 가격 요약 (현재가 = 정제본 마지막 값, 최근 {len(o['recent'])}행)")

o = T.vol_forecast("옥수수", "GJR"); chk(o, ["persistence", "vol_now_ann_pct"])
assert 0.8 < o["persistence"] < 1 and 5 < o["vol_now_ann_pct"] < 80
r = RK.returns(A.load_levels()["yf:ZC=F"]).loc["2010-01-01":]
assert abs(o["riskmetrics_ann_pct"] - float(np.sqrt(RK.ewma_var(r).iloc[-1] * 252))) < 1e-2
for m in ("GARCH", "EGARCH"):
    chk(T.vol_forecast("밀", m))
ok(f"변동성: GJR 지속성 {o['persistence']} · 연 {o['vol_now_ann_pct']}% (EWMA 값 대시보드 함수와 일치), GARCH·EGARCH도")

o = T.var_es("대두박", 0.99, 20, "short", notional_usd=1e7); chk(o, ["now", "backtest"])
assert len(o["now"]) == len(RK.VAR_METHODS) and all(x["VaR (%)"] > 0 and x["ES (%)"] >= x["VaR (%)"] - 1e-9 for x in o["now"])
assert all(abs(x["VaR (USD)"] - x["VaR (%)"] / 100 * 1e7) < 100 for x in o["now"])
ok("VaR·ES: 5방법 모두 양수, ES ≥ VaR, 금액 = % × 명목, 백테스트 표 포함")

o = T.regime("옥수수", 2); chk(o, ["summary", "current"])
assert abs(sum(o["current"].values()) - 1) < 1e-3 and o["summary"][0]["변동성(σ)"] <= o["summary"][1]["변동성(σ)"]
ok(f"레짐: 확률 합 1, 레짐1이 저변동 (현재 {o['current_regime']})")

o = T.factors(weeks=260); chk(o, ["explained_ratio", "loadings"])
assert 0 < o["explained_ratio"]["PC1"] < 1 and 0 < o["spillover_total_pct"] < 100
o2 = T.correlation(["옥수수", "밀", "대두"]); chk(o2, ["recent"]); assert -1 <= o2["avg_pair_recent"] <= 1
ok(f"요인: PC1 몫 {o['explained_ratio']['PC1']}, 스필오버 총 {o['spillover_total_pct']}% · 상관 표")

o = T.buy_signal(["옥수수", "대두박"]); chk(o, ["table"])
ref = PR.today_table(A.load_levels(), A.load_levels(roll_adjusted=False), ["yf:ZC=F"]).iloc[0]
assert abs(o["table"][0]["매수 점수"] - ref["매수 점수"]) < 1e-2 and 0 <= o["table"][1]["매수 점수"] <= 100
chk(T.fundamentals(["옥수수"]), ["table"])
ok(f"구매 판단: 옥수수 점수 {o['table'][0]['매수 점수']} = 대시보드 today_table과 같음 · 펀더멘털")

o = T.get_results(); chk(o, ["available"]); assert "procurement_lump" in o["available"]
o = T.get_results("procurement_lump"); chk(o, ["table"]); assert len(o["table"]) <= T.MAX_ROWS
ok(f"실험 결과 표 {len(T.RESULT_FILES)}개 읽기")

o = T.ontology_graph(); chk(o, ["edges"]); assert any(e["from"] == "대두" and e["to"] == "대두박" for e in o["edges"])
o = T.ontology_backtrace("대두박", "2025-01-01", "2025-06-30"); chk(o, ["contributions"])
c = o["contributions"]; assert abs(sum(x["기여"] for x in c) - c[0]["실제_누적"]) < 0.5 + 0.06 * len(c)   # 반올림·0.05 미만 생략분
ok(f"온톨로지: 연결 {len(T.ontology_graph()['edges'])}개, 역추적 기여 합 ≈ 실제 누적 {c[0]['실제_누적']}%")

bak = T.EXPERT; T.EXPERT = Path(__file__).parent / "_tmp_views.csv"
try:
    T.save_expert_view("옥수수", 0.5, 0.7, "남미 건조로 상방", "테스트", "2026-11-01")
    try:
        T.save_expert_view("옥수수", 2, 0.5, "x", "y"); raise AssertionError
    except ValueError:
        pass
    o = T.expert_views("옥수수"); assert o["total_views"] == 1 and o["views"][0]["direction"] == 0.5
finally:
    T.EXPERT.unlink(missing_ok=True); T.EXPERT = bak
ok("전문가 의견: 저장 · 범위 밖 값 거부 · 조회 (임시 파일)")

for k in ("price", "vol"):
    p = Path(T.make_chart(k, "옥수수")["path"]); assert p.exists() and p.stat().st_size > 1000; p.unlink()
ok("그림: Plotly HTML 생성")
print(f"\n모두 통과 ({time.time()-t0:.0f}s)")
