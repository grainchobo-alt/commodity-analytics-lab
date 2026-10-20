"""실행 기록 (버전관리처럼): 내가 돌린 계산의 '설정 + 결과'를 runs/ 폴더에 남기고, 나중에 좋음·나쁨을 판정·비교.

구조 (git에 함께 올라가는 작은 텍스트 파일만):
  runs/registry.jsonl                  한 줄 = 실행 1건 요약 (run_id·시각·종류·제목·코드 버전·데이터 시점·설정 지문·핵심 지표·자동 판정 힌트)
  runs/verdicts.jsonl                  판정 이력 (추가만 함, 같은 run_id는 마지막 줄이 유효) — good / bad / neutral / pending + 메모
  runs/YYYY-MM/<run_id>/run.json       전체 설정·요약 지표
  runs/YYYY-MM/<run_id>/<표>.csv       결과 표 (성과 지표·walk-forward·누적 성과 등)

설정 지문(config_hash) = 설정 dict의 해시 → 같은 조합을 다시 돌렸는지(데이터·코드만 달라졌는지) 묶어 볼 수 있음.
환경변수 RUNLOG_DIR 로 저장 위치를 바꿀 수 있음 (테스트용).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VERDICTS = {"pending": "⏳ 미판정", "good": "✅ 좋음", "neutral": "➖ 보통", "bad": "❌ 나쁨"}


def runs_dir() -> Path:
    d = Path(os.environ.get("RUNLOG_DIR", ROOT / "runs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)) or hasattr(x, "isoformat"):
        return x.isoformat()
    return x


def config_hash(settings: dict) -> str:
    s = json.dumps(_jsonable(settings), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def code_version() -> str:
    """git describe (태그-커밋수-해시, 미커밋 변경이 있으면 -dirty)."""
    try:
        return subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def data_version() -> str:
    """분석이 쓰는 정제 데이터 파일의 수정 시각 (데이터가 갱신됐는지 구분)."""
    for f in ["daily_adj.parquet", "daily_clean.parquet", "daily.parquet"]:
        p = ROOT / "data" / f
        if p.exists():
            return f"{f}@{pd.Timestamp(p.stat().st_mtime, unit='s'):%Y-%m-%d %H:%M}"
    return "unknown"


def record(kind: str, title: str, settings: dict, summary: dict | None = None, tables: dict[str, pd.DataFrame] | None = None,
           hints: list[str] | None = None, verdict: str = "pending", note: str = "") -> str:
    """실행 1건 저장 → run_id. summary = 목록에 보일 핵심 지표(숫자), tables = 이름 → DataFrame (csv로 저장)."""
    now = pd.Timestamp.now()
    ch = config_hash(settings)
    run_id = f"{now:%Y%m%d-%H%M%S}-{kind}-{ch}"
    d = runs_dir() / f"{now:%Y-%m}" / run_id
    d.mkdir(parents=True, exist_ok=True)
    tnames = []
    for nm, T in (tables or {}).items():
        if T is None or len(T) == 0:
            continue
        T.to_csv(d / f"{nm}.csv", encoding="utf-8-sig")
        tnames.append(nm)
    row = dict(run_id=run_id, at=f"{now:%Y-%m-%d %H:%M:%S}", kind=kind, title=title, code=code_version(), data=data_version(),
               config_hash=ch, summary=_jsonable(summary or {}), hints=list(hints or []), tables=tnames,
               path=str(d.relative_to(runs_dir())).replace("\\", "/"))
    (d / "run.json").write_text(json.dumps(dict(row, settings=_jsonable(settings), note=note), ensure_ascii=False, indent=1), encoding="utf-8")
    with open(runs_dir() / "registry.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if verdict != "pending" or note:
        set_verdict(run_id, verdict, note)
    return run_id


def set_verdict(run_id: str, verdict: str, note: str = "") -> None:
    assert verdict in VERDICTS, verdict
    with open(runs_dir() / "verdicts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(run_id=run_id, verdict=verdict, note=note, at=f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}"),
                           ensure_ascii=False) + "\n")


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_registry() -> pd.DataFrame:
    """전체 실행 목록 (최신 먼저) + 최신 판정·메모. summary 지표는 각각 열로 펼침."""
    rows = _read_jsonl(runs_dir() / "registry.jsonl")
    if not rows:
        return pd.DataFrame(columns=["run_id", "at", "kind", "title", "verdict", "note", "code", "data", "config_hash", "hints"])
    R = pd.DataFrame(rows).drop_duplicates("run_id", keep="last")
    V = {v["run_id"]: v for v in _read_jsonl(runs_dir() / "verdicts.jsonl")}       # 마지막 판정이 유효
    R["verdict"] = [V.get(i, {}).get("verdict", "pending") for i in R.run_id]
    R["note"] = [V.get(i, {}).get("note", "") for i in R.run_id]
    S = pd.json_normalize(R["summary"].tolist()).set_index(R.index)
    R = pd.concat([R.drop(columns="summary"), S], axis=1)
    R["same_config_runs"] = R.groupby("config_hash")["run_id"].transform("size")
    return R.sort_values("at", ascending=False).reset_index(drop=True)


def load_run(run_id: str) -> dict:
    """run.json 전체 + 저장된 표들(DataFrame)."""
    R = load_registry()
    hit = R[R.run_id == run_id]
    if hit.empty:
        raise KeyError(run_id)
    d = runs_dir() / hit.iloc[0]["path"]
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    meta["tables"] = {p.stem: pd.read_csv(p, index_col=0, encoding="utf-8-sig") for p in sorted(d.glob("*.csv"))}
    meta["verdict"], meta["note"] = hit.iloc[0]["verdict"], hit.iloc[0]["note"]
    return meta


def settings_diff(a: dict, b: dict) -> pd.DataFrame:
    """두 실행의 설정 차이 (같은 값은 생략)."""
    keys = sorted(set(a) | set(b))
    rows = [(k, json.dumps(a.get(k), ensure_ascii=False), json.dumps(b.get(k), ensure_ascii=False)) for k in keys if a.get(k) != b.get(k)]
    return pd.DataFrame(rows, columns=["설정", "A", "B"]).set_index("설정")
