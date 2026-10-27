"""Reddit 원자재 토론 수집 → FinBERT 감성 → data/mm/reddit_posts.parquet 에 누적 (매일 한 번 실행).

공식 API는 최근 글만 주므로 과거 백테스트에는 쓰지 못하고, 오늘부터 쌓아 나중에 검증한다.
필요: .env 의 REDDIT_CLIENT_ID · REDDIT_CLIENT_SECRET · REDDIT_USER_AGENT (.env.example 참고).
실행: .venv\Scripts\python.exe collect_reddit.py
"""
import sys

import multimodal as MMd

if __name__ == "__main__":
    try:
        df = MMd.reddit_collect()
    except RuntimeError as e:
        print(e)
        sys.exit(1)
    print(f"누적 {len(df)}건 · 품목별 최근 7일 평균 감성")
    if len(df):
        import pandas as pd
        recent = df[df.created >= df.created.max() - pd.Timedelta(days=7)]
        print(recent.groupby("id").tone.agg(["mean", "size"]).round(3).to_string())
