# 세션 인수인계 (HANDOFF)

> 새 세션은 이 문서를 먼저 읽고 시작한다. 세션을 마칠 때는 **현재 상태 · 남은 과제 · 최근 결정**을 갱신하고 `CHANGELOG.md`에 버전을 추가한다.

- **최종 갱신**: 2026-10-15 · **현재 버전**: v1.0.0 (git 태그)
- **사용자**: 곡물·에너지·금속·운임을 다루는 **중간 트레이더**(직접 사용 X, 매입 후 재판매). 실무 의사결정 = **"언제, 얼마나 살까"** (숏 아님).
- **저장소**: https://github.com/grainchobo-alt/commodity-analytics-lab (private, `main`) · 협업자 `seokhwan33`(write, 초대 발송)
- **작업 폴더**: `C:\Users\user\Desktop\Commodity` (구 폴더 `Desktop\261013`에는 다른 프로젝트 공존 — 삭제 금지)

## 1. 현재 상태 한눈에
| 영역 | 상태 |
|---|---|
| 데이터 파이프라인 | 완료 — `run_daily.bat` (collect → clean → roll_adjust → data_quality → export_excel → build_dashboard), 증분 수집 |
| 분석 앱 (Streamlit, 9탭) | 완료 — `run_lab.bat` → http://localhost:8501 |
| 테스트 | 전부 통과 (2026-10-15) — `tests/test_*.py` 15개, 앱 전체 `tests/test_app.py` (~10~20분) |
| 문서 | `README.md`(설치·구성), `대시보드_기획서.docx`(1장, git 제외), 이 폴더 |

## 2. 탭 구성 → 모듈
| 탭 | 함수(app.py) | 엔진 모듈 |
|---|---|---|
| ★ 구매 타이밍 (기본 프리셋 '추천') | `tab_procure` | `procurement.py` |
| ① 시장 구조·동조성 (상관·PCA·DY·TVP-VAR / FAVAR) | `page_structure` → `tab_comove`, `tab_favar` | `analytics.py`, `regime.py` |
| ② 레짐 (Markov / HMM·MS·Kalman·TAR·STAR) | `tab_regime` (+ 공용 `model_detail`) | `analytics.py`, `regime.py` |
| ③ 변동성·공분산 (GARCH / DCC·BEKK / MIDAS) | `page_vol` → `tab_vol`, `tab_mvgarch`, `tab_midas(vol_only=True)` | `analytics.py`, `mvgarch.py`, `midas.py` |
| ④ 수익률 예측 (단일 품목 27모델 / 패널 / MIDAS 회귀) | `page_predict` → `tab_ml`, `tab_panel`, `tab_midas(False)` | `analytics.py`, `models_dense.py`, `models_ml.py`, `panel.py` |
| ⑤ 포트폴리오 | `tab_portfolio` | `portfolio.py` |
| ⑥ 실험 결과 (재실행 버튼) | `tab_experiments` | `experiments.py` → `data/*_compare.csv`, `experiments_meta.json` |
| ⑦ 모델 도감 (31개) | `page_atlas` | `model_atlas.py` |
| ⑧ 데이터·방법론 | `tab_data` | — |
| (탭 외) 기관 리포트 수집·평가 | — | `reports.py` (WASDE 1997~, EIA STEO 2010~) |

## 3. 핵심 결론 (의사결정에 쓰는 근거)
- **수익률 방향 예측은 모든 모델군에서 OOS R² ≈ 0** → 방향 베팅 근거로 쓰지 않음. 예측 가능한 것은 **변동성·위험**(MIDAS > HAR > EWMA).
- **포트폴리오**: 리스크 버짓(τ=0.5) ≥ 리스크 패리티 > 평균-분산 최적화. 모멘텀은 12-1개월이 최선, 무거래 구간(20%, 최대 63일)이 회전율 절반에 같은/더 높은 Sharpe. 변동성 타기팅은 위험 안정화용(Sharpe 개선 없음, 롱온리는 악화).
- **구매 타이밍(추천 프리셋)**: 상대가치 40% + 가격 위치 27% + 단기 반등 18% + 최근가중 계절 15%, MIDAS 위험, 6개월 창 → 균등 매수 대비 +13bp (2008~16 +14 / 2017~ +12, t=2.35, 싸게 산 비율 60%). **상대가치(실제 시세의 크러시 마진·대체 가격비)가 최근에도 유효한 유일한 신호**. 추세 신호는 구매자에게 해로움(→ 가중 0). 대두유는 바이오연료 정책으로 상대가치 효과 없음.
- **기관 리포트(WASDE·STEO)**: 발표 당일 반응은 강함(t≈−7)이나 이후 예측력은 과거 20년에서 재현 안 됨 → 점수 미반영, 맥락 표시만. 3년 표본만으로 찾은 패턴 2개가 가짜였음(짧은 표본 경계).

## 4. 지켜야 할 설계 원칙
- **미래정보 차단**: 모든 신호는 결정 시점까지의 데이터, 체결 = 다음 거래일 종가, 재추정은 walk-forward, HMM은 필터 확률만, 계절성은 올해 이전 연도만. 새 기능은 `tests/test_lookahead.py` 또는 해당 테스트에 **미래 데이터 교란 테스트**를 추가.
- **가격 기준**: 신호·수익률 = 롤 보정 가격(`load_levels(roll_adjusted=True)`, 키워드 인자로!), 매입가·경제적 관계(크러시·가격비) = 실제 근월물(`roll_adjusted=False`). 포트폴리오 손익 기본 = ETF 기반 초과수익(롤 비용 반영).
- **과적합 방지**: 하위 기간(2008~16 / 2017~) 모두에서 일관된 설정만 채택. 표에서 최고값 고르기 금지.
- **UI 언어**: 한국어. 새 위젯은 고유 `key` 필수(`tests/test_app.py`가 key로 조작).

## 5. 환경 · 작업 요령
- Windows 11, Python 3.12 `.venv` (`.venv\Scripts\python.exe`), LightGBM 4.7, linearmodels, hmmlearn, numba, torch 포함.
- GitHub CLI: `C:\Program Files\GitHub CLI\gh.exe` (PATH 미등록일 수 있음), 계정 `grainchobo-alt`. 레포 로컬 작성자 = grainchobo@gmail.com.
- Streamlit 재시작: 8501 포트를 이전 프로세스가 잡고 있으면 `netstat -ano | findstr :8501` → 해당 PID 종료 후 재실행.
- Bash heredoc에 `'''`·한글이 섞이면 파싱 실패하는 경우가 있음 → 긴 패치는 스크래치 파일(.py)로 작성 후 실행.
- 실행 시간: 앱 전체 테스트 10~20분, `experiments.py` 전체 ~20분, `reports.py --full` 수집 ~10분(EIA 서버 느림). 오래 걸리는 작업은 **시작 전 예상 시간을 사용자에게 알릴 것**(사용자 요청).
- 데이터(`data/`)는 git 제외 → 새 PC에서는 `collect.py --full`, `reports.py --full`, `run_daily.bat`로 재생성.

## 6. 남은 과제 · 아이디어 (우선순위 순)
1. **착지 원가 기준 구매 타이밍**: 선물 + 운임(BDI·파나막스, 데이터 보유) + 산지 베이시스(사내 데이터 필요) — 중간 트레이더에게 가장 실무적.
2. **사내 실제 매입 이력과 규칙의 사후 비교** (매입 기록 CSV 업로드 기능).
3. **구매 신호의 품목 확장**: 철광석·석탄·구리 (사용자 주력 품목; 상대가치 관계 정의 필요 — 예: 철광석↔강재 마진).
4. **알림 자동화**: 매수 점수 급변·주간 요약을 월요일 아침 알림으로.
5. 미완료 제안: 패널 탭 RE + Mundlak(강건 하우스만) 검정, 회귀 탭 SUR(다중 품목 잔차 상관) 모드, 설명 모드 동시점 헤지비율 뷰.
6. 사용자 본인의 **Kalman 레짐 모델(Sharpe ≈ 0.5)** 사양을 받아 MS·TAR와 같은 조건으로 비교 (사양 미수령).
7. `Desktop\261013` 폴더의 옛 원자재 파일 정리 (삭제는 사용자 확인 후에만).

## 7. 최근 결정 로그
- 2026-10-15: 구매 타이밍 기본 프리셋 = '추천(고도화)'. WASDE·COT는 점수 제외(맥락만). 기획서 docx는 git 제외. 프로젝트 지침(CLAUDE.md): 큰 편집 후 커밋+해시 보고, push 전 사용자 확인, 답변 끝에 개선안 3개.
