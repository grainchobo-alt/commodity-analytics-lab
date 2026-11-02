# Commodity Dashboard & Analytics Lab

원자재 데일리 시황 대시보드 + 시계열 분석 앱.

## 처음 설정 (1회)
`setup.bat` 실행 → `.venv` 가상환경 생성 및 패키지 설치 (Python 3.12 필요, torch 포함 약 1.3GB).

- **API 키 (선택)**: `.env.example`을 `.env`로 복사해 값 입력 (`NASS_API_KEY` = USDA 작황 등급). `.env`는 git에 올라가지 않음. 키가 없으면 해당 소스만 건너뜀.
- **데이터**: `data/` 원자료(약 300MB, 제3자 데이터)는 저장소에 포함하지 않음 → 처음 받은 뒤 `.venv\Scripts\python.exe collect.py --full` 후 `run_daily.bat`로 재생성.
  기관 리포트 이력은 `python reports.py --full` (WASDE 1997~·EIA STEO 2010~). 저장소에는 수동 정제 규칙(`data/cleaning_rules.csv`)과 비교 실험 결과 요약만 포함.

## 사용
| 파일 | 용도 |
|---|---|
| `run_daily.bat` | 데이터 수집(`collect.py`) → 대시보드 생성(`build_dashboard.py`) → `dashboard.html` 열기 |
| `대시보드_유리_열기.bat` | 유리+에디토리얼 테마판(`app_glass.py`, 포트 8502, 밝은 테마 고정) — `app.py`를 그대로 실행하고 CSS만 얹음, 기본판과 동시에 열 수 있음 |
| `헤지랩_열기.bat` | 헤지 랩 초안(`hedge_lab.py`, 포트 8507, 기존 대시보드와 별개) — A1 헤지 비율→VaR(USDA 보고서 포함 기간 반영) · A2 최적 헤지 비율·효과 · A3 헤지 수단 추천 · A4 크러시 묶음 헤지. 계산 `hedge_core.py` |
| `quant16.py` | 퀀트 전략 16개(옵션 2개 제외) 같은 조건 백테스트 + 기존 실험 요약 → `data/quant16_compare.csv` (발표 5장 카드 · 검증 백로그 '퀀트 전략 16' 시트) |
| `causal_data.py` | 곡물 온톨로지용 추가 자료 수집(키 불필요): NOAA 위성 식생건강지수(VHI) 주산지 20곳 · USDA 주간 수출 판매 · 걸프→일본 운임 · FAS PSD 세계 수급 → `data/mm/` |
| `causal_grain.py` | 곡물 2단 온톨로지 역방향 인과 추론: ① 정책 이벤트 반사실(위약 p값) ② 구조 모형 역사적 분해(결과 → 원인·경로) ③ PCMCI(직접 구현) 연결 검증 → `data/causal_*.csv` |
| `range_bt.py` | 가격 '구간 확률' 예측 백테스트(ForecastBench 방식): 4·13주 뒤 5구간 확률 · 오를 확률 · 80% 범위를 RPS·Brier·적중률로 채점, 과거 분포 vs 변동성 vs 온톨로지 vs 전부 넣기 → `data/range_bt_compare.csv` |
| `defer_ont.py` | 구매 '지금 살까, 미룰까' + 온톨로지 특성(원인 노드 · 원인 분해) 백테스트, 대시보드와 같은 규칙 → `data/defer_ont_compare.csv` |
| `range_defer.py` | 범위 기반 일부 물량 미루기: P(4주 뒤 −5% 아래) ≥ q*면 50% 미룸(고정형 · 지정가형), 변동성 모형 확률, q*는 학습 기간에서 선택 → `data/range_defer_compare.csv` |
| `docs/온톨로지_실험로그.md` | 곡물 2단 온톨로지 구조(그림) · 자료 · 백테스트 결과 · 결정 기록 |
| `회사노트북_설치가이드.md` · `setup_check.py` | 다른 PC(회사 노트북)로 옮길 때: 받기 → setup.bat → 점검(`setup_check.py --net`, 필수 실패 0개면 준비 완료) → 실행. 회사망 차단 시 대처·플랜 B 포함 |
| `옵션랩_열기.bat` | 옵션 랩(`options_lab.py`, 포트 8504, 기존 대시보드와 별개) — 블랙숄즈 원리(수식·3D), 금·은·WTI(GLD·SLV·USO) 내재변동성 스마일·곡면, 다른 원자재 CSV 추가 방법. 계산 `options_bs.py`, 점검 `tests/test_options_bs.py` |
| `대시보드_열기.bat` | **평소에 이것만 더블클릭** — 앱이 꺼져 있으면 켜고(최소화 창) 준비되면 브라우저 열기, 켜져 있으면 브라우저만 열기 |
| `run_lab.bat` | 분석 앱 실행 → 브라우저에서 http://localhost:8501 |
| `dashboard.html` | 생성된 대시보드 (인터넷 필요: Plotly CDN) |

## 구성
| 파일 | 내용 |
|---|---|
| `collect.py` | Yahoo(선물·달러/위안 등) / FRED / World Bank / akshare(BDI·DCE·SHFE 철근) / CFTC COT 수집 → `data/*.parquet` |
| `build_dashboard.py`, `dashboard_template.html` | 정적 대시보드 생성 |
| `analytics.py` | 분석 엔진: PCA, 흡수비율, Diebold-Yilmaz, GARCH/GJR/EGARCH, Markov-switching, OLS·Ridge·Lasso·EN·Adaptive Lasso·SCAD·MCP·분위수·Kalman TVP·RF·GBM·LSTM, FAVAR |
| `mvgarch.py` | 다변량 GARCH 직접 구현 (numba): 단변량 GARCH 요약, DCC/ADCC, BEKK(1,1) Full·Diagonal·Scalar + 전이표 |
| `portfolio.py` | 원자재 바스켓 백테스트: Newton-Raphson 평균-분산-비용 최적화(롱온리·롱숏·종목상한), 선물 틱 기준 거래·롤 비용, 다음 거래일 실행(미래정보 차단), ETF 기반 선물 초과수익 손익 |
| `models_dense.py` | A그룹: PCR·PLS·3PRF·Complete Subset·BMA·Horseshoe·Minnesota BVAR·Stacking |
| `panel.py` | B그룹: 원자재 패널(순위 특성) 예측, Fama-MacBeth·고정효과 OLS·섹터 기울기 |
| `midas.py` | B그룹: MIDAS 회귀·일간 변동성 예측(EWMA·HAR·MIDAS-RV)·GARCH-MIDAS |
| `regime.py` | C그룹: Markov-switching 회귀·TAR·STAR·TVP-VAR 연결성 |
| `models_ml.py` | D그룹: LightGBM(+TreeSHAP)·Gaussian Process·Kernel Ridge, Conformal 예측구간(분할·적응형 ACI) |
| `reports.py` | 기관 리포트 수집·평가: USDA WASDE(1997~)·EIA STEO 빈티지(2010~), 발표일 이벤트 스터디 |
| `risk.py` | ◆ 리스크 (공용 EWMA·HMM 필터·수익률 함수 포함): 시변 감쇠 λ(EWMA 롤링·HMM 3상태·Kalman TVP-AR(1)), GARCH·HAR·MIDAS 비교, QLIKE·DM·MCS·LR·부호 편향·마할라노비스, VaR/ES(정규·t·FHS·EVT·HS)와 Kupiec·Christoffersen·ES 백테스트, 꼬리 의존·스트레스·물량 위험 |
| `forecast_lab.py` · `validation.py` | 최신 시계열 파운데이션 모델(Chronos·TimesFM·TiRex) 목표별 비교, Purged K-fold·Deflated Sharpe·PBO, 메타 라벨링 (⑥ '최신 시계열 모델') |
| `procurement.py` | ★ 구매 타이밍 (36품목, 재고 기반 '이번 주 카고 1척 살까' 판단 · 1~2주 예측 · 학습/시험 검증): 매수 점수(가격 위치·단기 반등·상대가치·계절성·추세), 기간 내 매수 속도 규칙, 균등 매수 대비 평균 매입가 절감 백테스트 |
| `runlog.py` | 실행 기록(버전관리): `runs/registry.jsonl`·실행별 폴더(설정·성과 표)·판정 이력 — ④ 실행 자동 기록, ⑥ '내 실행 기록' |
| `experiments.py` | ⑥ 실험 결과: 구매 타이밍·모멘텀 기간·거래 규칙·변동성 사이징·리스크 버짓 비교 실험 (`python experiments.py [이름]`) |
| `app.py` | Streamlit 분석 앱 — ★ 구매 타이밍 / ◆ 리스크 (VaR·꼬리) / ① 시장 구조·동조성(상관·PCA·DY·TVP-VAR·FAVAR) / ② 레짐(Markov·HMM·MS 회귀·Kalman·TAR·STAR) / ③ 변동성·공분산(GARCH·DCC·BEKK·GARCH-MIDAS) / ④ 수익률 예측(단일 품목 대표 7 + 고급 모델·위약 기준선, 패널, MIDAS 회귀) / ⑤ 포트폴리오 / ⑥ 실험 결과 / ⑦ 모델 도감 / ⑧ 데이터·방법론 |
| `model_atlas.py` | ⑥ 모델 도감: 31개 모델·방법론의 원리를 합성 데이터 시각화·수식·용도·함정으로 설명 |
| `tests/` | `python tests/test_<모듈>.py` (analytics, lookahead, dense, panel, midas, regime, ml, reports, portfolio, app 등) |
| `test_sources.py` | 데이터 소스 연결 점검 |

## 데이터 저장 구조 (`data/`)
| 파일 | 내용 |
|---|---|
| `daily.parquet` | **원본 마스터** (일별, 52개 시리즈, 1927~). 수정하지 않음 |
| `daily_adj.parquet` | **롤오버 갭 보정본** (roll_adjust.py) — 수익률·변동성·계절성·예측 분석에 사용 |
| `daily_clean.parquet` | 정제본 (주말 레코드·중복·`cleaning_rules.csv` 오류값 제거). 분석·대시보드가 사용 |
| `monthly.parquet`, `cot.parquet` | World Bank·IMF 월간, CFTC COT 주간 |
| `cleaning_rules.csv` | 오류값 수동 목록 (id, date, action=nan, 사유) — 직접 추가·삭제 가능 |
| `cleaning_log.csv` | 정제 적용 내역 |
| `excel/원자재_일별_원본.xlsx` / `_정제.xlsx` | 일별 (날짜 × 품목) |
| `excel/원자재_월별.xlsx` | 월말종가·평균·고가·저가·수익률·관측일수 + World Bank/IMF 원천 월간 |
| `excel/원자재_연도별.xlsx` | 연말종가·평균·고가·저가·수익률·변동성·관측일수 + World Bank 연평균 |
| `excel/COT_주별.xlsx` | Managed Money·상업 순포지션, 미결제약정 |
| `excel/data_quality_report.xlsx` | 결측·이상치·정체값·소스 교차검증 리포트 |

**업데이트 방식**: `collect.py`는 저장된 마지막 날짜 −14일부터만 받아 병합(약 20초). 전체 재다운로드는 `collect.py --full`.
**파이프라인**: `run_daily.bat` = collect → clean → roll_adjust → data_quality → export_excel → build_dashboard

## 메모
- USDA NASS 작황 데이터: 무료 API 키 발급 후 환경변수 `NASS_API_KEY` 설정 시 자동 수집.
- ④ 수익률 예측의 실행 결과는 `runs/`(실행 기록)에 자동 저장 → ⑥ 실험 결과 '📂 내 실행 기록'에서 판정·비교·설정 불러오기.
- Yahoo 연속월물(=F)은 롤오버 갭 포함 → `roll_adjust.py`가 만기 규칙으로 롤일을 찾아 검증된 13개 품목(CBOT 곡물·돈육·생우·SGX 철광석·RBOB·난방유·설탕)을 보정. 그 외(WTI·금·구리 등)는 롤 패턴이 불분명해 ±15σ 클리핑만 적용.


