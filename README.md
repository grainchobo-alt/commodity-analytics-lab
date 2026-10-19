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
| `procurement.py` | ★ 구매 타이밍 (곡물 6 + 철광석·점결탄·구리): 매수 점수(가격 위치·단기 반등·상대가치·계절성·추세), 기간 내 매수 속도 규칙, 균등 매수 대비 평균 매입가 절감 백테스트 |
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
- 보유한 장기 데이터는 분석 앱 사이드바에서 CSV(첫 열=날짜, 나머지=가격 레벨)로 업로드.
- Yahoo 연속월물(=F)은 롤오버 갭 포함 → `roll_adjust.py`가 만기 규칙으로 롤일을 찾아 검증된 13개 품목(CBOT 곡물·돈육·생우·SGX 철광석·RBOB·난방유·설탕)을 보정. 그 외(WTI·금·구리 등)는 롤 패턴이 불분명해 ±15σ 클리핑만 적용.


