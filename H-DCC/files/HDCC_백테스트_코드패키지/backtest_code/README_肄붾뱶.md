# H-DCC 백테스트 코드 패키지

## 의존성·데이터
- Python 3.10+, `numpy`, `pandas`, `openpyxl` (외부 통계 라이브러리 불필요 — 칼만/EM/QMLE 전부 자체 구현)
- 데이터: `HDCC_Master_Data.xlsx`를 코드와 같은 폴더에 두고, `backtest_core.py` 상단 `M` 경로를 수정
  - 시트 `01_Monthly_Returns_12`(월간 384×12), `02_Daily_Returns_12`(일간), `08_EWMA_Vol_Path`·`09_Std_Residuals_z`(논문 정본 σ·z)

## 공통 프로토콜
Long-only MinVar(활성집합 반복), 월말 리밸런싱, TC=7.5bp, H=DRD.
상관: 월간 Rolling-60M / 일간 Rolling-252d. λ경계: 월간 (0.79, 0.986) / 일간 (0.90, 0.999).
모든 워크포워드는 인과적(causal): 매년 1월, 직전까지의 데이터만으로 재추정.

## 파일 구조 (실행 순서대로)

### 핵심 모듈
| 파일 | 내용 |
|---|---|
| `backtest_core.py` | 데이터 로더(`load`/`load_daily`), `minvar_lo`(롱온리 MinVar), 월간 백테스터, `ewma_qmle_vec`(그리드 벡터화 QMLE) |
| `tvp_kalman.py` | `kalman_tvp_ar1`(시변 α,φ 칼만 필터, 1기 선행 잔차), `ewma_qmle_lambda`, 월간 TVP 백테스트(`run_tvp`) |
| `hmm3_lambda.py` | 자산별 3-상태 가우시안 HMM(Baum-Welch EM `fit_hmm`, 인과 필터 `causal_filter`), 통일 프로토콜 백테스트 — **가설3 정본(0.575)** |
| `hmm_lambda.py` | 2-상태 HMM 초기 구현 (참고용) |
| `hmm_wf.py` | **자산 벡터화** 3-상태 EM(`em_fit_vec`) + 워크포워드 λ경로(`hmm_lambda_path`) — 일간 대량 재적합용 |
| `daily_engine.py` | 일간 엔진: `build_paths`(모형×λ모드별 σ·z 경로), `daily_backtest`, 경제국면 정의(`ERAS`) |
| `cpd_models.py` | BOCPD·E-Divisive 시변 λ (V13에서 논문 제외, 기록용) |

### 분석 스크립트 → 결과
| 스크립트 | 산출 | 결과 요약 |
|---|---|---|
| `python backtest_core.py` | — | H-DCC Main 재현 0.473 (논문 0.479) |
| `python tvp_kalman.py` | — | TVP 0.555~0.558 (Q 둔감) |
| `python hmm3_lambda.py` | — | HMM 2/3-state 0.530/0.575 |
| `subperiod_analysis.py` | `subperiod_results.csv` | 월간 워크포워드 국면별 (H-DCC/TVP) |
| `task3_era_daily.py` | `task3_era_daily.csv` | 일간 워크포워드 국면별 |
| `task8_hmm_wf.py` | `task8_hmm_era.csv` | HMM 워크포워드 국면별 월·일 (0.692/0.797) |
| `window_analysis.py` | `window_results.csv` | 월간 윈도우 길이 (H-DCC/TVP) |
| `task5_window_daily.py` | `task5_window_daily.csv` | 일간 윈도우 길이 |
| `task9_hmm_window.py`, `task9b_...` | `task9*.csv` | HMM 윈도우 길이 일·월 |
| `task4_mahalanobis.py` | `task4_mahal_*.csv` | 마할라노비스 3모형×월·일 |
| `task6_daily_full.py`, `task7_verify_daily.py` | — | 일간 전체기간(95~/96~) 검증 + 무결성 |
| `tvp_daily.py` | — | 초기 일간 TVP (구버전 클리닝, 참고) |

### 주의사항
- `task8_hmm_wf.py`는 import 시 era 분석이 재실행되는 부작용 있음 (task9b가 이를 import)
- HMM EM은 국소해 민감: `hmm3_lambda`(단일적합 정본)와 `hmm_wf`(연도별 재적합) 의 IS고정 수치가 소폭 다름 (0.575 vs 0.545)
- 워크포워드 H-DCC는 raw 혁신 사용 → 논문 Main(Ridge z, 0.479)과 직접 비교 불가, 상대 비교용

## 벤치마크 워크포워드 모듈 (추가)
| 파일 | 내용 |
|---|---|
| `bench_core.py` | 그리드 벡터화 적합기: GARCH(1,1)(분산타기팅), DCC/ADCC(상관타기팅), scalar-BEKK, RM, EWMA λ |
| `bench_wf.py` | 7모형 워크포워드 H경로 빌더(`build_paths`) + 포트폴리오/QLIKE 평가기 |
| `task10_bench_run.py m\|d` | 국면별 WF Sharpe + QLIKE 시계열 저장 |
| `task11_dm_mcs_lr.py` | DM(NW)·MCS(정상 부트스트랩)·LR 경로 |
| `task12_bench_window.py m` / `task12d_part.py 모형들` | 윈도우 민감도 (일간은 모형 그룹 분할 실행) |
| `task13_mc_wf.py` | Monte Carlo 워크포워드 (패널 벡터화 QMLE) |

## 매크로 증강 모듈 (H-DCC-X, Phase 1 — 결론: 개선 없음)
| 파일 | 내용 |
|---|---|
| `hdccx_macro.py` | σ²=τ×g 곱셈 분해 (τ=exp(γ′F_lag), 발표지연 정렬), X2/X5 변형 + γ 횡단면 |
| `hdccx_robust.py` | 강건성 5종 (단일팩터·γ축소·평활) — 전부 중단 기준 미달 |
