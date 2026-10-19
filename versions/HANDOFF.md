# 세션 인수인계 (HANDOFF)

> 새 세션은 이 문서를 먼저 읽고 시작한다. 세션을 마칠 때는 **현재 상태 · 남은 과제 · 최근 결정**을 갱신하고 `CHANGELOG.md`에 버전을 추가한다.

- **최종 갱신**: 2026-10-19 · **현재 버전**: v1.2.1 (git 태그)
- **사용자**: 곡물·에너지·금속·운임을 다루는 **중간 트레이더**(직접 사용 X, 매입 후 재판매). 실무 의사결정 = **"언제, 얼마나 살까"** (숏 아님).
- **저장소**: https://github.com/grainchobo-alt/commodity-analytics-lab (private, `main`) · 협업자 `seokhwan33`(write, 초대 발송)
- **작업 폴더**: `C:\Users\user\Desktop\Commodity` (구 폴더 `Desktop\261013`에는 다른 프로젝트 공존 — 삭제 금지)

## 1. 현재 상태 한눈에
| 영역 | 상태 |
|---|---|
| 데이터 파이프라인 | 완료 — `run_daily.bat` (collect → clean → roll_adjust → data_quality → export_excel → build_dashboard), 증분 수집 |
| 분석 앱 (Streamlit, 9탭) | 완료 — `run_lab.bat` → http://localhost:8501 |
| 테스트 | 전부 통과 (2026-10-19, v1.2.1) — `tests/test_*.py` 15개, 앱 전체 `tests/test_app.py` (~10~20분) |
| 문서 | `README.md`(설치·구성), `대시보드_기획서.docx`(1장, git 제외), 이 폴더 |

## 2. 탭 구성 → 모듈
| 탭 | 함수(app.py) | 엔진 모듈 |
|---|---|---|
| ★ 구매 타이밍 (기본 노출 '추천'·'균형', 보수적·공격적은 고급 체크, 곡물 6 + 철광석·점결탄·구리) | `tab_procure` | `procurement.py` |
| ◆ 리스크 (시변 λ · 변동성 예측 경주·검증(유일) · VaR/ES · 꼬리 · 물량 위험) | `tab_risk` | `risk.py` |
| ① 시장 구조·동조성 (상관·PCA·DY·TVP-VAR / FAVAR) | `page_structure` → `tab_comove`, `tab_favar` | `analytics.py`, `regime.py` |
| ② 레짐 (Markov / HMM·MS·Kalman·TAR·STAR) | `tab_regime` (+ 공용 `model_detail`) | `analytics.py`, `regime.py` |
| ③ 변동성·공분산 (GARCH / DCC·BEKK / GARCH-MIDAS 분해만) | `page_vol` → `tab_vol`, `tab_mvgarch`, `tab_midas(vol_only=True)` | `analytics.py`, `mvgarch.py`, `midas.py` |
| ④ 수익률 예측 (단일 품목: 대표 7 + 고급 20모델, 위약 기준선 / 패널 / MIDAS 회귀) | `page_predict` → `tab_ml`, `tab_panel`, `tab_midas(False)` | `analytics.py`, `models_dense.py`, `models_ml.py`, `panel.py` |
| ⑤ 포트폴리오 | `tab_portfolio` | `portfolio.py` |
| ⑥ 실험 결과 (재실행 버튼, 구매 타이밍 = 단계별·프리셋 비교 한 섹션, 철강 원료·구리 상대가치 포함) | `tab_experiments` | `experiments.py` → `data/*_compare.csv`, `experiments_meta.json` |
| ⑦ 모델 도감 (31개) | `page_atlas` | `model_atlas.py` |
| ⑧ 데이터·방법론 | `tab_data` | — |
| (탭 외) 기관 리포트 수집·평가 | — | `reports.py` (WASDE 1997~, EIA STEO 2010~) |

## 3. 핵심 결론 (의사결정에 쓰는 근거)
- **수익률 방향 예측은 모든 모델군에서 OOS R² ≈ 0** → 방향 베팅 근거로 쓰지 않음. 예측 가능한 것은 **변동성·위험**(MIDAS > HAR > EWMA).
- **포트폴리오**: 리스크 버짓(τ=0.5) ≥ 리스크 패리티 > 평균-분산 최적화. 모멘텀은 12-1개월이 최선, 무거래 구간(20%, 최대 63일)이 회전율 절반에 같은/더 높은 Sharpe. 변동성 타기팅은 위험 안정화용(Sharpe 개선 없음, 롱온리는 악화).
- **구매 타이밍(추천 프리셋)**: 상대가치 40% + 가격 위치 27% + 단기 반등 18% + 최근가중 계절 15%, MIDAS 위험, 6개월 창 → 균등 매수 대비 +13bp (2008~16 +14 / 2017~ +12, t=2.35, 싸게 산 비율 60%). **상대가치(실제 시세의 크러시 마진·대체 가격비)가 최근에도 유효한 유일한 신호**. 추세 신호는 구매자에게 해로움(→ 가중 0). 대두유는 바이오연료 정책으로 상대가치 효과 없음.
- **철강 원료·구리(2013~, 전반 ~2019.6 / 후반 2019.7~)**: 철광석 = SGX/DCE 수입 차익 +27bp, 점결탄 = 철근 고로 마진 +23bp — 8·13·26주, 계수 ±에서도 전·후반 모두 +. 구리는 알루미늄 대체 비율이 후반에 음(−)이라 기각했고, 구리/금은 +5bp(t≈1)로 참고 수준. 표본이 짧고 DCE 연속월물은 롤 보정이 없음.
- **리스크(12품목, 2010~ 평가)**: 같은 품목의 λ는 기간마다 유의하게 다르지만(LR), 매년 λ를 바꾸는 롤링·HMM은 1일 분산 예측에서 오히려 나빴음(MCS 포함 6·1/12). 1일 예측 최선 = GARCH(1,1)(12/12), 그다음 품목별 고정 λ·Kalman TVP 잔차(9/12) → 시변 λ는 레짐 진단용, VaR의 σ에는 GARCH·품목별 λ가 안정적. 1일 99% VaR는 정규가 거의 탈락(평균 초과율 1.8%), FHS·EVT가 정확(1.0~1.1%). 4주 범위는 MIDAS σ 기준 정규도 ~90%이나 경험적 분위가 대두박·대두유·철광석을 개선.
- **H-DCC 논문(사용자 졸업논문)**: 모형 자체는 대시보드에 넣지 않음(학계 검증 전, 사용자 결정). 논문의 시변 λ(롤링·HMM·Kalman TVP)와 통계 검증(DM·MCS·LR·부호 편향·마할라노비스)만 ◆ 리스크 탭에 일반 방법으로 반영. 원자료 `H-DCC/HDCC_Master_Data.xlsx`(git 미추적).
- **기관 리포트(WASDE·STEO)**: 발표 당일 반응은 강함(t≈−7)이나 이후 예측력은 과거 20년에서 재현 안 됨 → 점수 미반영, 맥락 표시만. 3년 표본만으로 찾은 패턴 2개가 가짜였음(짧은 표본 경계).

## 4. 지켜야 할 설계 원칙
- **공용 구현 (v1.2.1)**: 수익률 `risk.returns`, RiskMetrics EWMA `risk.ewma_var`/`ewma_std`, 시변 λ EWMA `risk.ewma_path`, HMM 전방 필터 `risk.hmm_filter`, 일간 HAR·MIDAS 변동성 `midas.daily_vol_forecasts`. 변동성 모형 비교는 `risk.vol_models` 하나 — 새로 만들지 말고 여기에 추가.
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
3. **구매 신호 품목 확장 후속**: 일간 연료탄(뉴캐슬) 가격 소스 확보 → 연료탄 추가 (현재 월간만 있음). 구리는 LME·양산 프리미엄 등 더 나은 상대가치 관계 탐색. DCE 연속월물(I0·JM0·RB0) 롤 보정.
4. **알림 자동화**: 매수 점수 급변·주간 요약을 월요일 아침 알림으로.
5. 미완료 제안: 패널 탭 RE + Mundlak(강건 하우스만) 검정, 회귀 탭 SUR(다중 품목 잔차 상관) 모드, 설명 모드 동시점 헤지비율 뷰.
6. 사용자 본인의 **Kalman 레짐 모델(Sharpe ≈ 0.5)** 사양을 받아 MS·TAR와 같은 조건으로 비교 (사양 미수령).
7. `Desktop\261013` 폴더의 옛 원자재 파일 정리 (삭제는 사용자 확인 후에만).

## 7. 최근 결정 로그
- 2026-10-19 (v1.2.1): 중복 정리. 변동성 경주는 ◆ 리스크로 일원화(③ 월간 경주 `midas.vol_race` 삭제 — 21일 시계 경주 결과는 v1.2.0 이전 기록으로만 남음), HMM 필터·EWMA는 risk.py 공용(결과 수치 불변 확인). ④ 기본 모델 7개 + 고급 20개, 성과 표에 무작위 부호 위약(단일 95%·N개 최댓값 95%) 기준선. 구매 타이밍 기본 프리셋 2개(보수적·공격적은 고급). ⑥ 구매 타이밍 두 섹션 통합.
- 2026-10-19 (v1.2.0): ◆ 리스크 탭 추가. H-DCC 공분산 모형은 넣지 않음(사용자: 학계 검증 전 자체 제안 모형). 시변 λ 3종·검정·VaR/ES·꼬리 위험 추가. 구매 타이밍 4주 범위 = 경험적 분위. 구매자 위험은 '가격 상승'(숏 노출)으로 별도 평가.
- 2026-10-15 (v1.1.0): 구매 신호 9품목 확장. 금속은 곡물과 표본 기간이 달라 별도로 검증(`procure_metals`)하고, 단계별 실험은 곡물 6품목으로 고정. 점결탄 $/t는 그날 환율로 환산(증치세 포함). 개선안에서 GitHub 관련 내용은 제외(CLAUDE.md).
- 2026-10-15: 구매 타이밍 기본 프리셋 = '추천(고도화)'. WASDE·COT는 점수 제외(맥락만). 기획서 docx는 git 제외. 프로젝트 지침(CLAUDE.md): 큰 편집 후 커밋+해시 보고, push 전 사용자 확인, 답변 끝에 개선안 3개.
