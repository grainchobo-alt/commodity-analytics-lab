# 변경 이력

형식: `vMAJOR.MINOR.PATCH — 날짜 — 요약`. 기능 단위로 MINOR, 버그 수정은 PATCH. 버전마다 git 태그(`git tag vX.Y.Z`)를 단다.
v1.0.0 이전 단계는 git 도입 전 작업(2026-10-13 ~ 15, 날짜는 대략)을 정리한 것.

## v1.4.0 — 2026-10-19 — 변동성 중복 정리 (탭 역할 분리)
- 역할 분리: ① 무엇이 같이 움직이나 · ② 지금 어떤 국면인가 · ③ σ·Σ를 어떻게 추정하나 · ◆ 손실이 얼마나 될 수 있나
- ◆ 리스크의 '변동성 모형 검증'(QLIKE·DM·MCS·부호 편향·Ljung-Box·공통 λ LR·마할라노비스) → ③ 새 하위 탭 '변동성 예측 검증'
- ◆ 리스크의 HMM 3상태 확률·블록별 λ 우도비 검정 → ② 새 하위 탭 '변동성 레짐'
- ◆ 리스크의 꼬리 의존(동반 폭락·급등) → ① 상관행렬(평시·위기) 아래로
- 제거: Kalman TVP-AR(1)+EWMA σ 모형(고정 λ와 성능 동일), 반감기 경로·φₜ 차트, ① 롤링 DY(TVP-VAR로 대체)
- ◆ VaR 기본 σ = GARCH(1,1) (12품목 MCS 전부 포함), 리스크 실험도 GARCH σ로 재실행

## v1.3.0 — 2026-10-19 — ④ 모델 선택 메타 전략 (시간·레짐별 ML 모델 선택 + 데이터 스누핑 보정)
- 새 모듈 `metasel.py`: 후보 모델 walk-forward OOS 경로 → 중첩 walk-forward 메타 선택 (a) 직전 N개월 최소손실 (b) 지수가중 역MSE 결합 (c) 레짐 조건부(HMM 3상태 필터·실현변동성 3분위·평균 쌍별 상관 3분위) (d) 동일가중 (e) 과거 평균. 손실은 실현이 끝난 것(s ≤ t−h)만 사용
- 평가: OOS R², 부호전략 Sharpe(비용 전후)·B&H·롱 비중·하위기간(2008~16 / 2017~); 위약 = 롱 비중·전환 빈도 일치 무작위 부호(단일·'후보 중 최댓값'), White Reality Check·Hansen SPA(정상 부트스트랩)
- ④ 단일 품목 탭에 '모델 선택 메타 전략' 섹션(`ms_run`), ⑥ 실험 결과에 '모델 선택 메타 전략'(6품목 × 일간·주간, `experiments.py metasel`, 누적 성과 `data/model_selection_cum.parquet`)
- 결과: 156칸 중 위약 95% 분위 초과 0개, SPA 기각(B&H·과거평균 대비) 0건 → 유의하지 않음(운과 구별 불가)으로 기록
- 테스트: `tests/test_metasel.py`(가격 교란·손실 교란 미래정보 테스트, SPA 크기·검정력, 위약 검사), `tests/test_app.py`에 새 버튼·섹션 경로

## v1.2.1 — 2026-10-19 — 중복 기능 정리 · 단순화 (결과 수치 불변)
- 변동성 예측 경주 일원화: ③ '변동성 예측 (MIDAS·HAR·GARCH-MIDAS)'의 월간 경주(`midas.vol_race`·`vol_metrics` 삭제)를 ◆ 리스크 '변동성 모형 검증'(`risk.vol_models`)으로 통합. ③ 하위 탭은 'GARCH-MIDAS (장기·단기 변동성 분해)'만 남기고 안내 문구
- HMM 전방 필터 공용화: `procurement.hmm_crisis_prob`(2상태)가 `risk.hmm_filter`(구 `_hmm_filter`)를 사용 — 구매 타이밍 백테스트 차이 < 1e-8bp (HMM 적합 자체의 실행 간 잡음 수준)
- EWMA(RiskMetrics λ=0.94) 공용 함수 `risk.ewma_var`·`risk.ewma_std` → `midas.daily_vol_forecasts`(포트폴리오 변동성 모형 EWMA 포함)·`procurement.signals`가 사용, 수익률 계산은 `risk.returns` 공용. 수치 완전 동일
- ④ 단일 품목: 기본 모델 = 모형군별 대표 7개(`A.CORE_MODELS`: OLS·Ridge·Lasso·PLS·Kalman TVP·HMM·LightGBM), 나머지 20개는 '고급: 추가 모델' 확장 영역(`key="models_adv"`)
- ④ 성과 표에 다중 비교 기준선 2행: '위약 · 무작위 부호 95% 분위', '위약 · 모델 N개 중 최댓값 95% 분위' (`A.placebo_sharpe`, 롱 비중·전환 확률을 모델 평균에 맞춘 마르코프 무작위 부호) + 기준선을 넘은 모델 표시
- ★ 구매 타이밍 프리셋: 기본 노출 '추천'·'균형'(`PR.BASIC_PRESETS`), 검증상 열위인 '보수적'·'공격적'은 '고급: 비교용 프리셋' 체크(`key="pc_adv"`) 시에만
- ⑥ 실험 결과: '구매 신호 고도화 (단계별)'와 '구매 타이밍'을 '구매 타이밍 (프리셋 · 단계별)' 한 섹션으로 합치고 보기 전환(`key="ex_pc_view"`)
- 테스트: `test_app` key 갱신(경주 단계 삭제, 고급 프리셋·고급 모델·위약 행 확인), `test_midas` 실데이터 경주 → GARCH-MIDAS만

## v1.2.0 — 2026-10-19 — ◆ 리스크 탭 (시변 감쇠 변동성 · 예측 검증 · VaR/ES · 꼬리 위험)
- 새 모듈 `risk.py` + 앱 탭 '◆ 리스크 (VaR·꼬리)' (하위 5탭)
- 시변 감쇠 λ: EWMA 롤링 λ(직전 3년), HMM 3상태 레짐 λ(필터 확률 가중), Kalman TVP-AR(1) 잔차 + EWMA — 모두 walk-forward, 반감기 경로·HMM 확률·φₜ 시각화
- 예측 검증: QLIKE, Diebold-Mariano(Newey-West), Model Confidence Set(T_max, 블록 부트스트랩), 우도비(품목 간·기간 간 공통 λ), Engle-Ng 부호 편향, Ljung-Box, 다변량 마할라노비스
- VaR/ES: 정규·Student-t·FHS·EVT-GPD·역사적 시뮬레이션, 롱(재고)/숏(매입 예정) 양방향, 1~20일; Kupiec·Christoffersen·McNeil-Frey ES·바젤 신호등
- 꼬리 위험: Hill 지수·표준화 잔차 첨도, 동반 폭락/급등(꼬리 의존), 낙폭·저점 대비 상승, 최악 구간; 물량 위험: 다품목 정규·다변량 FHS VaR, 오일러 기여도, 위기 시나리오
- 구매 타이밍 '4주 90% 범위'를 경험적 분위로 보정 (`procurement.band_4w`)
- ⑥ 실험 결과에 '리스크 모형'(12품목: 변동성 경주·VaR 백테스트·4주 범위 적중률), `tests/test_risk.py`(미래정보 교란·크기/검정력 시뮬레이션)

## v1.1.0 — 2026-10-15 — 구매 신호 품목 확장 (철광석·점결탄·구리)
- 구매 타이밍에 철광석(SGX 62%)·점결탄(DCE)·구리(COMEX) 추가 → 9품목. 점결탄은 그날 달러/위안 환율로 $/t 환산
- 상대가치 정의: 철광석 = SGX/DCE 수입 차익(환율·증치세 환산), 점결탄 = 철근 고로 마진(철근 − 1.6·철광석 − 0.5·코크스), 구리 = 구리/금(참고 수준)
- 검증(추천 프리셋, 13주): 철광석 +27bp(전 +30/후 +24), 점결탄 +23bp(전 +31/후 +17), 구리 +5bp. 8·26주와 마진 계수 ±에도 일관. 탈락 후보는 ⑥ 실험 결과에 기록
- 수집 추가: SHFE 철근 연속(`ak:RB0`), 달러/위안(`yf:CNY=X`). COT 맥락에 구리 추가
- ⑥ 실험 결과에 '철강 원료·구리 상대가치' 실험(`experiments.py procure_metals`), 단계별 실험은 곡물 6품목으로 고정(`PR.GRAIN_IDS`)
- 테스트: 금속 상대가치·환산의 미래정보 교란 테스트 추가

## v1.0.0 — 2026-10-15 — 구매 의사결정 대시보드 · git 도입
- ★ 구매 타이밍 탭: 매수 점수(가격 위치·단기 반등·상대가치·계절성), 매수 속도 규칙, 4주 가격 범위, 펀더멘털 맥락(WASDE 재고/사용비율, COT), 계산기, 과거 검증(최근 가중·국면별·3년 이동)
- 구매 신호 고도화: MIDAS·HMM 위험 판단, 상대가치(크러시 마진·대체 가격비), 계절성 최근 가중, 가격 위치 창, 보유비용 → '추천' 프리셋
- ⑥ 실험 결과 탭 + `experiments.py` (재실행·실행 기록)
- 리스크 버짓(신호 기울기) 전략, 모델 도감 31개
- `.gitignore`·`.env` 분리, private GitHub 레포, 프로젝트 지침(CLAUDE.md), 인수인계 문서(versions/)

## v0.9 — 탭 재구성 · 포트폴리오 운용 규칙
- 9개 → 질문 기준 탭(시장 구조 / 레짐 / 변동성 / 수익률 예측 / 포트폴리오 / 도감 / 방법론), 모델 상세 공용화
- 모멘텀 기간 선택(CTA 혼합 포함), 무거래 구간·부분 조정 거래 규칙, MIDAS 일간 변동성 예측 → 변동성 사이징·타기팅

## v0.8 — D그룹 머신러닝
- LightGBM + TreeSHAP, Gaussian Process, Kernel Ridge, Conformal 예측구간(분할·적응형 ACI)

## v0.7 — 기관 리포트 검증
- USDA WASDE(1997~, txt·xls·원본 우선)·EIA STEO(2010~) 수집·파싱, 이벤트 스터디, 3년 발견 → 20년 재현 검증

## v0.6 — C그룹 레짐 · 패널 계량
- Markov-switching 회귀, TAR·STAR, TVP-VAR 연결성 / Fama-MacBeth·고정효과 OLS·섹터 기울기

## v0.5 — A·B그룹 예측 모델
- PCR·PLS·3PRF·Complete Subset·BMA·Horseshoe·Minnesota·Stacking / 원자재 패널 모델, MIDAS·GARCH-MIDAS

## v0.4 — 백테스트 정합성
- 미래정보 감사(클리핑·Kalman·공표지연·CV gap), Sharpe 감사, 롤오버 갭 보정, Newton-Raphson 바스켓 최적화, ETF 기반 초과수익

## v0.3 — 변동성 분석
- GARCH·GJR·EGARCH, DCC·ADCC·BEKK, HMM, Diebold-Yilmaz

## v0.2 — 데이터 저장 · 품질
- 최대 이력 수집, 증분 업데이트, Excel(일·월·연), 결측·이상치 리포트, 정제 규칙

## v0.1 — 대시보드 초안
- Yahoo·FRED·World Bank·akshare·USDA 무료 소스 조사, 일간 정적 대시보드, Streamlit 분석 앱 초안
