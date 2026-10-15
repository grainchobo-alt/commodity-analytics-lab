# 변경 이력

형식: `vMAJOR.MINOR.PATCH — 날짜 — 요약`. 기능 단위로 MINOR, 버그 수정은 PATCH. 버전마다 git 태그(`git tag vX.Y.Z`)를 단다.
v1.0.0 이전 단계는 git 도입 전 작업(2026-10-13 ~ 15, 날짜는 대략)을 정리한 것.

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
