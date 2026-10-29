# 발표 자료 생성 스크립트

`원자재_퀀트_MI_발표_v*.pptx`를 다시 만드는 스크립트. 산출물(pptx·이미지·중간 json)은 git 제외.

| 파일 | 역할 |
|---|---|
| `charts.py` | 실제 데이터·개념 예시 차트 PNG → `img/`, 수치 → `nums.json` |
| `charts3d.py` | 3D 상관 지형·3D 효율적 투자선 PNG |
| `deck.js` | pptxgenjs로 슬라이드 조립 (`data.json`, `img/`, `notes.js` 사용) |
| `build.py` | 위 단계를 순서대로 한 번에 실행 + `data.json` 생성 |
| `notes.js` | 장별 발표자 노트(장 순서와 1:1, 개수가 다르면 경고) |

## 실행 (프로젝트 루트에서) — 한 번에
```
.venv\Scripts\python.exe docs\발표\scripts\build.py
```
- 차트 PNG(`charts.py`·`charts3d.py`) → `data.json` → `deck.js` 순서로 실행, 출력 = `docs\발표\원자재_퀀트_MI_발표_v{다음 번호}.pptx`(기존 파일 덮어쓰지 않음). 경로를 인자로 주면 그 파일로.
- 처음 한 번만 `cd docs\발표\scripts` 후 `npm install` (pptxgenjs).
- `data.json` = `nums.json` + `voltgt`(`data/voltgt_compare.csv`의 가치 혼합 · 배율 상한 4배 행: 목표·산술·복리·MDD·실현 변동성).
- ★ 이번 주 판단 카드(30번 장)·횡단면 군집 품목(29번 장)은 앱 화면 값을 `deck.js`에 직접 적음 — 발표일 기준으로 갱신 필요.
