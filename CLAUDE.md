# 프로젝트 지침

## 세션 시작 · 종료
- **시작할 때 `versions/HANDOFF.md`를 먼저 읽는다** (현재 상태·구조·원칙·남은 과제).
- **세션을 마치거나 큰 작업을 끝내면** `versions/HANDOFF.md`(현재 상태·남은 과제·결정 로그)를 갱신하고, `versions/CHANGELOG.md`에 버전을 추가한 뒤 커밋하고 `git tag vX.Y.Z`를 단다.

## 답변 방식
1. **질문에 답할 때는 끝에 개선안·아이디어를 3개 제시한다** — 각각 한 문장으로.

## Git 커밋 · GitHub 업로드
1. **기능 추가나 큰 편집을 마치면 바로 커밋한다** (관련 테스트 통과 후).
   - 커밋 메시지: 첫 줄 한국어 한 줄 요약 + 빈 줄 + 주요 변경 내용 bullet.
   - 커밋 후 사용자에게 **커밋 해시와 요약 내용**을 함께 보고한다.
   - 오타·문구 수정 같은 작은 변경은 따로 커밋하지 않고 다음 커밋에 포함한다.
   - 커밋 전 `git status`로 민감·대용량 파일이 섞이지 않았는지 확인한다 (`.env`, `data/` 원자료, `*.docx`/`*.pptx` 내부 문서, `.claude/`, `.omc/`는 `.gitignore`로 제외).
2. **GitHub에 올리기(push) 전에는 반드시 사용자에게 묻는다.** 커밋까지만 하고, 승인을 받은 뒤에 push한다.
   - 원격: `origin` = https://github.com/grainchobo-alt/commodity-analytics-lab (private), 브랜치 `main`.
