# FaceGuard Backend (Spring Boot)

사진을 받아 AI 서버(`server/`, FastAPI)에 보호를 맡기고, 결과를 저장·조회한다.

```text
사용자 → Spring Boot (8080) → AI 서버 (8000) → 보호된 사진
```

## 실행 순서
두 서버를 모두 켜야 한다. **AI 서버를 먼저** 켠다.

**1. AI 서버 켜기** (둘 중 하나, 프로젝트 폴더 `faceguard/`에서)

Mac에서 바로 실행 (더 빠름, 사진 1장 약 40초):
```bash
.venv/bin/uvicorn app:app --app-dir server --host 127.0.0.1 --port 8000
```

Docker로 실행 (Docker 앱을 먼저 켜 둠, 사진 1장 약 90초):
```bash
docker run -d --rm --name faceguard-ai -p 127.0.0.1:8000:8000 -v "$PWD/models:/app/models" faceguard-ai
```

http://127.0.0.1:8000/health 에서 `{"status":"ok"}`가 나오면 준비된 것이다.

**2. Spring Boot 켜기** (새 터미널, `faceguard/backend/`에서)
```bash
./mvnw spring-boot:run
```
`Started BackendApplication` 줄이 나오면 준비된 것이다.

**3. 써 보기**: 브라우저에서 http://localhost:8080/swagger-ui.html
1. **사진** → `POST /api/images` → Try it out → 사진 선택 → Execute → 응답의 `id` 확인
2. **보호 작업** → `POST /api/protection-jobs` → Try it out → `{"imageId": (1의 id)}` → Execute → 바로 작업 번호(`id`)와 `QUEUED`가 나옴
3. `GET /api/protection-jobs/{id}`에 작업 번호를 넣고 Execute. 40 ~ 90초 뒤 다시 누르면 `DONE`
4. `protectedImageUrl`(예: `/api/images/2`)을 브라우저에서 열면 보호된 사진

**4. 끄기**
- Spring Boot: 그 터미널에서 `Ctrl+C`
- AI 서버: 그 터미널에서 `Ctrl+C`, Docker라면 `docker stop faceguard-ai`

## API
| 요청 | 하는 일 |
|---|---|
| `POST /api/images` (form-data `file`) | 사진 올리기 → 사진 번호. JPEG, PNG, WEBP, 최대 20MB |
| `POST /api/protection-jobs` `{"imageId": 1}` | 보호 작업 접수(202). 작업 번호를 바로 돌려주고 뒤에서 한 장씩 처리 |
| `GET /api/protection-jobs/{id}` | 작업 상태 (QUEUED → PROCESSING → DONE / FAILED), 실패 이유, 시도 횟수 |
| `GET /api/images/{id}` | 사진 파일 (원본 또는 보호된 사진) |
| `GET /api/images/{id}/info` | 사진 정보 |

오류는 모두 `{"error": "메시지"}` 형태다.

- AI 서버에 연결이 안 되면 최대 3번 시도한다(5초, 10초 쉬고). 얼굴이 없거나 응답이 너무 늦으면 다시 시도하지 않는다.
- 처리 중에 서버가 꺼져도, 다시 켜면 끝내지 못한 작업을 이어서 한다.

## 저장 위치 (git에 올리지 않음)
- 사진 파일: `backend/storage/images/`
- DB(H2 파일): `backend/storage/db/`. 개발용 확인 화면 http://localhost:8080/h2-console (JDBC URL `jdbc:h2:file:./storage/db/faceguard`, 사용자 `sa`, 비밀번호 없음)

## 보안 (개발용 설정)
- `server.address=127.0.0.1`: 내 Mac에서만 접속된다. 로그인이 없어서 같은 와이파이의 다른 기기가 사진을 내려받지 못하게 막는다.
- DB 비밀번호가 없다. 배포할 때는 진짜 DB 서버 + 비밀번호(환경 변수), 로그인, 추측하기 어려운 사진 번호가 필요하다.
