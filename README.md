# RAG Backend API

FastAPI 기반 RAG (Retrieval-Augmented Generation) 챗봇 백엔드 시스템

## 주요 기능

- **PDF 처리**: Docling을 사용한 고급 PDF 파싱 (OCR 자동 감지, 이미지/테이블 추출)
- **비동기 작업 큐**: Celery + Redis를 사용한 백그라운드 PDF 처리
- **우선순위 큐**: 파일 크기 기반 작업 우선순위 설정
- **벡터 데이터베이스**: Qdrant를 사용한 문서 임베딩 저장
- **메타데이터 관리**: MySQL을 사용한 파일 및 노트북 메타데이터 관리
- **고급 RAG**: Context expansion, re-ranking 등 고급 검색 기능

## 시스템 요구사항

- Python 3.11+
- Redis 7.0+
- MySQL 8.0+
- Qdrant 1.7+
- Ollama (LLM 및 임베딩용)

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Redis 설치 및 실행

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# macOS
brew install redis
brew services start redis

# Docker
docker run -d -p 6379:6379 --name redis redis:latest
```

Redis 연결 확인:
```bash
redis-cli ping
# 응답: PONG
```

### 3. MySQL 설정

```bash
# MySQL 접속
mysql -u root -p

# 데이터베이스 생성
CREATE DATABASE rag_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

# 스키마 생성
USE rag_db;
SOURCE schema/chat_history.sql;
```

### 4. Qdrant 설치 및 실행

```bash
# Docker
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant qdrant/qdrant:latest
```

### 5. 환경변수 설정

`.env.example`을 복사하여 `.env` 파일 생성:

```bash
cp .env.example .env
```

`.env` 파일 수정:
```env
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=rag_db
OLLAMA_URL=http://localhost:11434
```

### 6. FastAPI 서버 실행

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Celery Worker 실행 (별도 터미널)

#### Worker 실행 (PDF 처리용)

```bash
celery -A app.celery_app worker \
  --loglevel=info \
  --concurrency=2 \
  --queue=pdf_processing \
  --pool=prefork
```

**옵션 설명:**
- `--concurrency=2`: 동시에 처리할 작업 수 (CPU 코어 수에 따라 조정)
- `--queue=pdf_processing`: PDF 처리 전용 큐
- `--pool=prefork`: 멀티프로세싱 사용

#### Flower 모니터링 (선택사항)

```bash
# Flower 설치
pip install flower

# Flower 실행
celery -A app.celery_app flower --port=5555
```

웹 브라우저에서 `http://localhost:5555` 접속하여 작업 모니터링

## API 엔드포인트

### 파일 업로드 (비동기)

```bash
POST /api/v1/file
Content-Type: multipart/form-data

{
  "file": <PDF 파일>,
  "user_id": "user123",
  "notebook_id": "notebook456",
  "generate_image_description": false,
  "async_processing": true  # 비동기 처리 (기본값: true)
}
```

**응답:**
```json
{
  "pdf_id": "uuid-1234",
  "filename": "document.pdf",
  "chunk_count": 0,
  "message": "PDF 업로드 완료. 백그라운드에서 처리 중입니다.",
  "task_id": "celery-task-id-5678"
}
```

### 작업 상태 조회

```bash
GET /api/v1/tasks/status/{task_id}
```

**응답:**
```json
{
  "task_id": "celery-task-id-5678",
  "state": "PROCESSING",
  "status": "Running Docling parser (OCR detection)",
  "progress": 30,
  "filename": "document.pdf"
}
```

**State 종류:**
- `PENDING`: 큐에서 대기 중
- `PROCESSING`: 처리 중
- `SUCCESS`: 완료
- `FAILURE`: 실패
- `RETRY`: 재시도 중

### 작업 결과 조회

```bash
GET /api/v1/tasks/result/{task_id}
```

### 작업 취소

```bash
POST /api/v1/tasks/cancel/{task_id}
```

## 작업 우선순위

파일 크기에 따라 자동으로 우선순위가 설정됩니다:

| 파일 크기 | 우선순위 | 설명 |
|----------|---------|------|
| < 1MB | 9 | 최우선 처리 |
| 1-5MB | 7 | 높음 |
| 5-10MB | 5 | 보통 (기본값) |
| 10-50MB | 3 | 낮음 |
| > 50MB | 1 | 최저 |

**동작 방식:**
- 3분 걸리는 작은 파일이 2시간 걸리는 큰 파일보다 먼저 처리됩니다.
- 여러 사용자가 동시에 업로드해도 작은 파일이 우선 처리됩니다.
- Worker 개수(`--concurrency`)를 늘려 병렬 처리량을 증가시킬 수 있습니다.

## 프로젝트 구조

```
back/
├── app/
│   ├── main.py              # FastAPI 앱
│   ├── celery_app.py        # Celery 설정
│   ├── tasks.py             # 비동기 작업 정의
│   ├── routers/             # API 라우터
│   │   ├── pdf.py           # PDF 업로드
│   │   ├── tasks.py         # 작업 상태 조회
│   │   ├── chat.py          # 채팅
│   │   └── notebook.py      # 노트북 관리
│   ├── services/
│   │   ├── document/        # PDF 파싱
│   │   ├── database/        # DB 서비스
│   │   └── rag/             # RAG 파이프라인
│   └── models/
│       └── schemas.py       # Pydantic 스키마
├── storage/                 # PDF 파일 저장소
├── .env                     # 환경변수
├── requirements.txt
└── README.md
```

## 개발 팁

### Celery Worker 재시작

코드 변경 후 Worker를 재시작해야 변경사항이 반영됩니다:

```bash
# Worker 프로세스 찾기
ps aux | grep celery

# Worker 종료
pkill -f "celery worker"

# Worker 재시작
celery -A app.celery_app worker --loglevel=info --concurrency=2 --queue=pdf_processing
```

### 로그 확인

```bash
# FastAPI 로그
tail -f logs/api.log

# Celery Worker 로그
tail -f logs/celery.log
```

### Redis 큐 확인

```bash
# Redis CLI 접속
redis-cli

# 큐 길이 확인
LLEN celery

# 모든 키 확인
KEYS *
```

## 트러블슈팅

### 1. Redis 연결 실패

```bash
# Redis 상태 확인
sudo systemctl status redis-server

# Redis 재시작
sudo systemctl restart redis-server
```

### 2. Celery Worker가 작업을 받지 못함

- Redis URL 확인: `.env` 파일의 `REDIS_URL` 확인
- Worker 큐 이름 확인: `pdf_processing` 큐로 작업이 전송되는지 확인
- Flower에서 Worker 상태 확인

### 3. 메모리 부족

- Worker concurrency 줄이기: `--concurrency=1`
- Worker prefetch 줄이기: `celery_app.py`에서 `worker_prefetch_multiplier=1`

### 4. 작업 타임아웃

- Soft timeout 조정: `celery_app.py`에서 `task_soft_time_limit` 증가
- Hard timeout 조정: `task_time_limit` 증가

## 라이센스

MIT License
