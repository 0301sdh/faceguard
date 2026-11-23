# FaceGuard AI 서버 이미지 (CPU용)
# 빌드: docker build -t faceguard-ai .
# 실행: docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/models:/app/models" faceguard-ai
# 모델 파일(models/)은 이미지에 넣지 않고 실행할 때 연결한다 (크기 400MB, 라이선스상 배포하지 않음)
FROM python:3.14-slim

# insightface 설치에 C++ 컴파일러가 필요할 수 있음
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 패키지 먼저 설치 (코드만 바뀌면 이 단계는 다시 하지 않아 빌드가 빠름)
# torch는 CPU 전용판을 받음 (GPU판은 수 GB)
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r server/requirements.txt
# insightface가 화면용 opencv-python을 함께 설치해 headless를 덮어씀 → 화면 라이브러리(libxcb)가 없어 실행 실패
# 화면용을 지우고 headless를 다시 설치
RUN pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless==5.0.0.93

COPY server/ server/

ENV FACEGUARD_MODELS=/app/models
EXPOSE 8000
CMD ["uvicorn", "app:app", "--app-dir", "server", "--host", "0.0.0.0", "--port", "8000"]
