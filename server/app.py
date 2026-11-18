"""FaceGuard AI 서버 (Week 9).

실행: .venv/bin/uvicorn app:app --app-dir server --port 8000
문서: http://127.0.0.1:8000/docs (브라우저에서 사진을 올려 바로 시험할 수 있음)

POST /protect : 사진 업로드 → 보호된 사진(PNG) 반환
GET  /health  : 서버가 켜져 있고 모델이 준비됐는지
"""
import os
import threading
from contextlib import asynccontextmanager

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import Response  # noqa: E402

from protect import FaceGuard  # noqa: E402

MODEL_DIR = os.environ.get('FACEGUARD_MODELS', 'models')
MAX_BYTES = 20 * 1024 * 1024        # 업로드 최대 20MB
MAX_PIXELS = 50_000_000             # 약 8000×6000. 이보다 크면 거절 (메모리 보호)

state = {}
lock = threading.Lock()             # 모델이 하나라 한 번에 한 장씩 처리 (난수·GPU 메모리 공유)


@asynccontextmanager
async def lifespan(app):
    # 서버를 켤 때 모델을 한 번만 불러 둔다 (요청마다 부르면 매번 수 초가 걸림)
    state['fg'] = FaceGuard(MODEL_DIR)
    yield
    state.clear()


app = FastAPI(title='FaceGuard AI', lifespan=lifespan)


@app.get('/health')
def health():
    fg = state.get('fg')
    return {'status': 'ok' if fg else 'loading', 'device': fg.device if fg else None}


@app.post('/protect', responses={200: {'content': {'image/png': {}}, 'description': '보호된 사진 (PNG)'}})
def protect(photo: UploadFile = File(..., description='보호할 사진 (JPEG, PNG 등)')):
    # def(async 아님) → FastAPI가 별도 스레드에서 실행해서, 처리 중에도 /health는 응답함
    data = photo.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, detail='파일이 너무 큽니다 (최대 20MB)')
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # 휴대폰 사진의 회전 정보(EXIF)도 반영
    if img is None:
        raise HTTPException(400, detail='사진 파일이 아닙니다')
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise HTTPException(413, detail='사진이 너무 큽니다 (최대 약 5000만 화소)')

    with lock:
        out, info = state['fg'].protect(img)
    if out is None:
        raise HTTPException(422, detail='얼굴을 찾지 못했습니다')

    ok, png = cv2.imencode('.png', out)   # PNG(무손실): JPEG로 저장하면 노이즈가 일부 지워짐
    if not ok:
        raise HTTPException(500, detail='결과를 저장하지 못했습니다')
    w, h = info['size']
    return Response(png.tobytes(), media_type='image/png', headers={
        'Content-Disposition': 'inline; filename="protected.png"',
        'X-Output-Size': f'{w}x{h}',
        'X-Processing-Seconds': f"{info['seconds']['total']:.1f}",
    })
