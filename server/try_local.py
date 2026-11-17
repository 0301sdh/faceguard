"""이 컴퓨터에서 보호 코드가 돌아가는지, 얼마나 걸리는지 확인.

사용법: .venv/bin/python server/try_local.py [사진 경로 ...] [--device cpu|mps|cuda]
결과: data/week9/local/<이름>.png (data/는 git에 올리지 않음)
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')   # Mac GPU가 지원하지 않는 연산은 CPU로
from protect import FaceGuard  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('photos', nargs='*', default=['data/person_a_1.jpg'])
parser.add_argument('--device', default=None)
args = parser.parse_args()

fg = FaceGuard('models', device=args.device)
print('device:', fg.device)
os.makedirs('data/week9/local', exist_ok=True)
for path in args.photos:
    photo = cv2.imread(path)
    out, info = fg.protect(photo)
    name = os.path.splitext(os.path.basename(path))[0]
    if out is None:
        print(f'{name} | {info}')
        continue
    cv2.imwrite(f'data/week9/local/{name}.png', out)
    sec = ' '.join(f'{k} {v:.1f}s' for k, v in info['seconds'].items())
    print(f'{name} {photo.shape[1]}×{photo.shape[0]} → {info["size"][0]}×{info["size"][1]} | {sec}')
