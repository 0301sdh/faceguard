import glob
import os
import random

import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import torch.nn.functional as F

# Week 7: FaceNet 눈금 다시 잡기.
# 지금까지 FaceNet 성공선(0.365)은 타인 1명(target_specific1, 0.044)으로 정했다.
# 공개 얼굴 데이터셋 LFW에서 타인을 많이 뽑아 "내 사진 vs 남" 유사도가 어떻게 퍼져 있는지 보고,
# FaceID ensemble 결과(FaceNet 0.438)가 그 분포의 어디쯤인지 확인한다.
#
# 준비: data/lfw/lfw_funneled/<사람 이름>/*.jpg (LFW funneled, 연구 공개용)
# 사용법: .venv/bin/python experiments/week7_facenet_calibration.py [뽑을 사람 수, 기본 500]

import sys

N_PEOPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 500
LFW_DIR = "data/lfw/lfw_funneled"
random.seed(0)

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


@torch.no_grad()
def get_embedding(image_path):
    face = mtcnn(Image.open(image_path).convert('RGB'))
    if face is None:
        return None
    return resnet(face.unsqueeze(0))


names = ['person_a_1', 'person_a_2', 'person_a_3', 'person_a_4']
mine = {name: get_embedding(f"data/{name}.jpg") for name in names}

# 사람마다 사진 1장 (같은 사람이 여러 번 들어가지 않게)
people = sorted(d for d in os.listdir(LFW_DIR) if os.path.isdir(f"{LFW_DIR}/{d}"))
picked = random.sample(people, N_PEOPLE)

sims = []          # (내 사진 k, 타인) 모든 쌍의 유사도
per_person = []    # 타인마다 내 사진 4장과의 평균
skipped = 0
for person in picked:
    path = sorted(glob.glob(f"{LFW_DIR}/{person}/*.jpg"))[0]
    e = get_embedding(path)
    if e is None:
        skipped += 1
        continue
    row = [F.cosine_similarity(mine[n], e).item() for n in names]
    sims.extend(row)
    per_person.append((np.mean(row), person))

sims = np.array(sims)
print(f"LFW 타인 {len(per_person)}명 (얼굴 못 찾음 {skipped}명) × 내 사진 4장 = {len(sims)}쌍")
print(f"평균 {sims.mean():.3f} | 표준편차 {sims.std():.3f} | 최대 {sims.max():.3f}")
print("백분위: " + " | ".join(f"{p}% {np.percentile(sims, p):.3f}" for p in [50, 90, 95, 99, 99.9]))

# 지금까지 쓴 기준선·결과가 타인 분포의 어디쯤인가
for label, value in [("기존 타인 (target_specific1)", 0.044),
                     ("기존 성공선", 0.365),
                     ("FaceID ensemble 노이즈 seed 2", 0.325),
                     ("FaceID ensemble 평균", 0.438),
                     ("FaceID ensemble 노이즈 seed 0", 0.510),
                     ("Week 6 SimSwap 보호 평균", 0.297),
                     ("기존 실패선 (내 사진끼리 최소)", 0.686)]:
    print(f"{label:<30} {value:.3f} → 타인 쌍 중 이보다 높은 비율 {(sims > value).mean():.2%}")

print("\n나와 가장 비슷한 타인 10명 (내 사진 4장 평균):")
for score, person in sorted(per_person, reverse=True)[:10]:
    print(f"  {score:.3f}  {person}")
