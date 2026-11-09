import os
import sys

import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch.nn.functional as F

# Week 7 평가: Colab의 run_gen()이 Drive에 저장한 IP-Adapter FaceID 생성 결과를 FaceNet으로 채점한다.
# FaceID 안의 insightface ArcFace로 잰 값은 유리하게 높게 나오므로, 도구와 관계없는 FaceNet으로도 잰다.
#
# 사용법: Drive의 week7/<tag>/ 폴더를 data/week7/<tag>/ 로 내려받은 뒤
#   .venv/bin/python experiments/week7_facenet_eval.py baseline ...
# tag를 주지 않으면 아래 DEFAULT_TAGS를 채점한다.

DEFAULT_TAGS = ['baseline']
SEEDS = [0, 1, 2]

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        print(f"[경고] {image_path}: 얼굴을 찾지 못했습니다.")
        return None
    return resnet(face.unsqueeze(0))


def similarity(a, b):
    return F.cosine_similarity(a, b).item()


names = ['person_a_1', 'person_a_2', 'person_a_3', 'person_a_4']
tags = sys.argv[1:] or DEFAULT_TAGS
mine = {name: get_embedding(f"data/{name}.jpg") for name in names}
# 타인: Week 5~6과 같은 SimSwap target → 판정선이 Week 6과 같아서 SimSwap 결과와 바로 비교할 수 있음
target = get_embedding("data/week5/target_specific1.png")

# 1) 판정 기준 (Week 5~6과 같은 방식: 본인 최소 / 본인 최소와 타인의 중간)
same = [similarity(mine[a], mine[b]) for i, a in enumerate(names) for b in names[i + 1:]]
other = np.mean([similarity(mine[n], target) for n in names])
fail_line = min(same)
success_line = (fail_line + other) / 2
print(f"FaceNet 본인 최소 {fail_line:.3f} | 타인 평균 {other:.3f}")
print(f"FaceNet 판정 기준: 실패 >= {fail_line:.3f} | 부분 {success_line:.3f} ~ {fail_line:.3f} | 성공 < {success_line:.3f}")


def judge(score):
    if score >= fail_line:
        return "실패"
    return "부분" if score >= success_line else "성공"


# 2) 생성 결과 채점: 사진마다 seed 3개의 평균 ± 표준편차
for tag in tags:
    folder = f"data/week7/{tag}"
    if not os.path.isdir(folder):
        print(f"\n[건너뜀] {folder} 폴더가 없습니다. Drive의 week7/{tag}/를 내려받아 주세요.")
        continue
    print(f"\n=== {tag}: FaceID 생성 결과 vs 내 사진 (FaceNet), seed {len(SEEDS)}개 ===")
    all_values = []
    for name in names:
        values = []
        for seed in SEEDS:
            result = get_embedding(f"{folder}/{name}_seed{seed}.png")
            if result is not None:
                values.append(similarity(result, mine[name]))
        if not values:
            print(f"{name} | 채점할 결과 없음")
            continue
        values = np.array(values)
        all_values.extend(values)
        seeds = " ".join(f"{v:.3f}" for v in values)
        print(f"{name} | seed별 {seeds} | 평균 {values.mean():.3f}±{values.std():.3f} ({judge(values.mean())})")
    if all_values:
        overall = np.mean(all_values)
        print(f"4장 평균 {overall:.3f} ({judge(overall)})")
