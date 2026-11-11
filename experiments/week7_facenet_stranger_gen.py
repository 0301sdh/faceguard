import glob
import os
from collections import defaultdict

import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import torch.nn.functional as F

# Week 7: FaceID가 '남'을 그렸을 때의 FaceNet 기준선.
# LFW 타인 얼굴로 FaceID가 생성한 이미지(같은 프롬프트, 생성 seed 0~2)를 내 사진 4장과 비교하고,
# 같은 방식(내 사진 4장 평균)으로 baseline·pilot·ensemble 생성 결과도 다시 재서 나란히 놓는다.
#
# 준비: Drive week7/stranger_gen/ → data/week7/stranger_gen/ (파일 이름: <그룹>_<이름>_seed<k>.png)
# 사용법: .venv/bin/python experiments/week7_facenet_stranger_gen.py

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


@torch.no_grad()
def get_embedding(image_path):
    face = mtcnn(Image.open(image_path).convert('RGB'))
    if face is None:
        return None
    return resnet(face.unsqueeze(0))


names = ['person_a_1', 'person_a_2', 'person_a_3', 'person_a_4']
mine = [get_embedding(f"data/{n}.jpg") for n in names]


def mean_vs_me(path):
    # 생성 이미지 1장 vs 내 사진 4장의 평균 유사도 (누구를 넣었는지와 관계없이 같은 잣대)
    e = get_embedding(path)
    if e is None:
        return None
    return float(np.mean([F.cosine_similarity(m, e).item() for m in mine]))


def summarize(label, values):
    v = np.array(values)
    print(f"{label:<26} {len(v):>3}장 | 평균 {v.mean():.3f} | 중앙값 {np.median(v):.3f} | 최대 {v.max():.3f} "
          f"| 0.365 이상 {(v >= 0.365).mean():.0%}")
    return v


# 1) 타인 생성 이미지: 사람별, 그룹별
groups = defaultdict(list)
print("=== 타인 기준선: LFW 타인 얼굴로 FaceID 생성 → 내 사진 4장과 비교 (FaceNet) ===")
per_person = defaultdict(list)
for path in sorted(glob.glob("data/week7/stranger_gen/*.png")):
    stem = os.path.basename(path)[:-4]
    group, rest = stem.split("_", 1)
    person = rest.rsplit("_seed", 1)[0]
    s = mean_vs_me(path)
    if s is None:
        print(f"[경고] {path}: 얼굴을 찾지 못했습니다.")
        continue
    per_person[(group, person)].append(s)
    groups[group].append(s)

for (group, person), vals in sorted(per_person.items(), key=lambda kv: -np.mean(kv[1])):
    print(f"[{group}] {person:<20} | {' '.join(f'{v:.3f}' for v in vals)} | 평균 {np.mean(vals):.3f}")

# 2) 내 얼굴로 만든 생성 결과도 같은 잣대(내 사진 4장 평균)로
print("\n=== 같은 잣대 비교: 생성 이미지 vs 내 사진 4장 평균 (FaceNet) ===")
for group in ["random", "asian"]:
    if groups[group]:
        summarize(f"타인 ({group})", groups[group])
for tag_label, pattern in [("baseline (보호 안 함)", "data/week7/baseline/*.png"),
                           ("pilot (Week 6 보호)", "data/week7/pilot_eps5_seed0/*.png"),
                           ("ensemble (seed 0~2)", "data/week7/ens_eps5_seed[012]/*.png")]:
    vals = [s for s in (mean_vs_me(p) for p in sorted(glob.glob(pattern))) if s is not None]
    if vals:
        summarize(tag_label, vals)
