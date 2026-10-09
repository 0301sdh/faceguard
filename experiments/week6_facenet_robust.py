import csv
from collections import defaultdict

import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch.nn.functional as F

# Week 6 ⑤ 변형 견고성: Colab의 run_transforms()가 만든 결과를 FaceNet으로 채점한다.
# 보호 사진에 JPEG, Resize, Crop, Blur를 적용한 뒤 SimSwap으로 합성한 결과가
# SimSwap과 관계없는 FaceNet에서도 "나와 덜 닮게" 나오는지 확인한다.
#
# 사용법: Drive의 week6/robust_eps0, robust_eps5_seed0~2 를 data/week6/ 로 내려받은 뒤
#   .venv/bin/python experiments/week6_facenet_robust.py

EPS = 5
SEEDS = [0, 1, 2]
TRANSFORMS = ['none', 'jpeg90', 'jpeg75', 'jpeg50',
              'resize75_bili', 'resize50_bili', 'resize75_area', 'resize50_area',
              'crop10', 'blur0.5', 'blur1.0']

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_embedding(image_path):
    face = mtcnn(Image.open(image_path).convert('RGB'))
    if face is None:
        print(f"[경고] {image_path}: 얼굴을 찾지 못했습니다.")
        return None
    return resnet(face.unsqueeze(0))


def similarity(a, b):
    return F.cosine_similarity(a, b).item()


names = ['person_a_1', 'person_a_2', 'person_a_3', 'person_a_4']
mine = {name: get_embedding(f"data/{name}.jpg") for name in names}
target = get_embedding("data/week5/target_specific1.png")

# 판정 기준 (Week 5와 같은 방식)
same = [similarity(mine[a], mine[b]) for i, a in enumerate(names) for b in names[i + 1:]]
other = sum(similarity(mine[n], target) for n in names) / len(names)
fail_line = min(same)
success_line = (fail_line + other) / 2
print(f"FaceNet 판정 기준: 실패 >= {fail_line:.3f} | 부분 {success_line:.3f} ~ {fail_line:.3f} | 성공 < {success_line:.3f}")


def judge(score):
    if score >= fail_line:
        return "실패"
    return "부분" if score >= success_line else "성공"


# 1) 채점: (보호 안 함 + 보호 seed 3개) × 사진 4장 × 변형 11가지
settings = [(0, None, "robust_eps0")] + [(EPS, s, f"robust_eps{EPS}_seed{s}") for s in SEEDS]
rows = []
for eps, seed, tag in settings:
    print(f"[{tag}] 채점 중")
    for name in names:
        for t in TRANSFORMS:
            result = get_embedding(f"data/week6/{tag}/{name}_eps{eps}_{t}_realign.jpg")
            score = None if result is None else similarity(result, mine[name])   # 합성에 쓴 사진과의 유사도
            rows.append({'tag': tag, 'eps': eps, 'seed': seed, 'name': name, 'transform': t, 'facenet': score})

# 그래프용 저장 (data/ 아래라 git 제외)
with open("data/week6/robust_facenet.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# 2) 변형별 요약: 보호 안 함 vs 보호(seed 평균), 차이, 유지율
scores = defaultdict(list)
for r in rows:
    if r['facenet'] is not None:
        scores[(r['eps'], r['transform'])].append(r['facenet'])
gap_none = np.mean(scores[(0, 'none')]) - np.mean(scores[(EPS, 'none')])   # 변형 없을 때의 보호 효과

print(f"\n=== FaceNet (B) 재정렬, 결과 vs 합성에 쓴 내 사진 (사진 4장, 보호는 seed 평균) ===")
print(f"{'변형':<15}{'보호 안 함':>10}{'보호':>9}{'차이':>8}{'유지율':>8}   판정(보호)")
for t in TRANSFORMS:
    a, b = np.mean(scores[(0, t)]), np.mean(scores[(EPS, t)])
    print(f"{t:<15}{a:>10.3f}{b:>9.3f}{a - b:>8.3f}{(a - b) / gap_none:>8.0%}   {judge(b)}")

# 3) 사진별: 보호 결과 (seed 평균). 어떤 사진이 변형에 약한지 보기
print(f"\n=== 사진별 보호 결과 (FaceNet, seed 평균) ===")
print(f"{'변형':<15}" + "".join(f"{n[-3:]:>9}" for n in names))
for t in TRANSFORMS:
    cells = []
    for n in names:
        values = [r['facenet'] for r in rows if r['eps'] == EPS and r['name'] == n and r['transform'] == t and r['facenet'] is not None]
        cells.append(f"{np.mean(values):>9.3f}")
    print(f"{t:<15}" + "".join(cells))
