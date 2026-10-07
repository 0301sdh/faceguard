import re
import sys

import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import torch.nn.functional as F

# Week 6 평가: Colab의 run()이 Drive에 저장한 결과를 FaceNet으로 채점한다.
# ArcFace로 만든 노이즈가 SimSwap과 관계없는 FaceNet에서도 효과가 있는지 확인 (전이성).
#
# 사용법: Drive의 week6/<tag>/ 폴더를 data/week6/<tag>/ 로 내려받은 뒤
#   .venv/bin/python experiments/week6_facenet_eval.py shift_eps8 affine_eps8 ...
# tag를 주지 않으면 아래 DEFAULT_TAGS를 채점한다.

DEFAULT_TAGS = ['shift_eps8', 'affine_eps8', 'shift_eps12', 'affine_eps12']

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


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)


def psnr(path_a, path_b):
    # 노이즈 없는 얼굴(eps0)과 보호 얼굴의 픽셀 차이 (dB, 높을수록 원본과 비슷)
    a, b = load_rgb(path_a), load_rgb(path_b)
    return 10 * np.log10(255 ** 2 / ((a - b) ** 2).mean())


def ssim(path_a, path_b):
    # 밝기·대비·구조가 얼마나 비슷한지 (0~1, 1에 가까울수록 원본과 비슷)
    # 표준 설정: 11×11 가우시안 창(sigma 1.5)으로 동네마다 계산해 평균. 채널별로 계산 후 평균
    a = torch.from_numpy(load_rgb(path_a)).permute(2, 0, 1)[:, None]   # (3, 1, H, W)
    b = torch.from_numpy(load_rgb(path_b)).permute(2, 0, 1)[:, None]
    coords = torch.arange(11, dtype=torch.float64) - 5
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    window = (g[:, None] * g[None, :] / g.sum() ** 2)[None, None]       # (1, 1, 11, 11)

    def blur(x):
        return F.conv2d(x, window)

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return ssim_map.mean().item()


names = ['person_a_1', 'person_a_2', 'person_a_3', 'person_a_4']
tags = sys.argv[1:] or DEFAULT_TAGS
mine = {name: get_embedding(f"data/{name}.jpg") for name in names}
target = get_embedding("data/week5/target_specific1.png")

# 1) 판정 기준 (Week 5와 같은 방식: 본인 최소 / 본인 최소와 타인의 중간)
same = [similarity(mine[a], mine[b]) for i, a in enumerate(names) for b in names[i + 1:]]
other = sum(similarity(mine[n], target) for n in names) / len(names)
fail_line = min(same)
success_line = (fail_line + other) / 2
print(f"FaceNet 판정 기준: 실패 >= {fail_line:.3f} | 부분 {success_line:.3f} ~ {fail_line:.3f} | 성공 < {success_line:.3f}")


def judge(score):
    if score >= fail_line:
        return "실패"
    return "부분" if score >= success_line else "성공"


def score_line(label, result_path, name, extra=""):
    result = get_embedding(result_path)
    if result is None:
        return
    used = similarity(result, mine[name])
    others = [similarity(result, mine[o]) for o in names if o != name]
    others_avg = sum(others) / len(others)
    print(f"{label:<14} {name} | 합성에 쓴 사진 {used:.3f} ({judge(used)}) | "
          f"쓰지 않은 3장 평균 {others_avg:.3f} ({judge(others_avg)}){extra}")


# 2) 기준선: 보호 안 한 합성 결과 (Week 5 pilot의 eps0 재정렬 결과. 무작위 요소가 없어 Week 6과 같음)
print("\n=== 기준선: 보호 안 함 (eps=0) ===")
for name in names:
    score_line("eps0", f"data/week5/output_pilot/{name}_eps0_realign.jpg", name)

# 3) Week 6 결과 채점
for tag in tags:
    eps = int(re.search(r'eps(\d+)', tag).group(1))
    print(f"\n=== {tag}: (B) 재정렬, SimSwap 결과 vs 내 사진 (FaceNet) ===")
    for name in names:
        clean = f"data/week5/protected/{name}_eps0.png"
        protected = f"data/week6/{tag}/{name}_eps{eps}.png"
        quality = f" | PSNR {psnr(clean, protected):.1f}dB | SSIM {ssim(clean, protected):.3f}"
        score_line(tag, f"data/week6/{tag}/{name}_eps{eps}_realign.jpg", name, quality)
