import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch.nn.functional as F

# Week 5 pilot 평가: SimSwap 내부 ArcFace로 만든 보호 사진이
# SimSwap과 관계없는 FaceNet으로 채점해도 효과가 있는지 확인한다 (전이성).

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


def psnr(path_a, path_b):
    # 원본 얼굴(eps0)과 보호 얼굴의 화질 차이 (dB, 높을수록 원본과 비슷)
    a = np.asarray(Image.open(path_a).convert('RGB'), dtype=np.float64)
    b = np.asarray(Image.open(path_b).convert('RGB'), dtype=np.float64)
    mse = ((a - b) ** 2).mean()
    return 10 * np.log10(255 ** 2 / mse)


names = ['person_a_1', 'person_a_2', 'person_a_3', 'person_a_4']
epsilons = [0, 4, 8, 12]
mine = {name: get_embedding(f"data/{name}.jpg") for name in names}
target = get_embedding("data/week5/target_specific1.png")

# 1) 판정 기준: 내 사진끼리(본인)와 나 vs target(타인)
same = [similarity(mine[a], mine[b]) for i, a in enumerate(names) for b in names[i + 1:]]
other = sum(similarity(mine[n], target) for n in names) / len(names)
fail_line = min(same)                       # 진짜 내 사진끼리의 최솟값 이상이면 "나"로 볼 수 있음
success_line = (fail_line + other) / 2      # 본인 최소와 타인의 중간보다 아래면 "보호 성공"

print("=== FaceNet 판정 기준 ===")
print(f"본인 (6쌍) 평균 {sum(same) / len(same):.3f}, 최소 {fail_line:.3f} | 타인(target) {other:.3f}")
print(f"보호 실패 >= {fail_line:.3f} | 부분 보호 {success_line:.3f} ~ {fail_line:.3f} | 보호 성공 < {success_line:.3f}")


def judge(score):
    if score >= fail_line:
        return "실패"
    if score >= success_line:
        return "부분"
    return "성공"


# 2) pilot 결과 채점: 결과 vs 합성에 쓴 사진 / 쓰지 않은 3장 평균
for mode in ['realign', 'direct']:
    label = "(B) 재정렬" if mode == 'realign' else "(A) 직접"
    print(f"\n=== {label}: SimSwap 결과 vs 내 사진 (FaceNet) ===")
    for name in names:
        for e in epsilons:
            result = get_embedding(f"data/week5/output_pilot/{name}_eps{e}_{mode}.jpg")
            if result is None:
                continue
            used = similarity(result, mine[name])
            others = [similarity(result, mine[o]) for o in names if o != name]
            others_avg = sum(others) / len(others)
            quality = "" if e == 0 else (
                f" | PSNR {psnr(f'data/week5/protected/{name}_eps0.png', f'data/week5/protected/{name}_eps{e}.png'):.1f}dB")
            print(f"{name} eps={e:>2}/255 | 합성에 쓴 사진 {used:.3f} ({judge(used)}) | "
                  f"쓰지 않은 3장 평균 {others_avg:.3f} ({judge(others_avg)}){quality}")
