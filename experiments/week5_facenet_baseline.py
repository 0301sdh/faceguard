from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch.nn.functional as F

# Week 5 baseline: SimSwap 합성 결과를 FaceNet으로 평가한다.
# SimSwap 안의 ArcFace로 잰 값(평균 0.73)은 SimSwap이 학습 때 맞추려던 모델이라 유리하게 나온다.
# 그래서 SimSwap과 관계없는 FaceNet(Week 1과 같은 모델)으로도 "결과가 나인가"를 확인한다.

# Week 1과 같은 모델: 얼굴 검출(160x160으로 잘라냄) + 임베딩(VGGFace2로 사전학습)
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_embedding(image_path):
    # PNG는 투명도 채널(RGBA)이 있을 수 있어서 RGB로 맞춘다
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)  # 얼굴을 찾으면 (3, 160, 160) 텐서, 못 찾으면 None

    if face is None:
        print(f"[경고] {image_path}: 얼굴을 찾지 못했습니다.")
        return None

    return resnet(face.unsqueeze(0))  # (1, 512)


def similarity(a, b):
    return F.cosine_similarity(a, b).item()


names = ['person_a_1', 'person_a_2', 'person_a_3', 'person_a_4']

# 1) 임베딩을 한 번씩만 구해 둔다
mine = {name: get_embedding(f"data/{name}.jpg") for name in names}                 # 내 원본 사진
swaps = {name: get_embedding(f"data/week5/swap_{name}.jpg") for name in names}     # 합성 결과
target = get_embedding("data/week5/target_specific1.png")                           # target 원본

# 2) 기준값: 이번 사진들로 다시 잰 "본인끼리"와 "나 vs target"
print("=== 기준값 ===")
same_person = [similarity(mine[a], mine[b]) for i, a in enumerate(names) for b in names[i + 1:]]
print(f"내 사진끼리 (본인, {len(same_person)}쌍 평균): {sum(same_person) / len(same_person):.3f}")
mine_vs_target = [similarity(mine[name], target) for name in names]
print(f"내 사진 vs target (타인, 평균):  {sum(mine_vs_target) / len(mine_vs_target):.3f}")
print("참고: Week 1 기준값 본인 0.743 / 타인 0.372")

# 3) 합성 결과 평가
print("\n=== 합성 결과 (FaceNet) ===")
for name in names:
    if swaps[name] is None:
        continue

    used = similarity(swaps[name], mine[name])                                         # 합성에 쓴 사진
    others = [similarity(swaps[name], mine[o]) for o in names if o != name]            # 합성에 쓰지 않은 3장
    vs_target = similarity(swaps[name], target)

    print(f"{name}: 결과 vs 합성에 쓴 사진 {used:.3f} | "
          f"결과 vs 쓰지 않은 사진 평균 {sum(others) / len(others):.3f} | "
          f"결과 vs target {vs_target:.3f}")
