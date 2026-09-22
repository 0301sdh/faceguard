from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

# 얼굴 검출 모델: 사진에서 얼굴 위치를 찾아 160x160으로 잘라줌
mtcnn = MTCNN(image_size=160, margin=0)

# 얼굴 임베딩 모델: 잘라낸 얼굴을 512차원 벡터로 변환 (VGGFace2로 사전학습됨)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_embedding(image_path):
    img = Image.open(image_path)
    face = mtcnn(img)  # 얼굴을 찾으면 (3, 160, 160) 텐서, 못 찾으면 None

    if face is None:
        print(f"[경고] {image_path}: 얼굴을 찾지 못했습니다.")
        return None

    # 모델은 배치(batch) 단위 입력을 기대하므로 차원을 하나 추가 (1, 3, 160, 160)
    embedding = resnet(face.unsqueeze(0))
    return embedding


# 세 장의 임베딩 추출
emb_a1 = get_embedding("data/person_a_1.jpg")
emb_a2 = get_embedding("data/person_a_2.jpg")
emb_b1 = get_embedding("data/person_b_1.jpg")

# 코사인 유사도 계산 (1에 가까울수록 같은 사람, 0에 가까울수록 다른 사람)
if emb_a1 is not None and emb_a2 is not None:
    sim_same_person = torch.nn.functional.cosine_similarity(emb_a1, emb_a2)
    print("본인 vs 본인 (다른 사진) 유사도:", sim_same_person.item())

if emb_a1 is not None and emb_b1 is not None:
    sim_diff_person = torch.nn.functional.cosine_similarity(emb_a1, emb_b1)
    print("본인 vs 타인 유사도:", sim_diff_person.item())

# 강아지 사진 (얼굴 없는 경우 테스트)
emb_dog = get_embedding("data/dog.jpg")