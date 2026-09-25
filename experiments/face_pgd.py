"""
Week 4: Face Embedding 기반 보호 (Identity loss)

얼굴 사진에 PGD로 perturbation을 넣어서, FaceNet 임베딩이 원본과 멀어지게 만든다.
    L_id = cos( A(x+δ), A(x) )  → 최소화   (A: FaceNet 임베딩)

단순화: MTCNN으로 먼저 잘라낸 160x160 얼굴에 공격한다.
"""

import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# post_process=False: 정규화하지 않은 0~255 값으로 얼굴을 받는다 (공격은 0~1 범위에서 하기 위해)
mtcnn = MTCNN(image_size=160, margin=0, post_process=False)
resnet = InceptionResnetV1(pretrained="vggface2").eval()  # 공격(노이즈 생성)에 사용
# 같은 구조, 다른 데이터로 학습한 모델. 노이즈 생성에는 쓰지 않고 평가에만 사용 (전이성 확인)
resnet_casia = InceptionResnetV1(pretrained="casia-webface").eval()


def load_face(path):
    face = mtcnn(Image.open(path).convert("RGB"))  # (3, 160, 160), 0~255
    if face is None:
        raise ValueError(f"{path}: 얼굴을 찾지 못했습니다.")
    return (face / 255).unsqueeze(0)  # (1, 3, 160, 160), 0~1


def embed(face, model=resnet):
    # FaceNet이 학습할 때 쓴 정규화: (픽셀 - 127.5) / 128
    return model((face * 255 - 127.5) / 128)


def cos(a, b):
    return F.cosine_similarity(a, b).item()


def pgd_identity(face, epsilon, steps):
    alpha = 2.5 * epsilon / steps  # Week 3에서 정한 scaled 보폭
    orig_emb = embed(face).detach()  # 기준점: 원본 임베딩 (고정)
    adv = face.clone().detach()

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)

        loss = F.cosine_similarity(embed(adv), orig_emb).mean()
        resnet.zero_grad()
        loss.backward()

        # 유사도를 "낮추는" 것이 목표이므로 gradient 반대 방향으로 이동
        adv = adv - alpha * adv.grad.sign()

        perturbation = torch.clamp(adv - face, -epsilon, epsilon)
        adv = torch.clamp(face + perturbation, 0, 1)

    return adv.detach()


def png_roundtrip(face):
    # PNG로 저장하면 0~255 정수로 반올림되므로 같은 효과를 흉내 낸다
    return torch.round(face * 255) / 255


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(1 / mse)).item()


def main():
    face_a1 = load_face("data/person_a_1.jpg")
    face_a2 = load_face("data/person_a_2.jpg")
    face_b1 = load_face("data/person_b_1.jpg")

    with torch.no_grad():
        emb_a1, emb_a2, emb_b1 = embed(face_a1), embed(face_a2), embed(face_b1)

    print("=== 기준값 (보호 전) ===")
    print(f"a1 vs a2 (본인): {cos(emb_a1, emb_a2):.3f}")
    print(f"a1 vs b1 (타인): {cos(emb_a1, emb_b1):.3f}  ← 보호 후 이 값 아래로 내려가면 성공\n")

    steps = 20
    epsilons = [2 / 255, 3 / 255, 4 / 255, 8 / 255, 16 / 255]
    results = []

    print(f"=== PGD Identity 공격 (steps={steps}) ===")
    print(f"{'eps':<8}{'prot_vs_a1':<12}{'prot_vs_a2':<12}{'png_vs_a2':<11}"
          f"{'L2':<7}{'Linf':<8}{'PSNR':<8}{'time(s)'}")

    for epsilon in epsilons:
        start = time.time()
        protected = pgd_identity(face_a1, epsilon, steps)
        sec = time.time() - start

        saved = png_roundtrip(protected)
        with torch.no_grad():
            emb_prot = embed(protected)
            emb_saved = embed(saved)

        diff = (protected - face_a1).flatten()
        print(f"{f'{epsilon * 255:.0f}/255':<8}"
              f"{cos(emb_prot, emb_a1):<12.3f}{cos(emb_prot, emb_a2):<12.3f}"
              f"{cos(emb_saved, emb_a2):<11.3f}"
              f"{diff.norm().item():<7.2f}{diff.abs().max().item():<8.4f}"
              f"{psnr(protected, face_a1):<8.1f}{sec:.1f}")
        results.append((epsilon, protected))

    print_transfer(face_a1, face_a2, face_b1, results)
    save_visualization(face_a1, results)
    compare_steps(face_a1, face_a2)


def compare_steps(face_a1, face_a2):
    # 얼굴에서도 MNIST처럼 10~20 steps가 적당한지, steps가 전이성에 영향을 주는지 확인
    with torch.no_grad():
        vgg_a2 = embed(face_a2)
        casia_a2 = embed(face_a2, resnet_casia)

    print("\n=== 얼굴에서 steps 비교 (scaled alpha, steps=1은 FGSM과 같음) ===")
    print("성공 기준: vggface2 < 0.372, casia < 0.397")
    print(f"{'eps':<8}{'steps':<7}{'vgg_vs_a2':<11}{'casia_vs_a2':<13}{'PSNR':<8}{'time(s)'}")

    for epsilon in [4 / 255, 8 / 255]:
        for steps in [1, 5, 10, 20, 40]:
            start = time.time()
            protected = pgd_identity(face_a1, epsilon, steps)
            sec = time.time() - start

            with torch.no_grad():
                vgg_score = cos(embed(protected), vgg_a2)
                casia_score = cos(embed(protected, resnet_casia), casia_a2)
            print(f"{f'{epsilon * 255:.0f}/255':<8}{steps:<7}{vgg_score:<11.3f}"
                  f"{casia_score:<13.3f}{psnr(protected, face_a1):<8.1f}{sec:.1f}")


def print_transfer(face_a1, face_a2, face_b1, results):
    # vggface2로 만든 보호 사진을, 공격에 쓰지 않은 casia-webface로 평가
    with torch.no_grad():
        vgg_a2 = embed(face_a2)
        casia_a1, casia_a2, casia_b1 = (embed(f, resnet_casia) for f in (face_a1, face_a2, face_b1))

    casia_threshold = cos(casia_a1, casia_b1)
    print("\n=== 전이성: casia-webface로 평가 (노이즈는 vggface2로 생성) ===")
    print(f"casia 기준값: a1 vs a2 (본인) {cos(casia_a1, casia_a2):.3f}, "
          f"a1 vs b1 (타인) {casia_threshold:.3f}  ← casia에서는 이 값 아래면 성공")
    print(f"{'eps':<8}{'vgg_vs_a2':<11}{'casia_vs_a2':<13}{'casia_png_vs_a2':<17}{'casia 성공'}")

    for epsilon, protected in results:
        with torch.no_grad():
            vgg_score = cos(embed(protected), vgg_a2)
            casia_score = cos(embed(protected, resnet_casia), casia_a2)
            casia_png = cos(embed(png_roundtrip(protected), resnet_casia), casia_a2)
        ok = "O" if casia_score < casia_threshold else "X"
        print(f"{f'{epsilon * 255:.0f}/255':<8}{vgg_score:<11.3f}{casia_score:<13.3f}"
              f"{casia_png:<17.3f}{ok}")


def save_visualization(face, results):
    fig, axes = plt.subplots(len(results), 3, figsize=(7.5, 2.6 * len(results)))
    to_img = lambda t: t[0].permute(1, 2, 0).numpy()  # (1,3,H,W) -> (H,W,3)

    for row, (epsilon, protected) in enumerate(results):
        perturbation = protected - face
        # perturbation은 너무 작아서 보이지 않으므로 0.5를 중심으로 확대해서 표시
        amplified = (perturbation / (2 * epsilon) + 0.5).clamp(0, 1)

        axes[row][0].imshow(to_img(face))
        axes[row][0].set_title("original")
        axes[row][1].imshow(to_img(amplified))
        axes[row][1].set_title(f"perturbation (x{1 / (2 * epsilon):.0f})")
        axes[row][2].imshow(to_img(protected))
        axes[row][2].set_title(f"protected (eps={epsilon * 255:.0f}/255)")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    save_path = "experiments/face_pgd_visualization.png"
    plt.savefig(save_path, dpi=110)
    print(f"\n시각화 저장: {save_path}")


if __name__ == "__main__":
    main()
