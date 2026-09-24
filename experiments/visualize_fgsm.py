"""
Week 2: FGSM 결과 시각화

같은 이미지에 대해 epsilon별로 [원본 | perturbation | 공격 이미지]를 나란히 그려서
사람 눈에 얼마나 티가 나는지, 모델 예측이 어떻게 바뀌는지 확인한다.
"""

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from fgsm import fgsm_attack
from train_mnist import SimpleCNN


def main():
    model = SimpleCNN()
    model.load_state_dict(torch.load("experiments/mnist_cnn.pt"))
    model.eval()

    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )

    # 모델이 원래 맞히는 이미지 하나를 고른다
    for image, label in test_dataset:
        image = image.unsqueeze(0)  # (1, 28, 28) -> (1, 1, 28, 28) 배치 차원 추가
        label = torch.tensor([label])
        if model(image).argmax(dim=1).item() == label.item():
            break

    epsilons = [0.05, 0.1, 0.2, 0.3]
    fig, axes = plt.subplots(len(epsilons), 3, figsize=(7, 2.4 * len(epsilons)))

    for row, epsilon in enumerate(epsilons):
        adv_image = fgsm_attack(model, image, label, epsilon)
        perturbation = adv_image - image
        adv_pred = model(adv_image).argmax(dim=1).item()

        axes[row][0].imshow(image[0, 0], cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_title(f"original (pred {label.item()})")

        # perturbation은 -epsilon ~ +epsilon 범위라 0이 회색이 되도록 대칭으로 표시
        axes[row][1].imshow(perturbation[0, 0], cmap="gray", vmin=-epsilon, vmax=epsilon)
        axes[row][1].set_title(f"perturbation (eps={epsilon})")

        axes[row][2].imshow(adv_image[0, 0], cmap="gray", vmin=0, vmax=1)
        axes[row][2].set_title(f"attacked (pred {adv_pred})")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    save_path = "experiments/fgsm_visualization.png"
    plt.savefig(save_path, dpi=120)
    print(f"저장 완료: {save_path}")


if __name__ == "__main__":
    main()
