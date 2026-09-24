"""
Week 2: FGSM 공격 구현

학습된 MNIST 모델(experiments/mnist_cnn.pt)에 FGSM을 적용하고,
epsilon 크기에 따라 모델 정확도가 어떻게 변하는지 확인한다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_mnist import SimpleCNN


def fgsm_attack(model, image, label, epsilon, use_random = False):
    # 입력 이미지에 대한 gradient를 구하기 위해 필요
    image = image.clone().detach().requires_grad_(True)

    output = model(image)
    loss = nn.CrossEntropyLoss()(output, label)

    model.zero_grad()
    loss.backward()  # image.grad 에 "픽셀을 바꾸면 loss가 얼마나 변하는지"가 저장됨

    if use_random:
        perturbation = epsilon * torch.randn_like(image).sign()
    else:
        perturbation = epsilon * image.grad.sign()
    adv_image = image + perturbation

    # 픽셀 값은 0~1 범위여야 하므로 벗어난 값은 잘라냄
    adv_image = torch.clamp(adv_image, 0, 1)
    return adv_image.detach()


def main():
    model = SimpleCNN()
    model.load_state_dict(torch.load("experiments/mnist_cnn.pt"))
    model.eval()

    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    num_samples = 2000  # 시간 절약을 위해 테스트 이미지 일부만 사용

    for epsilon in epsilons:
        correct = 0
        example_printed = False

        for i, (image, label) in enumerate(test_loader):
            if i >= num_samples:
                break

            # 원래부터 틀리는 이미지는 공격 효과를 재기 어려우므로 제외
            if model(image).argmax(dim=1).item() != label.item():
                continue

            adv_image = fgsm_attack(model, image, label, epsilon)
            adv_pred = model(adv_image).argmax(dim=1).item()

            if adv_pred == label.item():
                correct += 1
            elif not example_printed and epsilon > 0:
                max_change = (adv_image - image).abs().max().item()
                print(f"  예시: 정답 {label.item()} -> 공격 후 예측 {adv_pred} "
                      f"(픽셀 최대 변화량 {max_change:.2f})")
                example_printed = True

        print(f"epsilon={epsilon:<5} 공격 후 정확도: {correct}/{num_samples}")


if __name__ == "__main__":
    main()
