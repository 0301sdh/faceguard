"""
Week 3: PGD 공격 구현 및 FGSM과 비교

비교 항목: 공격 후 정확도, perturbation 크기(L2, L∞), 반복 횟수, 실행 시간
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fgsm import fgsm_attack
from train_mnist import SimpleCNN


def pgd_attack(model, image, label, epsilon, alpha, steps):
    adv_image = image.clone().detach()

    for _ in range(steps):
        # 매 반복마다 새로 gradient를 구하기 위해 이전 계산 기록을 끊고 다시 추적 시작
        adv_image = adv_image.detach().requires_grad_(True)

        loss = nn.CrossEntropyLoss()(model(adv_image), label)
        model.zero_grad()
        loss.backward()

        # 작은 걸음(alpha)만큼 loss가 커지는 방향으로 이동
        adv_image = adv_image + alpha * adv_image.grad.sign()

        # projection: 원본에서 epsilon 이상 벗어나지 않게 되돌림
        perturbation = torch.clamp(adv_image - image, -epsilon, epsilon)
        adv_image = torch.clamp(image + perturbation, 0, 1)

    return adv_image.detach()


def evaluate(model, loader, attack_fn, num_batches):
    correct, total = 0, 0
    l2_sum, linf_max = 0.0, 0.0
    start = time.time()

    for i, (image, label) in enumerate(loader):
        if i >= num_batches:
            break

        # 원래부터 틀리는 이미지는 공격 효과를 재기 어려우므로 제외
        with torch.no_grad():
            keep = model(image).argmax(dim=1) == label
        image, label = image[keep], label[keep]

        adv_image = attack_fn(image, label)

        with torch.no_grad():
            pred = model(adv_image).argmax(dim=1)

        diff = (adv_image - image).flatten(1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        l2_sum += diff.norm(dim=1).sum().item()
        linf_max = max(linf_max, diff.abs().max().item())

    return {
        "acc": correct / total,
        "l2": l2_sum / total,
        "linf": linf_max,
        "sec": time.time() - start,
    }


def compare_fgsm_pgd(model, loader, num_batches):
    print("=== FGSM vs PGD (epsilon별) ===")
    steps = 10
    print(f"{'eps':<6}{'method':<8}{'steps':<7}{'acc(%)':<9}{'L2':<8}{'Linf':<8}{'time(s)'}")

    for epsilon in [0.05, 0.1, 0.15, 0.2, 0.3]:
        alpha = epsilon / 4

        attacks = [
            ("FGSM", 1, lambda x, y: fgsm_attack(model, x, y, epsilon)),
            ("PGD", steps, lambda x, y: pgd_attack(model, x, y, epsilon, alpha, steps)),
        ]
        for name, n_iter, attack_fn in attacks:
            r = evaluate(model, loader, attack_fn, num_batches)
            print(f"{epsilon:<6}{name:<8}{n_iter:<7}{r['acc'] * 100:<9.1f}"
                  f"{r['l2']:<8.2f}{r['linf']:<8.3f}{r['sec']:.1f}")


def compare_steps(model, loader, num_batches, epsilon=0.15):
    print(f"\n=== 반복 횟수(steps)별 PGD (epsilon={epsilon}) ===")

    r = evaluate(model, loader, lambda x, y: fgsm_attack(model, x, y, epsilon), num_batches)
    print(f"FGSM 기준: acc {r['acc'] * 100:.1f}%, L2 {r['l2']:.2f}, time {r['sec']:.1f}s\n")

    alpha_rules = [
        ("fixed", lambda steps: epsilon / 4),          # 보폭 고정
        ("scaled", lambda steps: 2.5 * epsilon / steps),  # 총 이동 거리 고정
    ]

    print(f"{'alpha':<8}{'steps':<7}{'alpha_val':<10}{'acc(%)':<9}{'L2':<8}{'time(s)'}")
    for rule_name, alpha_fn in alpha_rules:
        for steps in [1, 5, 10, 20, 40]:
            alpha = alpha_fn(steps)
            r = evaluate(
                model, loader,
                lambda x, y: pgd_attack(model, x, y, epsilon, alpha, steps),
                num_batches,
            )
            print(f"{rule_name:<8}{steps:<7}{alpha:<10.4f}{r['acc'] * 100:<9.1f}"
                  f"{r['l2']:<8.2f}{r['sec']:.1f}")


def main():
    model = SimpleCNN()
    model.load_state_dict(torch.load("experiments/mnist_cnn.pt"))
    model.eval()

    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )
    loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    num_batches = 20  # 2000장

    compare_fgsm_pgd(model, loader, num_batches)
    compare_steps(model, loader, num_batches)


if __name__ == "__main__":
    main()
