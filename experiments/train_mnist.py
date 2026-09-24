"""
Week 2 : FGSM 실습에 사용할 간단한 MNIST 분류기 학습

이 스크립트가 하는 일:
1. MNIST(손글씨 숫자) 데이터셋을 불러온다
2. 간단한 CNN을 정의한다
3. 몇 epoch 학습시킨다
4. 학습된 모델 가중치를 저장한다 (다음 단계 FGSM 공격에서 다시 불러와서 사용)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력: (1, 28, 28) 흑백 이미지
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)  # 가로세로 절반으로 줄임

        # conv1 -> pool -> conv2 -> pool 을 거치면 28 -> 14 -> 7 이 됨
        # 최종 feature map 크기: (32, 7, 7)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 숫자 0~9, 총 10개 클래스

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # (batch, 32, 7, 7) -> (batch, 32*7*7) 로 펼치기
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 device:", device)

    # MNIST 이미지를 텐서로 변환 (0~255 픽셀값 -> 0~1 범위)
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}/{epochs}] 평균 loss: {avg_loss:.4f}")

    # 테스트 정확도 확인
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"테스트 정확도: {accuracy * 100:.2f}%")

    save_path = "experiments/mnist_cnn.pt"
    torch.save(model.state_dict(), save_path)
    print(f"모델 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
