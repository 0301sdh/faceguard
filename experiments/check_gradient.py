import torch

# requires_grad=True: 이 텐서에 대한 미분을 추적하겠다는 의미
x = torch.tensor(3.0, requires_grad=True)

# y = x^2 계산 (수학적으로 dy/dx = 2x)
y = x ** 2

# y에서 거꾸로 미분 계산 (backpropagation)
y.backward()

# 계산된 미분값 확인
print("x:", x.item())
print("y:", y.item())
print("x.grad (dy/dx):", x.grad.item())