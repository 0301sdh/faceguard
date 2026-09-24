# Week 2 — Adversarial Attack 기본 원리 (FGSM)

## 목표
Adversarial Attack이 왜 가능한지 이해하고, FGSM(Fast Gradient Sign Method)을 직접 구현해 epsilon에 따른 효과를 확인한다.

## 한 일
- MNIST(손글씨 숫자) 분류용 간단한 CNN을 직접 정의하고 3 epoch 학습 (`experiments/train_mnist.py`)
- 학습된 weight를 `experiments/mnist_cnn.pt`로 저장 (git에서는 제외)
- FGSM 구현 및 epsilon별 정확도 측정 (`experiments/fgsm.py`)
- 같은 크기의 랜덤 노이즈와 비교하여 gradient 방향의 효과 검증 (`use_random=True`)
- 원본 / perturbation / 공격 이미지를 나란히 시각화 (`experiments/visualize_fgsm.py`)

## 핵심 개념 정리

### 1. loss와 학습
- **loss**: 모델이 얼마나 틀렸는지를 나타내는 숫자. 분류에는 `CrossEntropyLoss`를 사용하며, 정답 클래스에 낮은 확률을 줄수록 커진다.
- **학습 루프 4단계**: `zero_grad()` → forward → `loss.backward()`(gradient 계산) → `optimizer.step()`(weight 수정)
- 6만 장을 batch(64장) 단위로 3바퀴(epoch) 돌면서 weight를 고치므로 loss가 0.23 → 0.06 → 0.04로 감소했다.
- **테스트**는 학습에 쓰지 않은 1만 장으로 마지막에 1번만 채점한다. 예측은 `argmax`(점수가 가장 큰 클래스), 정확도는 `맞힌 개수 / 전체`이며 `torch.no_grad()`로 gradient 계산을 끈다. 결과는 약 98.5%.

### 2. gradient의 두 가지 쓰임
| | 대상 | 방향 | 목적 |
|---|---|---|---|
| 학습 | weight | gradient 반대 (loss 감소) | 모델이 잘 맞히게 함 |
| FGSM | 입력 이미지 픽셀 | gradient 방향 (loss 증가) | 모델이 틀리게 함 |

같은 `loss.backward()`를 쓰되, **무엇에 대한 gradient인지**와 **방향**만 반대다.

### 3. FGSM
```text
adv_image = clamp( image + epsilon * sign(∂loss/∂image), 0, 1 )
```
- **perturbation**: 원본에 더해지는 아주 작은 값들의 행렬
- **gradient (`requires_grad_(True)`)**: 이미지에 걸어야 픽셀별 gradient(`image.grad`)를 얻을 수 있음
- **`sign()`**: 크기는 버리고 방향(+1/-1)만 사용 → 모든 픽셀이 정확히 epsilon만큼 변함
- **epsilon**: perturbation의 최대 크기. 클수록 공격은 강하지만 눈에 잘 띔
- **`clamp(0, 1)`**: 픽셀 값이 유효 범위를 벗어나지 않게 함

## 결과

### epsilon별 FGSM 정확도 (테스트 2000장, 모델이 원래 맞힌 이미지 대상)
| epsilon | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|
| 정확도 | 95.0% | 82.7% | 62.0% | 37.6% | 6.6% |

epsilon이 커질수록 정확도가 급격히 떨어진다. 픽셀 값 범위 0~1 중 0.05만 바꿔도 오분류가 생기기 시작한다.

### FGSM vs 랜덤 노이즈 (같은 epsilon, 2000장 중 정답 수)
| epsilon | 0.15 | 0.2 | 0.3 |
|---|---|---|---|
| FGSM | 1217 | 735 | 146 |
| 랜덤 부호 | 1949 | 1949 | 1922 |

랜덤 노이즈는 epsilon=0.3에서도 거의 영향이 없었다(공격 전 약 1960). 랜덤 부호는 픽셀마다 loss를 키우는 방향과 줄이는 방향이 절반씩 섞여 서로 상쇄되지만, FGSM은 모든 픽셀이 loss를 키우는 방향으로 더해진다. **"같은 크기의 변화라도 gradient를 이용하면 효과가 완전히 다르다"**는 것이 이 프로젝트의 핵심 근거다.

### 시각화 (`experiments/fgsm_visualization.png`)
- epsilon 0.05~0.2에서는 사람 눈에 7이고 모델도 7로 맞힘, 0.3에서 배경에 얼룩이 뚜렷해지며 모델이 3으로 오분류
- perturbation 이미지는 모든 픽셀이 +epsilon 또는 -epsilon
- 한 장의 결과만으로 효과를 판단하면 안 되고(이미지마다 견디는 정도가 다름), 그래서 2000장 평균으로 측정했다.

## FaceGuard와의 연결
- 최종 서비스에서 사용자에게 돌려줄 "보호된 이미지"가 곧 공격 이미지(원본 + perturbation)에 해당한다.
- 앞으로 바뀌는 것은 속일 대상(MNIST 분류기 → 얼굴 embedding 모델)과 loss(정답 loss → embedding 거리)뿐이고, 입력 gradient로 이미지를 조정하는 원리는 같다.
- epsilon을 키울수록 보호 성능은 오르지만 이미지 품질이 떨어지는 trade-off가 확인됐다. 

## 한계 / 주의
- MNIST는 흑백 단순 이미지라 얼굴 사진과 다르다. 이 결과가 얼굴 인식 모델과 딥페이크 방어에 그대로 적용된다고 볼 수 없다.
- FGSM은 gradient를 한 번만 사용하는 가장 단순한 공격이다.
- 학습이 랜덤(초기값, shuffle)이라 모델을 다시 학습하면 정확도 수치가 조금씩 달라진다.


