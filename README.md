# FaceGuard

얼굴 사진에 사람 눈에는 거의 보이지 않는 노이즈(adversarial perturbation)를 넣어, 딥페이크가 그 사진으로 내 얼굴의 신원을 복제하기 어렵게 만드는 프로젝트입니다. 최종 목표는 사진을 올리면 보호된 사진을 돌려주는 웹 서비스입니다.


## 막으려는 것

사진 한 장만으로 내 얼굴을 다른 사람의 몸이나 다른 장면에 넣는 딥페이크입니다. 다음 순서로 다룹니다.

1. Face swap (SimSwap): 얼굴 안쪽만 바꾸는 방식. 신원을 ArcFace로 읽기 때문에 공격이 통하는지 먼저 확인하는 실험 대상으로 씁니다. 
2. 신원 유지 이미지 생성 (IP-Adapter FaceID, InstantID): 내 사진을 참고해 몸, 옷, 배경까지 새로 그리는 방식. 실제로 막고 싶은 위협에 더 가깝습니다. 

## 방법

딥페이크는 사진을 그대로 쓰지 않고, 얼굴 인식 모델이 만든 신원 임베딩(512차원 벡터)을 보고 얼굴을 그립니다. 그래서 그 임베딩이 원래 내 임베딩과 멀어지도록 사진을 아주 조금 바꿉니다.

- Identity loss: 보호 사진과 원본 사진의 임베딩 코사인 유사도. 이 값을 낮추는 방향으로 PGD를 반복합니다.
- 노이즈 크기 제한: 픽셀마다 eps(예: 5/255) 이상 바뀌지 않게 합니다.
- EOT: 딥페이크 도구가 얼굴을 다시 잘라 정렬해도 노이즈가 통하도록, 공격 중에 사진을 조금씩 회전, 확대, 이동한 여러 버전의 평균 기울기를 씁니다.
- 평가: 노이즈를 만들 때 쓴 모델(ArcFace)과 쓰지 않은 모델(FaceNet) 둘 다로 딥페이크 결과가 나를 닮았는지 측정합니다.

핵심 참고 연구는 FaceShield(arXiv 2412.09921)이며, 그중 Identity loss와 Projector loss를 현실적인 범위에서 구현합니다.

## 현재 결과 (Week 6, SimSwap)

| 항목 | 결과 |
|---|---|
| 노이즈 설정 | eps 5/255, 회전·크기 EOT (변형 4개), PGD 30 steps |
| SimSwap 결과와 내 사진의 유사도 (보호 전 → 보호 후) | ArcFace 0.72 → -0.13, FaceNet 0.63 → 0.29 |
| 판정선 (이보다 낮으면 다른 사람 수준) | ArcFace 0.2, FaceNet 0.365 |
| 변형 견고성 | JPEG 90/75/50, 크기 조절, 자르기, 블러 등 11가지에서 효과의 89% 이상 유지 |
| 화질 | PSNR 약 36.8dB |

[변형별 효과 유지율 그래프](docs/weekly/images/week6_3_robustness.png)

- 사진 4장, seed 3개 평균입니다. 4장 중 1장은 FaceNet 기준으로 부분적으로만 보호되었습니다.
- 얼굴만 잘라낸 224×224 사진으로 실험했습니다. 전체 사진에서의 효과는 아직 확인하지 않았습니다.

## 진행 상황

| 주차 | 내용 | 결과 | 문서 |
|---|---|---|---|
| 1 | AI/CV 기초, 얼굴 임베딩 | FaceNet으로 본인 0.743, 타인 0.372 구분 | [week1](docs/weekly/week1.md) |
| 2 | FGSM | 같은 크기라도 기울기 방향 노이즈만 정확도를 떨어뜨림 (MNIST) | [week2](docs/weekly/week2.md) |
| 3 | PGD | eps 0.15에서 정확도 FGSM 62.0%, PGD 19.2% | [week3](docs/weekly/week3.md) |
| 4 | 얼굴 임베딩 공격 (Identity loss) | FaceNet에서 eps 3/255로 보호, 다른 모델에는 전이가 약함 | [week4](docs/weekly/week4.md) |
| 5 | SimSwap 환경, ArcFace 공격 pilot | 사진 4장 모두 보호 확인 → 계속 진행 | [week5](docs/weekly/week5.md) |
| 6 | 재정렬·변형 견고성, eps와 화질 | eps 5/255, 변형 후에도 89% 이상 유지 | [week6](docs/weekly/week6.md) |

## 저장소 구조

```text
docs/
  weekly/        주차별 문서 (목표, 결과, 해석, 깨달은 점)
    images/      결과 그래프
  papers/        참고 논문 정리 (FaceShield, Fawkes)
experiments/     로컬에서 돌리는 코드 (Week 1~4 실험, FaceNet 평가, 그래프)
notebooks/       Colab 노트북 (SimSwap 실행과 보호 실험, GPU 필요)
results/         실험 결과 숫자 원본 (사진별 전체 수치)
```

얼굴 사진과 딥페이크 결과물은 `data/`에 두며 GitHub에 올리지 않습니다. 저장소에는 숫자와 그래프만 있습니다.

## 실행 환경

- 로컬 (Week 1~4 실험, FaceNet 평가): macOS, Python 가상환경(`.venv`), PyTorch, facenet-pytorch
- Colab (Week 5~6): T4 GPU, SimSwap, insightface. 노트북의 첫 셀부터 순서대로 실행하고, 사진은 Google Drive의 `faceguard/photos`에 둡니다.

```bash
.venv/bin/python experiments/week6_facenet_eval.py
```

## 한계

- 딥페이크를 완벽하게 막는 것이 목표가 아닙니다. 실험한 방식(현재 SimSwap)에 대해서만 효과를 말합니다.
- 내부 구조가 공개되지 않은 이미지 편집 AI에는 효과를 보장하지 않습니다.
- 사진 여러 장으로 모델을 학습시키는 방식(DreamBooth, LoRA)은 범위 밖입니다.
- 실험은 본인 사진 4장으로 진행했습니다.

## 참고

- FaceShield: Defending Facial Image against Deepfake Threats (arXiv 2412.09921) - [정리](docs/papers/FaceShield_summary.md)
- Fawkes: Protecting Privacy against Unauthorized Deep Learning Models - [정리](docs/papers/Fawkes_summary.md)
- SimSwap: https://github.com/neuralchen/SimSwap
- facenet-pytorch: https://github.com/timesler/facenet-pytorch
