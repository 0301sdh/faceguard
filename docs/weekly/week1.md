# Week 1 — AI/CV 기초와 얼굴 임베딩 실험

## 목표
AI 프로젝트를 시작하기 위한 최소한의 개념(Tensor, Gradient, Face Embedding)을 이해하고, 첫 번째 얼굴 임베딩 실험까지 성공한다.

## 한 일
- GitHub 저장소(`faceguard`) 생성 및 프로젝트 구조 설정
- Python 가상환경(`.venv`) 구축, PyTorch 설치 및 MPS(Mac GPU) 사용 가능 여부 확인
- PyTorch autograd(자동 미분) 동작 확인 — `y = x^2`의 gradient가 `2x`로 정확히 계산되는지 실험
- Pillow로 이미지 로드 및 크기/채널/dtype 확인
- `facenet-pytorch`(MTCNN + InceptionResnetV1)로 얼굴 검출 및 512차원 임베딩 추출 파이프라인 구현
- 코사인 유사도로 "본인 vs 본인", "본인 vs 타인" 비교

## 결과
| 비교 대상 | 코사인 유사도 |
|---|---|
| 본인 vs 본인 (다른 사진) | 0.743 |
| 본인 vs 타인 | 0.372 |
| 강아지 사진 | 얼굴 검출 실패 (정상 동작) |

같은 사람 쌍의 유사도가 다른 사람 쌍보다 약 2배 높게 나와, FaceNet 임베딩이 신원을 명확히 구분한다는 것을 확인했다. 이 값(0.743)이 앞으로 adversarial perturbation을 적용해 낮춰야 할 baseline이 된다.


