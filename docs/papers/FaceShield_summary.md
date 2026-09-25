# [논문 리뷰] FaceShield: 딥페이크 위협으로부터 얼굴 이미지를 보호하는 선제적 방어

## 1. 논문 정보
- **저자:** Jaehwan Jeong, Sumin In, Sieun Kim, Hannie Shin, Jongheon Jeong, Sang Ho Yoon, Jaewook Chung, Sangpil Kim
- **소속:** Korea University, KAIST, Samsung Research
- **발표 연도:** 2025
- **링크:** http://arxiv.org/pdf/2412.09921

## 2. 해결하려는 문제
- 딥페이크는 접근성이 높아 범죄에 악용될 가능성이 크다.
- 기존 대응은 두 가지다.
  - **탐지(Detection):** 진짜/가짜를 분류하는 사후 대응. 정교한 위조에는 한계가 있다.
  - **선제적 방어:** 얼굴 이미지에 눈에 띄지 않는 adversarial perturbation을 넣어 딥페이크가 이미지를 제대로 처리하지 못하게 한다.
- 기존 선제적 방어는 주로 **GAN 기반** 모델에만 효과적이고, 최신 **확산 모델(Diffusion Model)** 기반 딥페이크에는 잘 통하지 않는다.

## 3. 핵심 아이디어
확산 모델과 GAN 모두에 통하는 선제적 방어를, **세 가지 요소**로 구성한다.

| 구성 요소 | 하는 일 |
|---|---|
| 조건부 얼굴 공격 (Conditioned Face Attack) | 얼굴 이미지가 확산 모델의 조건 입력으로 들어갈 때 정보 전달을 방해 |
| 얼굴 특징 추출기 공격 (Facial Feature Extractor Attack) | MTCNN(얼굴 검출), ArcFace(신원 임베딩)를 공격해 검출/인식을 방해 |
| 향상된 노이즈 업데이트 (Enhanced Noise Update) | 가우시안 블러 + 저주파 필터링으로 눈에 덜 띄고 JPEG에 강한 노이즈 생성 |

### 3-1. 조건부 얼굴 공격
- 확산 모델 기반 딥페이크는 참조 얼굴 이미지를 임베딩으로 바꿔 denoising UNet의 cross-attention에 넣는다(IP-Adapter 사용). 이 구조를 이용한다.
- **얼굴 프로젝터 공격:** 사전학습 모델의 최상위 projection 레이어를 대상으로, 임베딩 단계에서 잘못된 정보가 투영되게 한다(L1 loss).
- **주의 방해 공격(Attention disruption):** UNet **중간 레이어**의 cross-attention 맵 분산(variance)을 이용해 loss를 만든다. 논문 실험에서 전체 레이어나 상/하단 레이어보다 중간 레이어를 노리는 편이 더 효과적이었다.

### 3-2. 얼굴 특징 추출기 공격
- **MTCNN 공격:** 이미지 스케일링/보간(interpolation) 방식이 달라져도 통하도록, 얼굴 검출 확률을 낮추는 loss를 사용한다.
- **ArcFace Identity 공격:** 원본과 보호 이미지의 **코사인 유사도를 낮추는** loss를 사용해 신원 정보를 모호하게 만든다.

```text
L_id(δ; x) = - cos( A(x+δ), A(x) )      # A: ArcFace 임베딩
```

### 3-3. 전체 loss
```text
L_total = λ_proj·L_proj + λ_attn·L_attn + λ_mtcnn·L_mtcnn + λ_id·L_id
```
- 각 λ는 하이퍼파라미터. 최소화/최대화 방향이 서로 달라 λ_proj, λ_id는 음수, λ_attn, λ_mtcnn은 양수로 둔다.
- 하나의 loss를 빼면 성능이 떨어지므로 loss 조합이 더 넓은 범위의 딥페이크를 막는다.

### 3-4. 향상된 노이즈 업데이트
표준 PGD(`δ ← Proj_{‖δ‖∞≤η}(δ − α·sign(∇δ L))`) 위에 두 가지를 더한다.
- **가우시안 블러:** Sobel 연산자로 변화가 큰 영역을 찾아 그 부분의 노이즈에 블러를 적용 → 눈에 덜 띈다.
- **저주파 필터링:** DCT로 주파수 영역에 옮겨 저주파 성분만 남김 → JPEG 압축에서 노이즈가 덜 손실된다.

## 4. 실험 및 결과 (요약본 기준)
- **데이터:** CelebA-HQ, VGGFace2-HQ 각각 200개 ID (ID당 100장 학습, 100장 테스트)
- **지표:** L2, ISM(Identity Score Matching, 낮을수록 원본과 다른 신원), PSNR, LPIPS, SSIM
- **결과:** 기존 확산 공격보다 보호 성능이 좋고 노이즈도 더 적다. JPEG(품질 75) 후에도 성능 유지. GAN 기반 모델에도 효과가 있고 가중치가 다른 유사 모델로의 전이성도 보인다. 사람 평가에서도 노이즈 가시성이 낮게 평가되었다.

## 5. FaceGuard 프로젝트에 참고할 점
- **PGD가 기반이다.** FaceShield의 노이즈 업데이트는 PGD에 블러/필터를 얹은 것이라, Week 3(PGD) 구현이 직접 이어진다.
- **Identity loss가 Week 4~5와 같다.** ArcFace 코사인 유사도 loss는 우리가 Week 1에서 쓴 코사인 유사도, Week 4~5의 embedding 공격과 같은 개념이다.
- **다중 loss 조합**은 Week 5(Protection Loss 설계), Week 8(여러 목표 동시 고려)에서 참고할 수 있다.
- **눈에 안 띄는 노이즈 + JPEG 견고성**은 Week 7(Robustness 실험)과 Week 8(개선)의 아이디어(블러, 저주파 필터)로 시도해볼 수 있다.
- **평가 지표**(L2, PSNR, SSIM, LPIPS, ISM)는 컨텍스트 문서 8번의 정량 지표 계획과 일치한다.

## 6. 논문의 한계 및 우리 프로젝트와의 차이
- **논문이 밝힌 한계:** JPEG/resize에는 강해졌지만, 다른 정제(purification) 기법에서는 보호 노이즈 정보가 손실될 수 있다.
- **범위 차이:** FaceShield의 핵심은 확산 모델(UNet, cross-attention)을 직접 겨냥하는 것이다. 우리 프로젝트는 시간과 자원을 고려해 **얼굴 인식 모델의 embedding 공격**(Fawkes, ArcFace/FaceNet 방식)부터 구현하고, 확산 모델 대상 공격은 범위 밖으로 둔다.
- **주의:** 논문의 성능은 논문 실험 조건에서의 결과이며, "딥페이크를 완벽히 막는다"로 해석하지 않는다(컨텍스트 문서 9번).
- **차별점:** 논문의 기법을 그대로 재현하지 않고, 그중 PGD + identity loss + 견고성 아이디어를 참고해 웹 서비스(Spring Boot + FastAPI)로 연결하는 데 집중한다.
