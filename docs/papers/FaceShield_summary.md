# [논문 리뷰] FaceShield: 딥페이크 위협으로부터 얼굴 이미지를 보호하는 선제적 방어

## 1. 논문 정보
- **저자:** Jaehwan Jeong, Sumin In, Sieun Kim, Hannie Shin, Jongheon Jeong, Sang Ho Yoon, Jaewook Chung, Sangpil Kim
- **소속:** Korea University, KAIST, Samsung Research
- **발표:** ICCV 2025 (공식 코드 저장소 이름 기준, 논문 본문에는 학회명이 명시되지 않음)
- **링크:** http://arxiv.org/pdf/2412.09921
- **공식 코드:** https://github.com/kuai-lab/iccv25_faceshield

## 2. 해결하려는 문제
- 딥페이크는 접근성이 높아 범죄에 악용될 가능성이 크다.
- 기존 대응은 두 가지다.
  - **탐지(Detection):** 진짜/가짜를 분류하는 사후 대응. 정교한 위조에는 한계가 있다.
  - **선제적 방어:** 얼굴 이미지에 눈에 띄지 않는 adversarial perturbation을 넣어 딥페이크가 이미지를 제대로 처리하지 못하게 한다.
- 기존 선제적 방어는 주로 **GAN 기반** 모델에만 효과적이고, 최신 **확산 모델(Diffusion Model)** 기반 딥페이크에는 잘 통하지 않는다.
- 기존 확산 모델 공격(AdvDM, Mist, PhotoGuard, SDST)은 이미지가 **query(편집 대상)**로 들어가는 상황을 가정한다. 하지만 딥페이크에서 source 얼굴은 **조건(key/value)**으로 cross-attention에 들어가므로 이 방법들이 잘 통하지 않는다(Fig. 2, Fig. 5).

## 3. 핵심 아이디어
확산 모델과 GAN 모두에 통하는 선제적 방어를 **세 가지 요소**로 구성한다.

| 구성 요소 | 하는 일 |
|---|---|
| 조건부 얼굴 공격 (Conditioned Face Attack) | 얼굴 이미지가 확산 모델의 조건 입력으로 들어갈 때 정보 전달을 방해 |
| 얼굴 특징 추출기 공격 (Facial Feature Extractor Attack) | MTCNN(얼굴 검출), ArcFace(신원 임베딩)를 공격해 검출/인식을 방해 |
| 향상된 노이즈 업데이트 (Enhanced Noise Update) | 가우시안 블러 + 저주파 필터링으로 눈에 덜 띄고 JPEG에 강한 노이즈 생성 |

### 3-1. 조건부 얼굴 공격
- 확산 모델 기반 딥페이크는 참조 얼굴 이미지를 임베딩으로 바꿔 denoising UNet의 cross-attention에 넣는다(IP-Adapter 사용). 이 구조를 이용한다.
- **얼굴 프로젝터 공격 (Projector loss):** CLIP 이미지 인코더의 최상위 projection 출력 P만 사용한다. 특정 목표값으로 수렴시키지 않고, 원본에서 **멀어지게(divergence)** 한다.

```text
L_proj(δ; x) = ‖ P(x+δ) − P(x) ‖₁      # 키운다
```

- **주의 방해 공격 (Attention disruption loss):** UNet **중간(mid) 레이어** cross-attention 맵의 분산(A_var)을 이용한다.
  - 원본 이미지로 A_var를 구하고, 분산이 낮은(= 조건을 약하게 반영하는) 영역을 quantile로 골라 마스크 M_var를 만든다.
  - 보호 이미지의 A'_var가 그 영역에서 최대 분산 σ_max에 가까워지도록 한다.
  - `L_attn = ‖(σ_max − A'_var) ⊙ M_var‖₂`
  - 전체 레이어나 up/down 레이어보다 mid 레이어를 노릴 때 효과가 가장 컸다(Fig. 6).

### 3-2. 얼굴 특징 추출기 공격
- **MTCNN 공격:** MTCNN의 **첫 단계 P-Net만** 공격한다. NMS 이후 단계는 거치지 않고, **P-Net이 출력하는 얼굴 확률 맵에 바로 MSE loss**를 건다.
  - 검출 확률이 threshold t_prob를 넘는 위치만 마스크(M_prob)로 골라, 확률을 threshold 아래로 끌어내린다.
  - 이미지 피라미드 스케일 중 최종 단계까지 살아남는 스케일만 골라 공격한다(부록 식 14).
  - BILINEAR 축소와, NEAREST 확대 + average pooling(= AREA 방식 흉내)을 **함께** 써서 다양한 resize 방식에 강하게 만든다.
- **ArcFace Identity 공격 (Identity loss):** 원본과 보호 이미지의 ArcFace 코사인 유사도를 낮춘다. 전이성을 높이려고 **ArcFace 두 종류를 동시에 공격(ensemble)**한다(공식 코드 기준 arcface50, arcface100).

```text
L_id(δ; x) = cos( A(x+δ), A(x) ) − 1      # 논문 식 (11), A: ArcFace
```

### 3-3. 전체 loss
```text
L_total = λ_proj·L_proj + λ_attn·L_attn + λ_mtcnn·L_mtcnn + λ_id·L_id
```
- 각 λ는 grid search로 정한 하이퍼파라미터다. 본문에는 "λ_proj, λ_id는 음수, λ_attn, λ_mtcnn은 양수"라고 적혀 있다.
-  **부호 주의:** 알고리즘 1은 `x_adv ← x_adv − α·sign(∇L_total)`, 즉 L_total을 **줄이는** 방향으로 업데이트한다.
  - λ_proj < 0이면 L_proj가 커진다. 의도대로 동작한다.
  - 하지만 λ_id < 0과 식 (11)을 그대로 조합하면 cos가 **커지는** 방향이 된다. 논문 표기가 서로 맞지 않는 것으로 보인다.
  - 따라서 부호를 그대로 옮기지 않는다. **구현 후 반복마다 cos 유사도가 실제로 내려가는지 출력해서 확인한다.**

### 3-4. 향상된 노이즈 업데이트 (알고리즘 1)
표준 PGD(`δ ← Proj_{‖δ‖∞≤η}(δ − α·sign(∇δ L))`)와 달리, **매 스텝의 업데이트량**에 블러와 필터를 적용한다.

```text
반복 n = 1..N:
  L_total 계산
  δ      = α · sign(∇ L_total)          # 이번 스텝 업데이트량
  δ_blur = GaussianBlur(δ)              # 경계 부분만 블러
  δ'     = LowPassFilter(δ_blur)        # 저주파만 남김
  x_adv  = x_adv − δ'
  x_adv  = x + clip(x_adv − x, −ε, ε)   # L∞ projection
마지막: x_adv = clip(x_adv, 0, 255)
```

- **가우시안 블러 (부록 C.1):**
  - 3×3 Sobel로 노이즈의 급격한 변화(경계)를 찾는다.
  - 찾은 영역을 9×9 padding으로 두껍게 만든 마스크 M_sob를 만든다.
  - 그 영역에만 블러를 적용한다: `δ_blur = G(δ)⊙M_sob + δ⊙(1−M_sob)`
- **저주파 필터링 (부록 C.1):**
  - 노이즈를 **8×8 블록으로 나눠 DCT**한다(JPEG과 같은 블록 크기).
  - YCbCr로 바꾸지 않고 **RGB에서 바로** 처리한다. 노이즈는 음수 값을 포함하므로 YCbCr 범위와 맞지 않기 때문이다.
  - JPEG **휘도 양자화 표에서 값이 40 미만인 위치**의 계수만 남기는 마스크를 사용한다.

## 4. 구현 설정 (부록 C.1)

| 항목 | 값 |
|---|---|
| 입력 크기 | 모든 이미지를 **512×512**로 resize |
| PGD | L∞, **ε = 12/255**, step size **1/255**, **30 steps** (비교 대상 방법들과 동일) |
| 사용 모델 | Stable Diffusion v1.5 mid-layer cross-attention, CLIP 이미지 projector 상단, MTCNN P-Net(PyTorch 버전), ArcFace 2종 |
| 처리 비용 | 이미지당 **24초**, 메모리 **15GB** (RTX A6000 48GB에서 실험, loss 4개 모두 사용) |
| 비용을 줄인 방법 | ① 조건부 얼굴 공격에 얼굴 영역만 입력 ② 여러 timestep에 걸친 gradient 누적 없이 조건 경로에서만 gradient 추출 ③ UNet mid 레이어만 사용 |

비교 대상 방법의 비용: AdvDM 20GB/39초, Mist 22GB/80초, PhotoGuard 28GB/234초, SDST 11GB/34초.

## 5. 실험 및 결과

### 5-1. 설정
- **데이터:** CelebA-HQ, VGGFace2-HQ에서 각각 **200명**을 무작위로 골라, **source 100장 + target 100장**으로 source–target 100쌍을 만든다. 학습 데이터가 아니다. 노이즈 생성은 학습이 아니라 이미지별 최적화이기 때문이다.
- **대상 딥페이크:** 확산 기반 DiffFace, DiffSwap, FaceSwap(via Diffusion), IP-Adapter / GAN 기반 SimSwap, InfoSwap
- **지표 (딥페이크 결과물 기준):**
  - L2↑, PSNR↓: 원본 사진으로 만든 딥페이크와 보호 사진으로 만든 딥페이크를 비교한다. 두 결과가 다를수록 보호가 잘 된 것이다.
  - **ISM↓** (Identity Score Matching, Anti-DreamBooth에서 가져온 지표): source 얼굴과 딥페이크 결과의 신원 유사도. 낮을수록 보호가 잘 된 것이다.
  - HE↑: 사람 평가. Likert 1~7점, 20장, 100명.
- **지표 (보호 이미지 품질 기준):** 원본 사진과 보호 사진을 비교하는 LPIPS↓, PSNR↑, SSIM↑, **FR↑**(노이즈 중 저주파 성분 비율)

### 5-2. 주요 수치 (CelebA-HQ)

**보호 이미지 품질 (표 2)**

| 방법 | LPIPS↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|
| AdvDM | 0.4214 | 30.45 | 0.844 |
| SDST | 0.5409 | 31.48 | 0.903 |
| **FaceShield** | **0.2017** | **32.63** | **0.939** |

같은 ε = 12/255인데도 블러와 저주파 필터 덕분에 가장 덜 티가 난다.

**딥페이크 결과 ISM↓ (표 1, 표 4)**

| 딥페이크 | 보호 안 함 | FaceShield | FaceShield + JPEG Q75 |
|---|---|---|---|
| IP-Adapter | – | 0.072 | 0.112 |
| DiffFace | – | 0.243 | 0.259 |
| **SimSwap** | **0.544** | **0.184** | – |
| InfoSwap | 0.431 | 0.237 | – |

VGGFace2-HQ에서 SimSwap: 0.681 → 0.314.

### 5-3. Ablation: loss 하나씩 뺐을 때 ISM↓ (표 3)

| | DiffFace | DiffSwap | FaceSwap | IP-Adapter | SimSwap | InfoSwap |
|---|---|---|---|---|---|---|
| w/o L_proj | 0.241 | 0.167 | 0.270 | 0.135 | **0.544** | 0.256 |
| w/o L_attn | 0.254 | 0.170 | 0.223 | 0.076 | 0.168 | 0.252 |
| w/o L_mtcnn | 0.231 | 0.174 | 0.166 | 0.047 | 0.183 | **0.354** |
| w/o L_id | **0.446** | 0.175 | 0.217 | 0.040 | **0.512** | **0.430** |
| L_total | 0.243 | 0.163 | 0.194 | 0.072 | 0.184 | 0.237 |

읽는 법: 어떤 loss를 뺐을 때 값이 크게 **올라가면**, 그 loss가 해당 딥페이크를 막는 데 중요하다는 뜻이다.

- **SimSwap:** L_id를 빼도(0.512), **L_proj를 빼도(0.544 = 보호 안 함과 같음)** 보호가 사라진다. 이 설정에서는 두 loss가 **모두 있어야** SimSwap이 막혔다.
- **DiffFace, InfoSwap:** L_id가 핵심이다. DiffFace는 0.243 → 0.446으로 크게 나빠진다. 확산 기반 face swap도 ArcFace로 신원을 추출하기 때문이다.
- **IP-Adapter:** L_proj가 핵심이다(빼면 0.072 → 0.135). 반면 L_id나 L_mtcnn을 빼면 오히려 **더 좋아진다**(0.040, 0.047). loss끼리 서로 방해할 수 있다는 뜻이다.
- **L_mtcnn:** InfoSwap에만 크게 기여한다. SimSwap에는 거의 영향이 없다(0.184 → 0.183).
- 논문에는 loss를 **하나만** 쓴 실험이 없다. leave-one-out 실험만 있다.

### 5-4. Robustness (부록 E, IP-Adapter에서만 평가)
- JPEG Q90/75/50, 비트 축소 8-bit/3-bit, resize 75%/50% 후 원래 크기로 복원(BILINEAR, INTER_AREA)
- 모든 조건에서 성능이 조금 떨어지지만, 비교 대상 방법보다는 여전히 좋다. 저주파 필터가 적용된 결과다.

### 5-5. IP-Adapter 버전별 차이 (부록 G)
- 기본 IP-Adapter, Plus 계열: **CLIP 이미지 인코더**로 참조 얼굴을 임베딩한다 → Projector loss의 대상
- **FaceID 계열: CLIP 대신 InsightFace(ArcFace) 임베딩**을 쓴다 → Identity loss가 통할 가능성이 있다
- FaceIDPlus: InsightFace + CLIP을 둘 다 쓴다
- 논문은 여러 버전(SD1.5 / SDXL)으로 전이성을 정성적으로 보였다.

## 6. FaceGuard 프로젝트에 참고할 점

### Week 5 (ArcFace Identity loss + SimSwap pilot)
- **baseline 목표치:** 논문의 SimSwap 원본 ISM은 0.544, 보호 후 0.184다(CelebA-HQ). 내 실험의 baseline과 비교할 기준으로 쓴다.
- ⚠️ **위험 요소:** 표 3에서 L_proj만 빼도 SimSwap 보호가 사라졌다. 즉 **Identity loss 하나만으로 SimSwap이 막힌다는 근거는 논문에 없다.** pilot 결과가 약하면 이것이 원인 후보 중 하나다. 이 경우 Projector loss를 앞당겨 붙여보는 것도 선택지로 둔다.
- ArcFace 2종 ensemble을 쓴다. 특히 SimSwap 내부의 ArcFace와 같은 가중치가 포함되어 있는지 확인한다.
- 부호 문제(3-3)가 있으니 반복마다 cos 유사도를 출력해서 확인한다.

### Week 6 (Robustness)
- 논문과 같은 조건(JPEG Q90/75/50, resize 75%/50% × BILINEAR/INTER_AREA)을 쓰면 논문 수치와 바로 비교할 수 있다.
- 이 시점에는 저주파 필터가 없으므로 논문보다 약하게 나오는 것이 자연스럽다. 이 차이가 Week 8 개선의 근거가 된다.
- 평가는 내 사진 한 장이 아니라 CelebA-HQ 등 **여러 명의 얼굴**(예: 20~50장)로 한다.

### Week 7 (IP-Adapter + Projector loss)
- **노이즈 생성에는 CLIP 이미지 인코더만** 필요하다. Stable Diffusion은 결과를 평가할 때만 필요하다. 그래서 노이즈 생성 자체는 8GB GPU에서도 가능할 수 있다.
- 어떤 IP-Adapter 버전을 쓰는지 명시한다. 기본 버전은 Projector loss, FaceID 버전은 Identity loss와 연결된다.
- "Identity만 / Projector만 / 둘 다" 비교는 논문에 없는 실험이다. **내 기여로 기록할 수 있다.** 표 3처럼 loss끼리 서로 방해할 수 있다는 점도 함께 확인한다.

### Week 8 (개선)
- 블러와 저주파 필터는 **매 스텝 업데이트량**에 적용한다(3-4). 설정: Sobel 3×3, 마스크 9×9, 8×8 DCT, 양자화 표 값 40 미만.
- Attention loss는 **약 15GB**가 필요하다. RTX 2080(8GB)으로는 어렵고, Colab T4(16GB)에서는 빠듯하다.
- (선택) DiffFace: 표 3에서 L_id의 기여가 크므로, Identity loss만으로 확산 기반 face swap까지 막을 수 있는지 확인할 후보다.

### Week 9~11 (서비스)
- **해상도:** 논문은 512×512로 줄여서 처리한다. 서비스에서는 얼굴 영역만 잘라 보호한 뒤 원본 해상도 이미지에 다시 붙이는 방식을 검토한다. 논문도 얼굴 영역만 입력해서 비용을 줄였다.
- **출력 형식:** 결과는 **PNG(무손실)**로 제공한다. SNS 업로드처럼 다시 압축되면 효과가 줄어든다는 안내를 넣는다.
- **처리 시간:** 논문 기준으로도 이미지당 24초가 걸린다. 동기 응답보다 비동기 작업(QUEUED → PROCESSING → COMPLETED) 구조가 필요하다.

### 평가 지표
- 딥페이크 결과 기준: ISM(ArcFace 기준 + **FaceNet 기준**으로 전이성 확인), 원본 결과와 보호 결과 사이의 PSNR/L2
- 보호 이미지 기준: PSNR, SSIM, LPIPS, (선택) FR

## 7. 논문의 한계 및 우리 프로젝트와의 차이
- **논문이 밝힌 한계:** JPEG, resize에는 강해졌지만, 다른 정제(purification) 기법에서는 보호 노이즈 정보가 손실될 수 있다.
- **범위 차이:** FaceShield는 loss 4개를 합쳐 노이즈 하나로 모든 딥페이크를 동시에 평가하고, 확산 모델 쪽에 무게를 둔다. 우리는 GPU와 경험의 제약 때문에 **face swap(ArcFace Identity loss) → IP-Adapter(Projector loss)** 순서로 하나씩 붙여가며 확인한다. Attention loss와 MTCNN loss는 제외한다.
  - MTCNN loss를 제외하는 이유: 표 3에서 SimSwap에 대한 기여가 거의 없다. InfoSwap처럼 MTCNN을 쓰는 모델까지 다룬다면 다시 검토한다.
- **주의:** 논문의 성능은 논문 실험 조건(512×512, CelebA-HQ/VGGFace2-HQ, 무손실 저장)에서의 결과다. "딥페이크를 완벽히 막는다"로 해석하지 않는다(컨텍스트 문서 9번).
- **차별점:** 논문의 기법을 그대로 재현하지 않는다. PGD + Identity loss + Projector loss + 견고성 아이디어를 참고하여, loss 조합별 비교 실험과 웹 서비스(Spring Boot + FastAPI) 연결에 집중한다.

## 8. 이전 요약본에서 바로잡은 것
- 데이터: "ID당 100장 학습, 100장 테스트"가 아니라 **source 100장 + target 100장**이다. 학습 과정은 없다.
- Identity loss 식: 논문 식은 `cos(·) − 1`이다. 부호 표기가 서로 맞지 않는 문제는 3-3에 정리했다.
- MTCNN 공격: P-Net 확률 맵에 바로 loss를 건다. NMS를 거치지 않는다.
- Attention loss 자원: 48GB가 필요한 것이 아니다. 실험 장비가 48GB였고 실제 사용량은 약 15GB다.
- 확산 모델 공격(IP-Adapter, Projector loss)은 이제 범위 밖이 아니라 Week 7 계획에 포함된다.
