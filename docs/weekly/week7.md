# Week 7 — 신원 유지 이미지 생성(IP-Adapter FaceID) 확장

> Week 7 진행 중. 지금까지의 결론: 보호 안 한 사진 한 장으로 FaceID가 나를 확실히 복제한다(FaceNet 0.82, 눈으로도 닮음). Week 6 노이즈는 FaceID에 일부만 전이됐다(ArcFace 0.73 → 0.48, FaceNet 0.82 → 0.69). SimSwap ArcFace와 FaceID의 ArcFace를 함께 공격(ensemble)하자 ArcFace 0.083(성공), 눈으로도 다른 사람이 됐지만, 공격에 쓰지 않은 FaceNet은 0.510(부분)이다. SimSwap 보호는 유지(4장 성공). 다음: 노이즈 seed 반복, FaceNet 쪽 원인 확인.

## 요약

| 항목 | 결과 |
|---|---|
| 평가 대상 | IP-Adapter FaceID (SD 1.5, Realistic Vision V4.0) — 내 사진 한 장으로 다른 장면 속의 나를 새로 그림 |
| Baseline (보호 안 함) | ArcFace **0.728**, FaceNet **0.820**, 눈으로 4장 모두 닮음 → 위협이 실제로 작동함 |
| Pilot (Week 6 보호 사진, eps 5/255) | ArcFace 0.728 → **0.476** (부분), FaceNet 0.820 → **0.685** (사실상 실패) |
| Pilot 판단 | **조건부 No-go.** 전이는 되지만 약하다 → FaceID가 쓰는 ArcFace를 직접 공격해야 함 |
| Ensemble (SimSwap ArcFace + FaceID ArcFace, eps 5/255, 노이즈 seed 0) | ArcFace 0.728 → **0.083** (성공), FaceNet 0.820 → **0.510** (부분), 눈으로 4장 모두 다른 사람. SimSwap은 ArcFace -0.002로 4장 성공 유지 |
| Ensemble 판단 | **조건부 Go.** 공격한 모델과 눈으로는 막힘. 공격하지 않은 FaceNet에서는 부분 보호 |

---

## 목표
실제로 막고 싶은 위협("내 얼굴이 다른 장면·몸에 들어간 이미지")을 만드는 신원 유지 이미지 생성으로 평가 대상을 넓힌다. Week 5~6의 Identity loss 노이즈가 이 방식에도 통하는지 확인한다.

## 한 일
- [x] 개념 공부: Diffusion, IP-Adapter FaceID 구조, 우리 노이즈가 통할 수 있는 이유
- [x] Colab 환경 구성 (`notebooks/ipadapter_faceid_eval.ipynb`), 트러블슈팅 2건
- [x] 눈금 잡기: insightface ArcFace의 본인·타인 기준선
- [x] Baseline: 보호 안 한 얼굴 4장 × seed 3개
- [x] Pilot: Week 6 보호 사진(eps 5/255, seed0) 4장 × seed 3개
- [x] Ensemble 준비: insightface ArcFace onnx → PyTorch 변환, gradient 확인, FaceID 정렬 따라 하기
- [x] Ensemble: SimSwap ArcFace + insightface ArcFace 함께 공격 (노이즈 seed 0)
- [ ] Ensemble 노이즈 seed 1·2 반복, SimSwap 결과 FaceNet 채점
- [ ] (선택) Projector loss: CLIP을 쓰는 FaceID Plus
- [ ] loss 조합별 비교 표

## 핵심 개념

### 1. Diffusion
- 랜덤 노이즈에서 시작해, UNet이 "섞인 노이즈"를 예측하고 빼는 일을 30번쯤 반복해서 그림을 **새로** 그린다.
- 무엇을 그릴지는 힌트(조건)로 정한다. 프롬프트는 CLIP 텍스트 인코더가 숫자로 바꿔 UNet의 cross-attention 통로로 넣는다.
- SD 1.5는 512×512 이미지를 64×64 압축 공간(latent)에서 다뤄서 Colab T4에서도 돌아간다.

### 2. IP-Adapter FaceID
- 사진을 숫자로 바꿔 힌트 통로를 하나 더 붙인다. 원래 UNet은 얼려 두고 새 통로만 학습했다.
- FaceID는 이 자리에 **insightface ArcFace**(buffalo_l)를 쓴다: 얼굴 검출 → 5개 점 정렬 → 112×112 → 512개 숫자(얼굴 명세서) → 투영 → cross-attention
- (FaceShield가 공격한 IP-Adapter 기본형은 CLIP 이미지 인코더를 쓴다 → Projector loss)

### 3. SimSwap과의 차이
| | SimSwap | IP-Adapter FaceID |
|---|---|---|
| 내 사진에서 가져가는 것 | ArcFace 512개 숫자 | ArcFace 512개 숫자 (같음) |
| 머리·몸·배경 | 실제 target 사진에서 | 프롬프트를 보고 새로 그림 |
| 얼굴 | target 얼굴 위에 내 특징을 입힘 | 전부 새로 그림 |

### 4. classifier-free guidance
- 얼굴 정보가 "있을 때"와 "없을 때(0)" 두 번 예측해서 그 차이를 키운다. 그래서 명세서 앞에 0으로 채운 명세서를 붙여 넣는다.

## 환경 구성
- Colab T4, diffusers + insightface(최신) + onnxruntime(CPU)
- 모델: `SG161222/Realistic_Vision_V4.0_noVAE` + `stabilityai/sd-vae-ft-mse`, DDIM, `h94/IP-Adapter-FaceID`의 `ip-adapter-faceid_sd15.bin`, scale 1.0 (공격자가 가장 닮게 만드는 상황), LoRA는 쓰지 않음
- 생성 설정: 512×512, 30 steps, 프롬프트 `upper body photo of a person wearing a suit in an office`, seed 0/1/2
- 생성 시간: 이미지당 약 5초 (30 steps, 약 6 it/s). 처음 1장은 약 8초
- Week 5~6 노트북은 SimSwap 때문에 옛 insightface(0.2.1)를 써서 노트북을 따로 만들었다

### 트러블슈팅 1: `cannot import name 'FqnToConfig' from 'torchao.quantization'`
- 원인: 최신 diffusers는 torchao(양자화 도구)가 깔려 있으면 불러오는데, Colab에 있는 옛 torchao에는 그 기능이 없다. 두 패키지의 버전이 맞지 않는 문제
- 해결: `pip uninstall torchao`. 양자화를 쓰지 않으므로 잃는 것이 없고, 없으면 diffusers가 불러오지 않는다

### 트러블슈팅 2: `CUDAExecutionProvider is not in available provider names`
- 원인: Colab에 CPU용 onnxruntime이 이미 있어서, onnxruntime-gpu를 추가로 깔아도 CPU용이 우선된다
- 해결: CPU로 실행. 얼굴 검출과 명세서 추출은 사진 몇 장이라 CPU로 충분하고, 무거운 생성은 PyTorch가 GPU로 한다

## 결과

### 1. 눈금 잡기 (insightface ArcFace)
| | 유사도 |
|---|---|
| 본인끼리 (6쌍) | 0.510 ~ 0.736 (최소 0.510) |
| 타인과 (SimSwap target, 4쌍) | -0.138 ~ -0.023 (평균 -0.066) |
| 판정 기준 | 실패 ≥ 0.510 / 부분 0.222 ~ 0.510 / 성공 < 0.222 |

- Week 5와 같은 방식(본인 최소 = 실패선, 본인 최소와 타인의 중간 = 성공선)
- 타인은 FaceNet 채점과 같은 SimSwap target이다. FaceNet 판정선은 Week 6과 같다 (실패 ≥ 0.686 / 성공 < 0.365)

### 2. Baseline: 보호 안 한 얼굴로 생성
- 입력: Week 6의 보호 안 한 224×224 정렬 얼굴 (`week6/robust_eps0/person_a_k_eps0.png`). 보호 사진과 같은 형식이어야 차이를 노이즈 때문이라고 말할 수 있다
- 채점: 생성 결과 vs 내 원본 사진, seed 3개 평균

| 사진 | ArcFace | FaceNet | 눈으로 |
|---|---|---|---|
| a_1 | 0.750 | 0.824±0.006 | 닮음 |
| a_2 | 0.686 | 0.800±0.004 | 닮음 |
| a_3 | 0.731 | 0.847±0.008 | 닮음 |
| a_4 | 0.745 | 0.810±0.019 | 닮음 |
| **평균** | **0.728 (실패)** | **0.820 (실패)** | **4장 모두** |

### Baseline 해석
- ArcFace 평균 0.728은 내 진짜 사진끼리(0.510 ~ 0.736)와 같은 수준이다.
- 도구와 관계없는 FaceNet에서 오히려 더 높았다(0.820). 내 진짜 사진끼리 최소(0.686)보다 높다. "도구 안의 모델로 채점해서 유리하게 나왔다"는 걱정은 해당하지 않는다.
- seed에 따른 흔들림이 작다(FaceNet 표준편차 0.004 ~ 0.019). 작은 차이도 노이즈 효과로 읽을 수 있다.
- SimSwap과 비교하면 더 강한 위협이다.

| 보호 안 함 | SimSwap (Week 5~6) | FaceID (Week 7) |
|---|---|---|
| ArcFace | 0.72 | 0.728 |
| FaceNet | 0.63 | **0.820** |
| 눈으로 | 안 닮음 | **닮음** |

### 3. Pilot: Week 6 보호 사진으로 생성
- 입력: Week 6 최종 설정(eps 5/255, EOT 변형 4개, steps 30, 노이즈 seed0)의 보호 얼굴 (`week6/eps5_n4_seed0/person_a_k_eps5.png`)
- 프롬프트와 생성 seed는 baseline과 같다 → 입력의 노이즈만 다른 비교

| 사진 | ArcFace baseline → pilot | FaceNet baseline → pilot |
|---|---|---|
| a_1 | 0.750 → 0.398 (부분) | 0.824 → 0.696±0.025 (실패) |
| a_2 | 0.686 → 0.430 (부분) | 0.800 → 0.631±0.002 (부분) |
| a_3 | 0.731 → 0.523 (실패) | 0.847 → 0.700±0.022 (실패) |
| a_4 | 0.745 → 0.553 (실패) | 0.810 → 0.712±0.017 (실패) |
| **평균** | **0.728 → 0.476 (부분)** | **0.820 → 0.685 (부분, 실패선 바로 아래)** |

사진별 전체 숫자: `results/week7/arcface_colab.txt`, `results/week7/facenet_ensemble.txt`

### Pilot 해석
- **전이는 된다.** 노이즈를 만들 때 쓰지 않은 insightface ArcFace에서도 seed 3개 모두 같은 방향으로 떨어졌다.
- **하지만 약하다.** 같은 노이즈가 직접 공격한 모델에서는 훨씬 크게 통했다.

| 같은 노이즈 (eps 5) | SimSwap | FaceID |
|---|---|---|
| 공격한 모델이 들어 있나 | O (SimSwap ArcFace) | X (insightface ArcFace) |
| ArcFace | 0.72 → -0.13 (성공) | 0.728 → 0.476 (부분) |
| FaceNet | 0.63 → 0.29 (성공) | 0.820 → 0.685 (사실상 실패) |

- 이름이 같은 ArcFace라도 학습 데이터와 가중치가 다른 별개의 모델이라, Week 4와 같은 전이 문제가 생겼다. 계획 단계에서 위험 요소로 적어 둔 것(컨텍스트 문서 16-3)이 실제로 일어났다.
- a_4는 이번에도 가장 약하다. Week 6에서도 FaceNet 기준 부분 보호였던 사진이다.

### Go / No-go: 조건부 No-go
- Week 6 노이즈만으로는 FaceID를 막을 수 없다. FaceNet이 거의 그대로다.
- 노이즈 seed1·2 반복은 하지 않았다. 방향이 분명해서, ensemble 버전을 만든 뒤 최종 비교 때 반복한다.
- 다음: SimSwap ArcFace와 insightface ArcFace를 함께 공격한다 (FaceShield가 ArcFace 여러 개를 함께 공격한 이유와 같다).

```text
L_id = cos(SimSwap ArcFace(x+δ), 원본) + cos(insightface ArcFace(x+δ), 원본)  → 최소화
```

- 풀어야 할 문제: insightface ArcFace는 `.onnx` 파일이라 기울기를 계산할 수 없다. PyTorch 모델로 변환해야 한다.

### 4. Ensemble 준비: FaceID의 ArcFace를 공격할 수 있게 만들기
`notebooks/simswap_robustness.ipynb` (Week 6 노이즈를 만든 노트북) 맨 아래

| 확인 | 결과 | 판단 |
|---|---|---|
| ① onnx → PyTorch 변환 (onnx2torch, `w600k_r50.onnx`) | onnx vs PyTorch 유사도 4장 모두 **1.00000** | 같은 계산을 함 |
| ② gradient | 평균 크기 4.46e-05, **0인 비율 0.0%** | 공격 가능 |
| ③ FaceID 정렬 따라 하기 (grid_sample) | 실제 방식(cv2 정렬 + onnx) 대비 **0.9998 ~ 0.9999** | FaceID가 보는 얼굴과 같음 |
| 참고: 단순 112 축소 | 실제 방식 대비 0.936 ~ 0.965 | 쓰면 과녁이 어긋남 |

- gradient 크기가 작아도 PGD는 방향(sign)만 쓰므로 0이 아니면 충분하다.
- ③이 필요했던 이유: FaceID는 224 얼굴을 그냥 줄이지 않고, 다시 얼굴을 찾아 눈·코·입꼬리 5점을 **insightface 기준 위치**에 맞춰 자른다. SimSwap의 224 crop은 기준 위치가 달라서(눈 사이 간격, 입 위치) 그냥 줄이면 원본 사진인데도 0.04 ~ 0.06만큼 다른 얼굴로 읽힌다.
  - 해결: 보호 전 얼굴에서 FaceID 방식으로 변환(M)을 한 번 구하고, 같은 변환을 미분 가능한 `grid_sample`로 따라 한다. Mac에서 `cv2.warpAffine`과 픽셀 차이가 최대 0.5/255임을 먼저 확인했다.

### 5. Ensemble 공격
- 설정: Week 6 최종 설정 그대로 (eps 5/255, EOT 변형 4개, steps 30, 노이즈 seed 0). **loss만** 바꿈

```text
loss = cos(SimSwap ArcFace(x+δ), 원본) + cos(FaceID ArcFace(정렬(x+δ)), 원본)  → 최소화 (EOT 변형 4개 평균)
```

| 사진 | SimSwap 결과 (SimSwap ArcFace) Week 6 → ensemble | 입력 단계 FaceID ArcFace Week 6 → ensemble |
|---|---|---|
| a_1 | -0.224 → -0.063 | 0.678 → 0.020 |
| a_2 | -0.193 → -0.079 | 0.530 → -0.119 |
| a_3 | -0.004 → 0.080 | 0.697 → 0.146 |
| a_4 | -0.093 → 0.054 | 0.662 → 0.012 |
| **평균** | **-0.129 → -0.002 (4장 성공)** | **0.642 → 0.015** |

- Week 6 SimSwap 값은 노이즈 seed 3개 평균, ensemble은 seed 0 하나다.
- 입력 단계: 생성 없이 보호 얼굴과 보호 안 한 얼굴의 FaceID 명세서를 비교한 빠른 사전 확인.

### 6. Ensemble 보호 사진으로 FaceID 생성
- 입력: `week6/ens_eps5_n4_seed0/person_a_k_eps5.png`. 프롬프트와 생성 seed는 baseline·pilot과 같다

| 사진 | ArcFace baseline → pilot → **ensemble** | FaceNet baseline → pilot → **ensemble** | 눈으로 |
|---|---|---|---|
| a_1 | 0.750 → 0.398 → **0.102** | 0.824 → 0.696 → **0.651±0.056** (부분) | 다른 사람 |
| a_2 | 0.686 → 0.430 → **0.003** | 0.800 → 0.631 → **0.331±0.042** (성공) | 다른 사람 |
| a_3 | 0.731 → 0.523 → **0.188** | 0.847 → 0.700 → **0.568±0.034** (부분) | 다른 사람 |
| a_4 | 0.745 → 0.553 → **0.038** | 0.810 → 0.712 → **0.491±0.024** (부분) | 다른 사람 |
| **평균** | **0.728 → 0.476 → 0.083 (성공)** | **0.820 → 0.685 → 0.510 (부분)** | **4장 모두** |

- ArcFace: 생성 seed 12개 중 성공선(0.222)을 넘은 것은 a_3 seed1(0.229) 하나

### Ensemble 해석
- **공격한 모델에서는 확실히 막혔다.** FaceID의 ArcFace 기준 0.728 → 0.083으로, 타인 수준(-0.066)에 가깝다. 눈으로도 4장 모두 다른 사람이다. Pilot에서 실패하던 a_3, a_4도 성공이 됐다.
- **입력 0.015 → 생성 0.083.** 생성 모델이 흐트러진 명세서를 그럴듯한 얼굴로 그리면서 조금 되돌아가지만, 타인 수준에 머문다.
- **SimSwap 보호는 유지됐지만 여유가 줄었다** (-0.129 → -0.002). 같은 eps 5 안에서 두 모델을 나눠 상대하기 때문이다.
- **공격하지 않은 FaceNet에서는 부분 보호다** (0.820 → 0.510). Pilot보다 크게 떨어졌지만 성공선(0.365) 아래는 a_2뿐이다.
  - 같은 eps 5 노이즈를 SimSwap에 썼을 때는 FaceNet 0.297(성공)이었다. FaceID는 얼굴을 통째로 새로 그리므로 FaceNet이 신원과 함께 보는 특징(성별, 나이, 인상 등)이 유지되면 점수가 남을 수 있다 (가설, 아직 확인 안 함).
  - FaceNet 성공선은 타인 1명(target_specific1, 0.044)으로 잡았다. 비슷한 인상의 타인이라면 0.044보다 높을 수 있어서, 성공선이 엄격하게 잡혔을 가능성도 있다 (확인 안 함).

### Go / No-go: 조건부 Go
- 막고 싶은 도구(FaceID)가 쓰는 모델과 사람 눈 기준으로는 막혔다. Week 6 노이즈만으로는 못 막던 것이다.
- 남은 확인
  - 노이즈 seed 1·2 반복 (지금은 seed 0 하나)
  - SimSwap 결과의 FaceNet 채점 (보호가 약해졌는지 독립 모델로 확인)
  - FaceNet 0.510이 남는 원인: 타인 여러 명으로 FaceNet 눈금 다시 잡기, 또는 FaceNet을 ensemble에 추가하는 실험

## 한계
- 프롬프트 1개, 상반신 장면만 시험했다. 전신처럼 얼굴이 작은 장면은 결과가 다를 수 있다.
- FaceID의 LoRA는 쓰지 않았다. LoRA를 쓰면 더 닮게 그려질 수 있다.
- 사진 4장, 본인 1명의 결과다.
- Ensemble은 노이즈 seed 1개, 생성 seed 3개의 결과다.
- Ensemble의 FaceID 정렬 변환(M)은 보호 전 얼굴에서 한 번 구했다. 노이즈 때문에 FaceID가 찾는 5점이 조금 바뀔 수 있지만, EOT 변형(회전·크기·이동)이 이 차이를 덮는다고 가정했다.

## 깨달은 점
- FaceID는 내 얼굴을 붙이는 게 아니라, ArcFace가 뽑은 512개 숫자만 보고 얼굴까지 새로 그린다. 내 사진에서 생성 모델로 가는 정보는 그 숫자뿐이다.
- SimSwap은 숫자로만 나였지만, FaceID는 숫자와 눈 모두 나였다. 얼굴형과 머리까지 새로 그리기 때문이다. 그래서 FaceID가 실제로 막아야 할 위협이고, 그 숫자를 트는 Identity loss가 정확한 공격 지점이다.
- 노이즈가 딥페이크 도구 안의 모델을 직접 공격하면 크게 통하고, 다른 모델을 거치면 약해진다. 이름이 같은 ArcFace라도 다른 모델이다. 그래서 막고 싶은 도구가 쓰는 모델을 함께 공격해야 한다.
- 수치를 해석하려면 그 모델의 눈금(본인·타인 기준선)이 먼저 있어야 한다. 0.746이라는 숫자만으로는 높은지 알 수 없었다.
- 모델을 공격하려면 모델 파일만 있으면 안 되고, gradient가 흐르는 형태(PyTorch)와 그 모델이 사진을 자르는 방식(정렬)까지 똑같이 따라 해야 한다. 그냥 줄여서 넣었다면 과녁 옆을 겨눴을 것이다.
- 공격할 모델을 하나 더하면 그 모델은 확실히 막히지만, 같은 노이즈 크기 안에서 힘을 나눠 쓰므로 원래 모델의 여유가 줄어든다.
- 공격에 쓴 모델로 잰 점수(0.083)와 쓰지 않은 모델로 잰 점수(0.510)는 다르다. 진짜 효과를 말하려면 공격에 쓰지 않은 모델로도 재야 한다.

## 참고한 코드와 자료
- IP-Adapter: https://github.com/tencent-ailab/IP-Adapter
- IP-Adapter-FaceID: https://huggingface.co/h94/IP-Adapter-FaceID (비상업 연구 용도) — 모델 카드의 기본 설정(Realistic Vision, DDIM, VAE)을 따름
- diffusers IP-Adapter 문서의 FaceID 예시 — 명세서 모양 `(2, 1, 1, 512)`와 `ip_adapter_image_embeds` 사용법
- insightface: https://github.com/deepinsight/insightface (모델은 비상업 연구 용도)
- FaceShield: ArcFace 여러 개를 함께 공격 (`docs/papers/FaceShield_summary.md`)
- onnx2torch: https://github.com/ENOT-AutoDL/onnx2torch — onnx 모델을 PyTorch 모델로 변환
- insightface `face_align.norm_crop`의 112×112 기준 위치(arcface_dst) — FaceID 정렬 따라 하기
