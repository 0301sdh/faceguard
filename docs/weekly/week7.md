# Week 7 — 신원 유지 이미지 생성(IP-Adapter FaceID) 확장

> **결론: Go.** 보호 안 한 사진 한 장으로 FaceID는 나를 확실히 복제한다(FaceNet 0.82, 눈으로도 닮음). Week 6 노이즈는 FaceID에 약하게만 통했다. SimSwap ArcFace와 FaceID ArcFace를 함께 공격(ensemble)하자 FaceID 결과가 ArcFace 기준 남 수준(0.062)이 됐고, 눈으로도 내가 아니다. SimSwap 보호도 유지된다. 공격에 쓰지 않은 FaceNet에는 흔적이 남아 있다(0.438). 이 부분은 한계로 기록하고 Week 8로 넘긴다.

## 목표
실제로 막고 싶은 위협("내 얼굴이 다른 장면·몸에 들어간 이미지")을 만드는 신원 유지 이미지 생성(IP-Adapter FaceID)으로 평가 대상을 넓히고, Week 5~6의 Identity loss 노이즈가 통하는지 확인한다.

## 핵심 개념
- **Diffusion:** 랜덤 노이즈에서 시작해 UNet이 노이즈를 예측하고 빼는 일을 약 30번 반복해 그림을 새로 그린다. 프롬프트 같은 힌트는 cross-attention으로 들어간다.
- **IP-Adapter FaceID:** 내 사진 → insightface ArcFace → 512개 숫자(얼굴 명세서) → cross-attention. 생성 모델은 내 사진 픽셀을 보지 않고, 이 숫자만 보고 얼굴까지 새로 그린다.
- **SimSwap과의 차이:** 둘 다 ArcFace 숫자를 가져가지만, SimSwap은 target 얼굴 위에 내 특징을 입히고 FaceID는 얼굴·머리·몸을 전부 새로 그린다.
- **Ensemble:** 같은 Identity loss를 여러 모델에 걸어 합을 낮춘다.

```text
loss = cos(SimSwap ArcFace(x+δ), 원본) + cos(FaceID ArcFace(정렬(x+δ)), 원본)  → 최소화
```

## 환경
- Colab T4. 생성은 `notebooks/ipadapter_faceid_eval.ipynb`, 노이즈는 `notebooks/simswap_robustness.ipynb` 맨 아래에서 만든다.
- 모델: Realistic Vision V4.0 + sd-vae-ft-mse, DDIM, `ip-adapter-faceid_sd15.bin`, scale 1.0, LoRA 없음
- 생성 설정: 512×512, 30 steps, 프롬프트 `upper body photo of a person wearing a suit in an office`, 생성 seed 0/1/2. 이미지당 약 5초
- 노이즈 설정: Week 6 최종과 같다 (eps 5/255, EOT 변형 4개, steps 30)

### 트러블슈팅
| 문제 | 원인 | 해결 |
|---|---|---|
| `cannot import name 'FqnToConfig'` | 최신 diffusers와 Colab의 옛 torchao 버전 불일치 | `pip uninstall torchao` (양자화를 쓰지 않음) |
| `CUDAExecutionProvider` 없음 | CPU용 onnxruntime이 우선됨 | CPU로 실행 (검출·명세서는 CPU로 충분) |
| insightface ArcFace는 `.onnx`라 gradient 없음 | 계산만 하는 완성품 형식 | onnx2torch로 PyTorch 변환 (onnx와 유사도 1.00000, gradient 0인 비율 0%) |
| 112로 그냥 줄이면 FaceID가 보는 얼굴과 다름 (0.94 ~ 0.96) | FaceID는 얼굴을 다시 찾아 5점을 자기 기준 위치에 맞춰 자름 | 같은 정렬을 미분 가능한 `grid_sample`로 따라 함 (0.9999) |

## 결과

### 1. 눈금 잡기
| 모델 | 실패 (나로 봄) | 성공 (남으로 봄) |
|---|---|---|
| FaceID ArcFace (insightface) | ≥ 0.510 | < 0.222 |
| FaceNet | ≥ 0.686 | < 0.365 |

- Week 5와 같은 방식이다. 실패선은 내 사진끼리의 최소값, 성공선은 그 값과 타인(target_specific1)의 중간이다.

### 2. FaceID: 어떤 사진을 넣었나에 따른 결과
생성 결과 vs 내 사진, 사진 4장 평균이다.

| 입력 | ArcFace | FaceNet | 눈으로 |
|---|---|---|---|
| Baseline: 보호 안 함 | 0.728 (실패) | 0.820 (실패) | 4장 모두 닮음 |
| Pilot: Week 6 보호 사진 (노이즈 seed 0) | 0.476 (부분) | 0.685 (부분) | - |
| **Ensemble 보호 사진 (노이즈 seed 3개)** | **0.062** (성공) | **0.438** (부분) | **4장 모두 다른 사람** |

- Pilot은 Week 6 노이즈를 그대로 넣어 전이를 시험했다. 결과는 조건부 No-go였다. 이름이 같은 ArcFace라도 별개의 모델이라 약하게만 통했다.
- Ensemble은 노이즈 seed마다 ArcFace 0.083 / 0.080 / 0.023으로 안정적이었다. 36장 중 성공선을 넘은 것은 1장(0.229)이다.
- FaceNet은 노이즈 seed마다 0.510 / 0.468 / 0.325로 흔들렸다. ArcFace를 공격하면서 덤으로 얻는 효과라서다.

### 3. 눈금 확인: FaceNet 0.438은 정말 '부분'인가
타인을 1명이 아니라 많이 넣어 확인했다. 타인은 LFW(연구 공개용 얼굴 데이터셋)에서 가져왔다.

| 비교 | ArcFace | FaceNet |
|---|---|---|
| 실제 사진 타인 500명 vs 내 사진 | - | 평균 0.007, 상위 1% 0.350 |
| FaceID가 타인을 그림 (무작위 16명) | 0.012 | 0.044 |
| FaceID가 타인을 그림 (동양인 남성 8명) | 0.036 | 0.078 |
| FaceID가 보호된 나를 그림 (ensemble) | 0.062 | 0.398* |

\* 타인과 같은 잣대(내 사진 4장 평균)로 다시 잰 값

- **눈금은 맞았다.** 성공선 0.365는 실제 남 100명 중 가장 닮은 1명 수준이다.
- **ArcFace 기준으로 ensemble은 남과 구별되지 않는다.**
- **FaceNet 기준으로는 남보다 확실히 나를 닮았다.** 생성기 특성 때문도 아니고 인종·성별 때문도 아니다. 노이즈가 ArcFace가 보는 신원은 지웠지만, FaceNet이 잡는 흔적은 남겼다.

### 4. 비교 표: loss에 넣은 모델별 보호 효과
결과 vs 내 사진이다. 낮을수록 보호가 잘 된 것이다.

| loss에 넣은 모델 | SimSwap: ArcFace | SimSwap: FaceNet | FaceID: ArcFace | FaceID: FaceNet | PSNR |
|---|---|---|---|---|---|
| 보호 안 함 | 0.72 (실패) | 0.629 (부분) | 0.728 (실패) | 0.820 (실패) | - |
| SimSwap ArcFace (Week 6) | **-0.129** (성공) | **0.297** (성공) | 0.476 (부분) | 0.685 (부분) | 36.7 ~ 37.1dB |
| + FaceID ArcFace (ensemble) | -0.025 (성공) | 0.340 (성공) | **0.062** (성공) | **0.438** (부분) | 36.3 ~ 36.6dB |

ArcFace 열은 도구마다 다른 모델(SimSwap의 것, FaceID의 것)이다. FaceNet은 공격에 쓰지 않은 같은 모델이다. SimSwap ArcFace 성공선은 0.2(Week 5)다.

- **loss에 넣은 모델이 들어 있는 칸만 확실히 막힌다.**
- 모델을 추가하면 새 도구는 크게 좋아지고, 원래 도구는 조금 손해를 본다(같은 eps 안에서 힘을 나눠 씀). 그래도 원래 도구는 성공을 유지했다. 화질 손해는 거의 없다.
- 남은 칸은 FaceID × FaceNet 하나다.

## Go / No-go: Go
- FaceID가 신원을 읽는 모델은 완전히 속았다. 남의 사진으로 만든 딥페이크와 구별하지 못한다.
- 공격하지 않은 FaceNet에서도 "나"의 범위(0.686) 밖이다. 보호 안 함의 절반 수준이다.
- 눈으로 봐도 4장 모두 내가 아니다.
- SimSwap 보호도 유지된다.

**Week 8 개선 후보:** FaceNet 등 성격이 다른 모델을 ensemble에 추가하고 채점은 또 다른 독립 모델로 하기, eps나 steps 조정, 나를 아는 사람들의 블라인드 테스트

## 한계
- FaceNet에 나의 흔적이 남아 있다(0.398, 남은 0.04 ~ 0.08).
- 눈으로 본 판단은 본인 1명의 것이다. 사진 4장, 본인 1명의 결과다.
- 프롬프트 1개(상반신)만 시험했다. LoRA는 쓰지 않았다.
- 노이즈 seed 0의 보호 사진은 반복 실행 중에 다시 만들어졌다(차이는 0.00x 수준).
- FaceID 정렬 변환은 보호 전 얼굴에서 한 번 구했다. 노이즈로 5점이 조금 바뀌는 것은 EOT가 덮는다고 가정했다.

## 깨달은 점
- FaceID는 ArcFace가 뽑은 512개 숫자만 보고 얼굴까지 새로 그린다. 그래서 그 숫자를 트는 Identity loss가 정확한 공격 지점이다.
- 노이즈는 공격한 모델에만 확실히 통한다. 이름이 같은 ArcFace라도 다른 모델이면 약해진다. 막고 싶은 도구의 모델을 공격에 넣어야 한다.
- 모델을 공격하려면 gradient가 흐르는 형태와, 그 모델이 사진을 자르는 방식(정렬)까지 똑같이 따라 해야 한다.
- 공격에 쓴 모델의 점수와 쓰지 않은 모델의 점수는 다르다. 진짜 효과는 공격에 쓰지 않은 모델로 재야 하고, 반복해야 안정적인 효과와 덤이 구별된다.
- 판정선은 타인을 많이 넣어 확인해야 믿을 수 있다. 가장 공정한 기준은 같은 조건(같은 생성기로 그린 남)이다.

## 자료
- 숫자 전체: `results/week7/` (`arcface_colab.txt`, `facenet_ensemble.txt`, `facenet_simswap_ensemble.txt`, `facenet_calibration_lfw.txt`, `facenet_stranger_gen.txt`)
- IP-Adapter: https://github.com/tencent-ailab/IP-Adapter, IP-Adapter-FaceID: https://huggingface.co/h94/IP-Adapter-FaceID (비상업 연구 용도)
- insightface: https://github.com/deepinsight/insightface (비상업 연구 용도), onnx2torch: https://github.com/ENOT-AutoDL/onnx2torch
- LFW: http://vis-www.cs.umass.edu/lfw/ (연구 공개용)
- FaceShield: 여러 모델을 함께 공격 (`docs/papers/FaceShield_summary.md`)
