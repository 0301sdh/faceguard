# Week 5 — Face swap 환경 준비, ArcFace Identity loss, Pilot 실험

> 진행 중 문서. 현재까지: SimSwap 실행 환경 구성, 데모 실행, 내 사진 baseline (ArcFace, FaceNet), pilot (ArcFace 기준 조건부 Go) 완료. 다음: pilot 결과를 FaceNet으로 채점.

## 목표
실제로 막으려는 딥페이크(face swap)를 직접 돌려보고, ArcFace를 공격한 보호 사진이 face swap 결과를 방해하는지 작은 실험으로 먼저 확인한다.

## 한 일
- [x] Face swap과 ArcFace 개념 정리
- [x] SimSwap 실행 환경 구성 (Google Colab, T4 GPU)
- [x] SimSwap 데모(영상 face swap) 실행 및 결과 확인
- [x] 사진 한 장 face swap으로 바꾸기
- [x] 내 원본 사진을 source로 face swap 실행 (baseline)
- [x] 결과물과 내 얼굴의 유사도 측정 (ArcFace, FaceNet)
- [x] `pgd_identity`를 ArcFace로 바꿔 보호 사진 생성 (pilot)
- [x] 보호 사진을 source로 face swap 실행 후 비교 (ArcFace 기준 조건부 Go)
- [ ] 보호 사진 SimSwap 결과를 FaceNet으로 채점 → 최종 Go/No-go 판단

## 핵심 개념 정리

### 1. Face swap의 구조 (몽타주 비유)
| 입력 | 역할 | 가져오는 것 |
|---|---|---|
| source (내 얼굴 사진) | 누구인가 | 신원 |
| target (다른 사람 사진/영상) | 어떤 상황인가 | 자세, 표정, 조명, 배경 |

```text
source → 얼굴 찾기·정렬 → ArcFace → 명세서(512차원 임베딩)
                                          ↓
target → 얼굴 찾기·정렬 → SimSwap이 명세서대로 얼굴을 다시 그림 → target에 다시 붙임
```
- SimSwap은 source 사진을 직접 보지 않는다. **ArcFace가 만든 명세서가 전부**다.
- 그래서 내 사진에 노이즈를 넣어 ArcFace가 틀린 명세서를 쓰게 만들면, 결과가 나를 닮지 않게 된다.
- Week 4에서는 FaceNet을 공격했고 다른 모델로 잘 전이되지 않았다. 이번에는 SimSwap이 실제로 쓰는 ArcFace를 직접 공격한다.

### 2. ArcFace vs FaceNet
| | FaceNet (Week 1~4) | ArcFace |
|---|---|---|
| 입력 크기 | 160×160 | 112×112 |
| 얼굴 준비 | 얼굴 영역을 네모로 잘라냄 | 눈·코·입 5개 점을 기준 위치에 맞춰 **정렬** |
| 비교 방법 | 코사인 유사도 | 코사인 유사도 |

- ArcFace도 얼굴 **전체**를 보고 명세서를 쓴다. 눈·코·입은 넣기 전에 위치를 맞추는 기준점이다.
- 공격 코드에서는 "사진 + δ → 정렬·112×112 → ArcFace" 단계가 추가된다. 노이즈가 정렬·축소를 거치며 뭉개질 수 있고, gradient가 정렬 단계를 지나 사진까지 흘러야 한다.

### 3. SimSwap 구성 부품
| 부품 | 파일 | 역할 |
|---|---|---|
| antelope | `insightface_func/models/antelope/` | 얼굴 찾기 + 정렬 |
| ArcFace | `arcface_model/arcface_checkpoint.tar` | 명세서 작성 (**공격 대상**) |
| SimSwap 본체 | `checkpoints/people/` | 명세서대로 얼굴 그리기 |
| parsing | `parsing_model/checkpoint/79999_iter.pth` | 합성한 얼굴을 자연스럽게 붙이기 |

### 4. 실행 환경
- SimSwap은 2021년 코드(Python 3.6, PyTorch 1.8, CUDA 전제)라 로컬 Mac(Python 3.14, M2)에서는 돌리지 않고 Colab에서 실행한다.
- SimSwap은 **보호 효과를 시험하는 상대**로만 쓴다. 서비스(FastAPI)에 들어가는 것은 노이즈 생성 코드와 ArcFace뿐이다.
- Colab 임시 컴퓨터의 파일(clone한 코드, 가중치, 결과물)은 연결이 끊기면 사라진다. 노트북 셀과 출력은 Drive 사본에 남는다.

## 환경 구성 과정 (오류와 해결)
- 공식 Colab 노트북(`SimSwap colab.ipynb`)을 Drive에 사본 저장해서 실행했다.
- 수정한 노트북: `notebooks/simswap_eval.ipynb`
- 대부분 **2021년 코드를 최신 Colab(Python 3.13)에서 돌려서** 생긴 문제였다.


- 경고(`SyntaxWarning`, `UserWarning`)는 실행을 멈추지 않으므로 무시했다. 오류는 `Traceback` + `...Error`로 멈춘다.
- ArcFace 가중치를 불러올 때 필요했던 `weights_only=False`는 Week 5 공격 코드에서 ArcFace를 불러올 때도 똑같이 필요하다.

## 결과

### SimSwap 데모 (영상)
- source: `demo_file/Iron_man.jpg`, target: `demo_file/multi_people_1080p.mp4` (594프레임, 1080p)
- 결과: 영상 속 얼굴이 source 신원으로 잘 바뀌었고, target의 표정·고개 방향을 따라 움직였다.
- 처리 속도: 프레임당 약 1.25초, 전체 약 12분 (T4 GPU)
  - 얼굴 검출(onnxruntime)은 CPU, 합성(PyTorch)은 GPU에서 실행됨
- 결과 영상은 실제 인물이 나오므로 `data/week5/`에만 저장 (git 제외)

### Baseline: 내 원본 사진으로 face swap (사진 한 장)
- 방법: SimSwap의 `test_wholeimage_swapsingle.py`를 노트북 셀로 옮겨 실행 (`notebooks/simswap_eval.ipynb`)
- target: `demo_file/specific1.png` (한 사람이 정면으로 나온 데모 사진)
- source: 내 사진 4장 (`person_a_1` ~ `person_a_4`), 각각 같은 target에 합성 → `swap_person_a_N.jpg`
- 유사도: 세 장을 SimSwap과 같은 방식(정렬 → 정규화 → 112×112 → SimSwap 내부 ArcFace)으로 임베딩한 뒤 코사인 유사도 계산

**ArcFace (SimSwap 내부 모델) 기준**
| source | 결과 vs 내 사진 | 결과 vs target | 내 사진 vs target (타인 기준) |
|---|---|---|---|
| `person_a_1` | **0.729** | 0.201 | 0.062 |
| `person_a_2` | **0.749** | 0.134 | -0.049 |
| `person_a_3` | **0.713** | 0.149 | -0.027 |
| `person_a_4` | **0.745** | 0.144 | -0.010 |
| 평균 | **약 0.734** | 약 0.157 | 약 -0.006 |

- 재현 확인: `person_a_1`은 첫 세션에서 0.732, 새 세션에서 처음부터 다시 실행해 0.729. 작은 차이는 결과를 JPG로 저장할 때의 압축 손실 정도로 본다.
- 코사인 유사도는 -1~1 범위이고, 0 근처(음수 포함)는 관계없는 사람이라는 뜻이다.


**눈으로 본 결과**
- 여러 사진 모두 target의 얼굴은 바뀌었지만 **내 얼굴처럼 보이지 않았다.**
- target의 머리 모양, 얼굴 윤곽, 피부색이 그대로 남아 "target 사람이 조금 바뀐 얼굴"로 보였다.

**FaceNet (SimSwap과 관계없는 모델) 기준**
- 코드: `experiments/week5_facenet_baseline.py` (Week 1과 같은 MTCNN + InceptionResnetV1 vggface2, Mac에서 실행)
- 입력: `data/week5/swap_person_a_N.jpg`, `data/week5/target_specific1.png` (Colab에서 내려받음, git 제외)

기준값 (이번 사진들로 다시 측정)
| 비교 | 값 |
|---|---|
| 내 원본 사진끼리 (4장 중 2장씩 6쌍 평균) — 본인 기준 | **0.761** |
| 내 원본 사진 vs target (4쌍 평균) — 타인 기준 | **0.044** |

합성 결과
| source | 결과 vs 합성에 쓴 내 사진 | 결과 vs 합성에 쓰지 않은 내 사진 (3장 평균) | 결과 vs target |
|---|---|---|---|
| `person_a_1` | 0.533 | 0.572 | 0.243 |
| `person_a_2` | 0.680 | 0.667 | 0.205 |
| `person_a_3` | 0.613 | 0.541 | 0.366 |
| `person_a_4` | 0.594 | 0.533 | 0.333 |
| 범위 | **0.53 ~ 0.68** | **0.53 ~ 0.67** | 0.21 ~ 0.37 |

- 예: `person_a_1` 줄은 `swap_person_a_1.jpg`를 `person_a_1`(합성에 쓴 사진), `person_a_2~4`(쓰지 않은 사진), `specific1`(target)과 비교한 것이다.

**두 목격자 비교**
| 목격자 | 결과 vs 내 사진 | 본인 기준 | 타인(target) 기준 |
|---|---|---|---|
| ArcFace (SimSwap 내부) | 평균 0.734 | (미측정) | 약 -0.006 |
| FaceNet (독립) | 0.53 ~ 0.68 | 0.761 | 0.044 |

참고: Week 1 FaceNet 기준값은 본인 0.743 / 타인(person_b) 0.372, 논문의 SimSwap 원본 ISM은 0.544 (CelebA-HQ).

## Pilot: SimSwap 내부 ArcFace 공격

### 방법
- 노트북: `notebooks/simswap_eval.ipynb` (Colab)
- Week 4의 `pgd_identity` 구조를 그대로 쓰고, **속일 목격자만** FaceNet → SimSwap 내부 ArcFace(`model.netArc`)로 바꿨다.
  - 공격할 얼굴: SimSwap이 정렬해서 잘라낸 224×224 얼굴(`app.get`). Week 4에서 MTCNN 160×160 얼굴에 공격한 것과 같은 단순화
  - `embed`: ImageNet 정규화 → 112×112 축소 → ArcFace → 길이 1로 맞춤 (SimSwap이 명세서를 만드는 순서와 같게)
  - PGD: steps 30, 보폭 `2.5 × eps / steps` (12/255에서 FaceShield와 같은 1/255), eps 4, 8, 12/255
  - 보호 얼굴은 **PNG**로 저장 (JPG 압축은 노이즈를 손실시킴)
- 평가: 보호 얼굴을 source로 SimSwap 실행 → 결과 vs 원본 내 사진 (ArcFace)
  - **(A) 직접:** 보호 얼굴을 그대로 ArcFace에 넣음. 노이즈가 손상 없이 전달되는 최선의 경우
  - **(B) 재정렬:** SimSwap이 보호 사진에서 얼굴을 다시 찾고 다시 정렬함. 실제로 딥페이크를 만들 때와 같은 경우. 얼굴만 꽉 찬 224×224 사진은 검출이 안 돼서 둘레에 112픽셀 검은 여백을 붙인 뒤 검출

### 트러블슈팅 1: 노이즈가 전혀 움직이지 않음
- **문제:** Week 4 코드를 목격자만 바꿔 그대로 썼더니, 보호 얼굴 vs 원본 얼굴이 **12개 전부 1.000**이었다.
- **확인:** 원본 얼굴에서 기울기를 직접 계산했다.
  ```text
  유사도: 1.0
  기울기 최대 크기: 0.0
  기울기가 0이 아닌 픽셀 비율: 0.0
  ```
- **원인:** 원본끼리의 유사도는 최댓값(1.0)이다. 최댓값 지점은 산꼭대기처럼 평평해서 기울기가 0이고, `sign(0) = 0`이라 PGD가 한 걸음도 움직이지 못한다.
- **해결:** **랜덤 출발(random start).** ±eps 안의 임의 지점에서 출발한다. 표준 PGD(Madry 등, 2018)의 방식이며, Week 3~4에서는 빠뜨렸던 부분이다.
  ```python
  adv = (face + torch.empty_like(face).uniform_(-epsilon, epsilon)).clamp(0, 1).detach()
  ```
- Week 4에서는 같은 구조인데도 움직였다. 계산 오차로 기울기가 정확히 0이 아니었고, `sign()`이 그 작은 값을 ±1로 키워 준 것으로 보인다(추정). 이번에는 명세서를 길이 1로 맞추는 과정(`F.normalize`)까지 들어가 그 오차마저 사라진 것으로 보인다.

### 트러블슈팅 2: 재정렬하면 노이즈 효과가 사라짐
- **문제:** 랜덤 출발 후 (A) 직접은 음수까지 떨어졌지만, **(B) 재정렬은 baseline 근처**였다.

  | eps | (A) 직접 | (B) 재정렬 |
  |---|---|---|
  | 0 | 0.71 ~ 0.75 | 0.68 ~ 0.77 |
  | 4/255 | -0.33 ~ -0.01 | 0.58 ~ 0.73 |
  | 8/255 | -0.50 ~ -0.43 | 0.55 ~ 0.74 |
  | 12/255 | -0.52 ~ -0.40 | 0.32 ~ 0.66 |

- **가설:** `embed`의 `F.interpolate(x, size=(112, 112))`는 기본 방식이 nearest라서, 224 → 112로 줄일 때 **2×2 칸 중 1칸만 뽑는다.** 뽑히지 않는 3칸은 결과에 영향을 주지 않아 기울기가 0이고, PGD는 그 칸을 다듬지 못한다. 그래서 **의미 있는 노이즈가 뽑히는 1칸에만 몰려 있다.** 재정렬로 얼굴 위치가 1칸만 어긋나도 다듬지 않은 칸이 뽑혀 효과가 사라진다.
- **확인:** 보호 얼굴(`person_a_1`, 8/255)과 원본 얼굴을 똑같이 옆으로 밀어서 비교했다.

  | 이동 | 보호 vs 원본 | |
  |---|---|---|
  | 0픽셀 | -0.716 | 다듬은 칸이 뽑힘 → 효과 있음 |
  | **1픽셀** | **0.982** | 다듬지 않은 칸이 뽑힘 → 효과 사라짐 |
  | 2픽셀 | -0.547 | 다시 다듬은 칸이 뽑힘 → 효과 돌아옴 |

  멀리 밀수록 약해지는 것이 아니라 **홀수 칸이면 사라지고 짝수 칸이면 돌아온다.** 원인이 "2칸 중 1칸만 뽑는 축소"라는 것을 보여 준다.
- **해결:** 노이즈를 만들 때 **4가지 이동((0,0), (0,1), (1,0), (1,1))에서 각각 유사도를 구해 평균을 낮춘다.** 이동마다 뽑히는 칸이 달라서 4칸 모두 결과에 영향을 주게 되고, 모든 칸에 기울기가 생겨 전부 다듬어진다. 여러 변형을 거쳐도 통하는 노이즈를 만드는 이 방법을 **EOT(Expectation over Transformation)**라고 한다. FaceShield가 MTCNN 공격에서 축소 방식을 여러 개 섞은 것과 같은 아이디어다.
  ```python
  SHIFTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
  sims = [F.cosine_similarity(embed(shift(adv, dy, dx)), orig).mean()
          for (dy, dx), orig in zip(SHIFTS, orig_embs)]
  loss = sum(sims) / len(sims)
  ```

### 최종 결과 (랜덤 출발 + 이동 평균)

보호 얼굴 vs 원본 얼굴 (ArcFace)
| eps | 보호 vs 원본 | 1픽셀 이동 후 |
|---|---|---|
| 4/255 | -0.34 ~ -0.06 | -0.42 ~ 0.04 |
| 8/255 | -0.70 ~ -0.47 | -0.74 ~ -0.62 |
| 12/255 | -0.77 ~ -0.70 | -0.81 ~ -0.73 |

SimSwap 결과 vs 원본 내 사진 (ArcFace)
| source | eps | (A) 직접 | (B) 재정렬 |
|---|---|---|---|
| `person_a_1` | 0 | 0.729 | 0.774 |
| | 4 | -0.181 | 0.005 |
| | 8 | -0.353 | 0.010 |
| | 12 | -0.443 | -0.139 |
| `person_a_2` | 0 | 0.749 | 0.684 |
| | 4 | -0.127 | 0.251 |
| | 8 | -0.455 | -0.077 |
| | 12 | -0.544 | -0.153 |
| `person_a_3` | 0 | 0.713 | 0.699 |
| | 4 | -0.178 | 0.281 |
| | 8 | -0.357 | 0.085 |
| | 12 | -0.387 | -0.230 |
| `person_a_4` | 0 | 0.745 | 0.733 |
| | 4 | -0.042 | 0.392 |
| | 8 | -0.404 | -0.176 |
| | 12 | -0.440 | -0.229 |

**(B) 재정렬: 고치기 전 vs 고친 후**
| eps | 고치기 전 | 고친 후 |
|---|---|---|
| 0 | 0.68 ~ 0.77 | 0.68 ~ 0.77 |
| 4/255 | 0.58 ~ 0.73 | **0.01 ~ 0.39** |
| 8/255 | 0.55 ~ 0.74 | **-0.18 ~ 0.09** |
| 12/255 | 0.32 ~ 0.66 | **-0.23 ~ -0.14** |

### Go / No-go: 조건부 Go
- **SimSwap 내부 ArcFace 기준으로는 Go.** 8/255부터 재정렬을 거쳐도 SimSwap 결과가 관계없는 사람(0 근처 이하) 수준으로 떨어졌다 (baseline 0.73).
- 남은 확인: ArcFace로 만든 노이즈를 ArcFace로 채점한 유리한 조건이다. **SimSwap과 관계없는 FaceNet으로도 떨어지는지** 확인해야 한다.
- 조건: 얼굴 조각(224×224)에 노이즈를 넣었고, 재정렬 검출을 위해 여백을 붙였다. 사진 전체에 노이즈를 넣는 실제 사용 상황은 아직 시험하지 않았다.

## 해석

### 오류 5개는 사실상 한 가지 원인이었다
- SimSwap은 2021년 환경(Python 3.6, PyTorch 1.8)에 맞춰 작성됐고, Colab은 2026년 환경(Python 3.13, 최신 PyTorch·numpy)이다.
- 그 사이에 바뀐 것들이 하나씩 오류로 나타났다.
  - 이름이 바뀜: `google_drive_downloader`
  - 기본값이 바뀜: `torch.load`의 `weights_only`, SimSwap 옵션 `crop_size`
  - 없어짐: `np.float`, `np.int`
  - 링크가 만료됨: antelope OneDrive 링크
- 즉 SimSwap 자체가 잘못된 게 아니라 **코드와 실행 환경의 시간 차이** 문제였다. 오류 메시지가 대부분 원인과 해결법을 직접 알려 주었다(`weights_only`, `np.float`).

### 데모가 잘 된 이유
- 영상 속 얼굴이 source 신원으로 바뀌면서도 target의 표정·고개 방향을 따라 움직였다.
- 신원은 ArcFace 명세서에서, 나머지(자세, 표정, 조명)는 target에서 가져오는 구조가 그대로 드러난 결과다.
- 명세서는 source 사진에서 **한 번만** 만들고 594프레임 전부에 같은 명세서를 썼다. 그래서 명세서를 망가뜨리면 영상 전체가 영향을 받는다. 보호 노이즈가 노려야 할 지점이 바로 여기다.

### 속도
- 프레임당 약 1.25초로 느린 편이었다. 1080p 프레임 전체에서 얼굴을 찾는 검출 모델(onnxruntime)이 GPU가 아니라 CPU에서 돌았기 때문으로 보인다.
- 실험은 사진 한 장씩 처리하므로 문제되지 않는다.

### Baseline: ArcFace는 "나"라고 판단하는데 눈으로는 닮지 않았다
- 결과 vs 내 사진(0.71~0.75)이 타인 기준(-0.05~0.06, 0 근처)보다 훨씬 높다. **ArcFace 기준으로는 내 신원이 확실히 옮겨 갔다.**
- 내 사진 4장 모두 0.71~0.75로 좁게 모였다. 어떤 내 사진을 넣어도 SimSwap은 나를 안정적으로 복제하므로, 보호 효과를 시험할 기준선으로 쓰기 좋다.
- 결과 vs target(0.13~0.20)은 타인 기준보다는 높다. target의 신원(머리 모양, 얼굴형 등)도 일부 남아 있다.
- 그런데 눈으로는 닮지 않았다. SimSwap은 **신원만** 바꾸고 머리 모양, 얼굴 윤곽, 피부색, 조명은 target에서 가져온다. 사람은 이런 전체 인상으로 누구인지 판단하므로 target 사람처럼 보인다. 224×224 해상도라 눈매, 코 같은 세부 특징도 흐려진다.
- 여러 사진에서 모두 같았으므로 사진 문제가 아니라 **SimSwap(224 모델)의 특성**으로 본다.
- **주의:** 0.73은 SimSwap 안의 ArcFace로 잰 값이다. SimSwap은 학습할 때 바로 이 ArcFace가 source와 같은 사람이라고 판단하도록 훈련됐으므로, 같은 모델로 재면 유리하게 높게 나온다. 그래서 SimSwap과 관계없는 FaceNet으로도 재야 한다 (Week 4의 "노이즈를 만들 때 쓰지 않은 모델로 평가한다"와 같은 이유).

### Baseline: FaceNet도 "나에 가깝다"고 판단했다
- 결과 vs 내 사진(0.53~0.68)이 타인 기준(target 0.044, Week 1 타인 0.372)보다 확실히 높다. **SimSwap과 관계없는 목격자로 재도 내 신원이 옮겨 갔다.**
- **합성에 쓰지 않은 사진과 비교해도 비슷하다**(0.53~0.67). 내가 올린 그 사진만 닮은 것이 아니라 "평소의 나"를 닮았다는 뜻이라, 실제 위협으로 볼 수 있다.
- 다만 진짜 내 사진끼리(0.761)보다는 낮다. FaceNet 기준으로는 "나와 꽤 닮은 사람" 수준이다. SimSwap 내부 ArcFace 값이 유리하게 나온다는 예상과 맞고, 눈으로 봤을 때 "나 같지 않다"고 느낀 것과도 어느 정도 맞는다.
- 결과 vs target(0.21~0.37)도 타인 기준보다 높다. ArcFace에서처럼 target의 흔적이 남아 있다.
- **타인 기준값은 비교 대상에 따라 크게 달라진다.** Week 1의 타인(person_b)은 0.372, 이번 target은 0.044였다. 그래서 판단할 때 타인 기준 하나만 보지 않고 **본인 기준(0.761)과의 거리**도 함께 본다.


## 유의할 점
- "Mac에는 저장만 하고, 코드를 만지려면 Colab을 연다." 노트북은 Colab에서 고치고 실행하며, Mac에는 `git pull`로 받아 보관만 한다.

## 깨달은 점
- "arcface가 명세서를 simswap에 전달하면 그걸 바탕으로 딥페이크를 하는거니까, arcface의 명세서를 노이즈로 방해하는 것이다." → 정확히는 노이즈는 명세서가 아니라 **사진**에 넣고, ArcFace가 그 사진을 보고 스스로 틀린 명세서를 쓰게 만든다.
- "우리가 simswap을 쓰는 이유는 단지 테스트 때문이다."
- "꼭대기는 평평해서 기울기가 0이라 기울기로 계산이 불가능하다. Week 4에서는 우연히 0.0000001이라 됐던 것이다."
- "노이즈는 모든 칸에 있지만 다듬은(의미 있는) 노이즈는 4칸 중 1칸에만 있다. 나머지 3칸은 기울기가 0이라 PGD가 다듬지 못한다. 그래서 모든 칸에 기울기가 생기도록 계산 방식을 바꿔야 한다."

### 틀린 예측 (pilot)
- **예측:** Week 4의 PGD를 목격자만 바꾸면 그대로 될 것이다. → **실제:** 1.000으로 전혀 움직이지 않았다 (트러블슈팅 1).
- **예측:** 보호 얼굴이 ArcFace를 속이면 SimSwap도 막힌다. → **실제:** 직접 넣을 때만 막혔고, SimSwap이 다시 정렬하면 효과가 사라졌다 (트러블슈팅 2).
- **예측:** 4가지 위치를 동시에 만족시켜야 하니 효과가 약해질 것이다. → **실제:** 보호 vs 원본 유사도는 거의 줄지 않았다.

### 다음 단계에서 활용할 것
- **보호 효과의 기준선이 생겼다.** 원본 사진으로 합성한 결과 vs 내 사진 = 평균 0.73 (ArcFace, 4장), 0.53~0.68 (FaceNet). 보호 사진으로 합성했을 때 이 값이 타인 기준(ArcFace 0 근처, FaceNet 0.04) 쪽으로 내려가는지가 pilot의 판단 기준이다. ArcFace로 노이즈를 만들 것이므로 **FaceNet 값도 함께 떨어지는지**가 전이성 확인이 된다.
- **눈으로 보여 주기는 어렵다.** 보호 전부터 눈으로는 닮지 않았으므로, 포트폴리오나 웹에서 "보호 전/후" 차이를 사진으로 보여 주려면 나와 머리 모양·얼굴형이 비슷한 target을 고르거나 SimSwap 512 모델을 검토한다.
- **SimSwap은 2021년 모델이다.** 결과 품질은 최신 도구보다 낮지만, 신원을 ArcFace 계열 명세서로 옮기는 구조는 최신 도구에도 남아 있다. 그래서 SimSwap에서 먼저 확인하고, Week 7~8에 최신 도구로 넓혀 시험한다.


## 참고한 코드와 자료
- SimSwap: https://github.com/neuralchen/SimSwap (CC BY-NC 4.0, 학술·비상업 용도) — 공식 Colab 노트북과 `test_wholeimage_swapsingle.py`를 바탕으로 실행. 최신 환경에서 돌리기 위한 수정, 유사도 측정 셀, pilot 공격 코드는 AI(Claude)의 도움을 받아 작성. 트러블슈팅의 실행과 결과 확인, 원인 이해는 직접 진행
- PGD 랜덤 출발: Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (ICLR 2018)
- EOT: Athalye et al., *Synthesizing Robust Adversarial Examples* (ICML 2018)
- insightface antelopev2: https://github.com/deepinsight/insightface/releases/tag/v0.7
- FaceShield 논문 요약: `docs/papers/FaceShield_summary.md`
