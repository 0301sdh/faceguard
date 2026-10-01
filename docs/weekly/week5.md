# Week 5 — Face swap 환경 준비, ArcFace Identity loss, Pilot 실험

> 진행 중 문서. 현재까지: SimSwap 실행 환경 구성 + 데모 실행 완료.

## 목표
실제로 막으려는 딥페이크(face swap)를 직접 돌려보고, ArcFace를 공격한 보호 사진이 face swap 결과를 방해하는지 작은 실험으로 먼저 확인한다.

## 한 일
- [x] Face swap과 ArcFace 개념 정리
- [x] SimSwap 실행 환경 구성 (Google Colab, T4 GPU)
- [x] SimSwap 데모(영상 face swap) 실행 및 결과 확인
- [ ] 사진 한 장 face swap으로 바꾸기
- [ ] 내 원본 사진을 source로 face swap 실행 (baseline)
- [ ] 결과물과 내 얼굴의 유사도 측정 (ArcFace, FaceNet)
- [ ] `pgd_identity`를 ArcFace로 바꿔 보호 사진 생성 (pilot)
- [ ] 보호 사진을 source로 face swap 실행 후 비교, Go/No-go 판단

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
- 3번은 Week 5 공격 코드에서 ArcFace를 불러올 때도 똑같이 필요하다.

## 결과

### SimSwap 데모 (영상)
- source: `demo_file/Iron_man.jpg`, target: `demo_file/multi_people_1080p.mp4` (594프레임, 1080p)
- 결과: 영상 속 얼굴이 source 신원으로 잘 바뀌었고, target의 표정·고개 방향을 따라 움직였다.
- 처리 속도: 프레임당 약 1.25초, 전체 약 12분 (T4 GPU)
  - 얼굴 검출(onnxruntime)은 CPU, 합성(PyTorch)은 GPU에서 실행됨
- 결과 영상은 실제 인물이 나오므로 `data/week5/`에만 저장 (git 제외)

### Baseline (예정)
| source | target | 결과 vs 내 원본 (ArcFace) | 결과 vs 내 원본 (FaceNet) |
|---|---|---|---|
| 내 원본 사진 | | | |

참고: Week 1 FaceNet 기준값은 본인 0.743 / 타인 0.372, 논문의 SimSwap 원본 ISM은 0.544 (CelebA-HQ).

## 해석

### 오류 5개는 사실상 한 가지 원인이었다
- SimSwap은 2021년 환경(Python 3.6, PyTorch 1.8)에 맞춰 작성됐고, Colab은 2026년 환경(Python 3.13, 최신 PyTorch·numpy)이다.
- 그 사이에 바뀐 것들이 하나씩 오류로 나타났다.
  - 이름이 바뀜: `google_drive_downloader`
  - 기본값이 바뀜: `torch.load`의 `weights_only`, SimSwap 옵션 `crop_size`
  - 없어짐: `np.float`, `np.int`
  - 링크가 만료됨: antelope OneDrive 링크
- 즉 SimSwap 자체가 잘못된 게 아니라 **코드와 실행 환경의 시간 차이** 문제였다. 오류 메시지가 대부분 원인과 해결법을 직접 알려 주었다(3번, 5번).

### 데모가 잘 된 이유
- 영상 속 얼굴이 source 신원으로 바뀌면서도 target의 표정·고개 방향을 따라 움직였다.
- 신원은 ArcFace 명세서에서, 나머지(자세, 표정, 조명)는 target에서 가져오는 구조가 그대로 드러난 결과다.
- 명세서는 source 사진에서 **한 번만** 만들고 594프레임 전부에 같은 명세서를 썼다. 그래서 명세서를 망가뜨리면 영상 전체가 영향을 받는다. 보호 노이즈가 노려야 할 지점이 바로 여기다.

### 속도
- 프레임당 약 1.25초로 느린 편이었다. 1080p 프레임 전체에서 얼굴을 찾는 검출 모델(onnxruntime)이 GPU가 아니라 CPU에서 돌았기 때문으로 보인다.
- 실험은 사진 한 장씩 처리하므로 문제되지 않는다.


## 유의할 점
- "Mac에는 저장만 하고, 코드를 만지려면 Colab을 연다." 노트북은 Colab에서 고치고 실행하며, Mac에는 `git pull`로 받아 보관만 한다.

## 깨달은 점
- "arcface가 명세서를 simswap에 전달하면 그걸 바탕으로 딥페이크를 하는거니까, arcface의 명세서를 노이즈로 방해하는 것이다." → 정확히는 노이즈는 명세서가 아니라 **사진**에 넣고, ArcFace가 그 사진을 보고 스스로 틀린 명세서를 쓰게 만든다.
- "우리가 simswap을 쓰는 이유는 단지 테스트 때문이다.


## 참고한 코드와 자료
- SimSwap: https://github.com/neuralchen/SimSwap (CC BY-NC 4.0, 학술·비상업 용도) — 공식 Colab 노트북을 바탕으로 실행, 위 표의 수정 사항만 직접 반영
- insightface antelopev2: https://github.com/deepinsight/insightface/releases/tag/v0.7
- FaceShield 논문 요약: `docs/papers/FaceShield_summary.md`
