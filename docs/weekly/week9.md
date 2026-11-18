# Week 9 — AI 서버 구축 (FastAPI, Docker)

> **결론:** Week 8 방식 C'를 `server/protect.py`로 옮기고, FastAPI 서버(`POST /protect`)로 감쌌다. 내 Mac에서 사진을 보내면 보호된 PNG가 돌아온다(CPU 약 37초). Docker 이미지로도 같은 결과가 나온다. 결과는 노트북과 사실상 같다(52 ~ 64dB, 입력 신원 차이 0.02 이내).

## 목표
노트북 셀에 흩어진 AI 코드를 실제 API 형태로 분리한다. 로컬에서 HTTP 요청을 보내면 AI 서버가 사진을 처리해 결과를 돌려준다.

## 핵심 개념
- **서버:** 켜 두면 요청을 기다렸다가 처리하고 응답하는 프로그램. 나만 쓰면 내 Mac에서 켜면 되고, 다른 사람도 쓰려면 클라우드 컴퓨터를 빌려 켜 둔다(배포).
- **API:** 서버가 받는 요청의 약속. `POST /protect`에 사진을 보내면 보호된 사진을 준다.
- **역할 나누기:** AI 서버(FastAPI, Python)는 노이즈만 만든다. 업로드 검증, 작업 관리, 저장, 조회는 Week 10의 Spring Boot가 맡는다. AI 코드가 Python에서만 돌고, 비싼 GPU를 AI 계산에만 쓰기 위해서다.
- **.onnx:** 어느 환경에서나 불러 쓸 수 있는 모델 파일 형식. gradient를 주지 않으므로 서버가 불러올 때 onnx2torch로 PyTorch 모델로 바꾼다.
- **Docker:** 운영체제, 패키지, 코드를 하나의 이미지로 묶어 어느 컴퓨터에서나 똑같이 실행되게 하는 도구.

## 구성
| 파일 | 내용 |
|---|---|
| `server/protect.py` | 방식 C' 보호 코드. `FaceGuard().protect(사진)` → 보호된 사진. 실험용 코드(SimSwap 합성, FaceID 생성, 채점)는 뺐다 |
| `server/app.py` | FastAPI 서버. 켤 때 모델을 한 번 불러 두고, 한 번에 한 장씩 처리 |
| `server/try_local.py` | 서버 없이 보호 코드만 실행해 시간 재기 |
| `notebooks/server_check.ipynb` | Colab에서 모델 파일 3개 만들기 + 노트북 결과와 비교 |
| `Dockerfile`, `.dockerignore` | CPU용 서버 이미지. 모델 파일은 이미지에 넣지 않고 실행할 때 연결 |

- 모델 파일(`models/`, git 제외): `scrfd_10g_bnkps.onnx`(얼굴 찾기), `simswap_arcface.onnx`(SimSwap ArcFace를 Colab에서 변환), `w600k_r50.onnx`(FaceID ArcFace)
- SimSwap 코드 없이 돌아가도록 SimSwap의 정렬 기준 위치(5가지 자세)를 원본 코드에서 그대로 옮겼다.

### API
| 요청 | 응답 |
|---|---|
| `POST /protect` (사진 파일) | 200, 보호된 PNG. 헤더에 결과 크기, 처리 시간 |
| 사진이 아닌 파일 | 400 "사진 파일이 아닙니다" |
| 20MB 초과 | 413 "파일이 너무 큽니다" |
| 얼굴 없음 | 422 "얼굴을 찾지 못했습니다" |
| `GET /health` | 서버와 모델 준비 상태 |

출력은 PNG(무손실)로 한다. JPEG로 다시 저장하면 노이즈가 일부 지워진다.


## 결과
### 1. 옮긴 코드가 노트북과 같은가 (Colab T4)
| 사진 | 노트북 결과와 같은 정도 | 입력 SimSwap ArcFace (새 / 노트북) | 입력 FaceID ArcFace (새 / 노트북) |
|---|---|---|---|
| a_1 | 64.2dB | -0.175 / -0.173 | -0.003 / -0.006 |
| a_2 | 52.3dB | -0.029 / -0.031 | -0.039 / -0.039 |
| a_3 | 54.7dB | 0.049 / 0.066 | 0.197 / 0.204 |
| a_4 | 52.6dB | 0.046 / 0.040 | 0.226 / 0.227 |

- 사실상 같은 사진이다. 차이는 GPU 계산의 미세한 비결정성과 insightface 버전 차이로 본다.

### 2. 처리 시간 (사진 1장, 방식 C')
| 환경 | 시간 |
|---|---|
| Colab T4 GPU | 약 16 ~ 17초 (노트북 13초보다 조금 느림, onnx 변환 영향으로 추정) |
| Mac CPU | 36 ~ 37초 |
| Mac GPU (mps) | 56초 (CPU보다 느림 → 쓰지 않음) |
| Docker (Mac 안 리눅스, CPU) | 86초 |

- 시간 대부분은 노이즈 계산이다. 얼굴 찾기와 붙이기는 합쳐 1초 안팎이다.

### 3. 서버 시험
- Mac 서버: 보호 확인(입력 신원) SimSwap 0.970 → -0.175, FaceID 1.000 → -0.042. `try_local.py` 결과와 픽셀까지 같음
- Docker 서버: SimSwap -0.180, FaceID -0.040. Mac 서버 결과와 63.8dB(사실상 같음)
- 오류 요청 3가지 모두 정해 둔 응답이 나옴

## 한계
- 사진 1장에 CPU로 37 ~ 86초가 걸린다. 요청이 그동안 기다려야 한다(Week 11 비동기 처리에서 다룸).
- 한 번에 한 장씩만 처리한다.
- Docker 이미지는 CPU용이다. 배포에는 GPU용 이미지가 필요하다(Week 12).
- Mac GPU, Docker가 느린 이유는 추정이다. 확인하지 않았다.
- SimSwap, insightface 모델은 비상업·연구용이다. 포트폴리오 용도로만 쓴다.

## 깨달은 점
- 연구 코드를 서비스로 옮길 때는 먼저 "같은 결과가 나오는가"를 확인해야 한다. 옮기다 생긴 실수는 숫자로만 드러난다.
- 같은 코드도 실행 환경에 따라 속도가 크게 다르다. GPU가 있다고 항상 빠르지 않고(Mac mps), 직접 재 봐야 한다.
- 패키지는 몰래 다른 패키지를 함께 설치한다. Docker로 깨끗한 환경에서 실행해 보면 이런 문제가 드러난다.

## 자료
- 숫자 전체: `results/week9/` (`colab_server_check.txt`, `local_speed.txt`, `api_test.txt`)
- FastAPI: https://fastapi.tiangolo.com / Docker: https://docs.docker.com
- SimSwap: https://github.com/neuralchen/SimSwap (CC BY-NC 4.0) / insightface: https://github.com/deepinsight/insightface (비상업 연구 용도)
