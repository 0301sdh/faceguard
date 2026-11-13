# results

실험 결과 숫자 원본. 주차 문서(`docs/weekly/`)에는 결론에 필요한 표만 두고, 사진별 전체 숫자는 여기에 둔다.
숫자만 있고 얼굴 사진은 없다 (사진과 합성 결과는 `data/`, git 제외).

| 파일 | 내용 | 만든 코드 |
|---|---|---|
| `week5/facenet_baseline.txt` | 보호 안 한 SimSwap 결과의 FaceNet 유사도 | `experiments/week5_facenet_baseline.py` |
| `week5/facenet_pilot.txt` | pilot(eps 4/8/12) SimSwap 결과의 FaceNet 유사도, PSNR | `experiments/week5_facenet_pilot.py` |
| `week6/facenet_step2.txt` | ② 이동만 vs 회전·크기 EOT (eps 8, 12) | `experiments/week6_facenet_eval.py` |
| `week6/facenet_step3.txt` | ③ eps 4/8/12 × steps 30/50 | `experiments/week6_facenet_eval.py` |
| `week6/facenet_step3_2.txt` | ③-2 eps 4(n4, n8), 5, 6 한 번 실행 | `experiments/week6_facenet_eval.py` |
| `week6/facenet_step4.txt` | ④ seed 3개 반복 (평균 ± 표준편차) | `experiments/week6_facenet_eval.py` |
| `week6/facenet_step5.txt` | ⑤ 변형 11가지 요약 | `experiments/week6_facenet_robust.py` |
| `week6/robust_facenet.csv` | ⑤ 사진·seed·변형별 FaceNet 점수 | `experiments/week6_facenet_robust.py` |
| `week7/facenet_ensemble.txt` | FaceID baseline, pilot(Week 6 보호 사진 eps 5), ensemble 보호 사진의 FaceNet 유사도 | `experiments/week7_facenet_eval.py` |
| `week7/facenet_simswap_ensemble.txt` | SimSwap 결과의 FaceNet 유사도: Week 6 보호(eps5_n4) vs ensemble, 노이즈 seed 3개 | `experiments/week6_facenet_eval.py` |
| `week7/facenet_calibration_lfw.txt` | FaceNet 눈금 다시 잡기: LFW 타인 500명 vs 내 사진 4장 유사도 분포 | `experiments/week7_facenet_calibration.py` |
| `week7/facenet_stranger_gen.txt` | FaceID가 LFW 타인 24명을 그린 이미지 vs 내 사진 (FaceNet) + baseline·pilot·ensemble을 같은 잣대(내 사진 4장 평균)로 | `experiments/week7_facenet_stranger_gen.py` |
| `week7/arcface_colab.txt` | FaceID 눈금 잡기, baseline, pilot, ensemble 준비·공격·생성의 insightface ArcFace 점수 | `notebooks/ipadapter_faceid_eval.ipynb` 셀 출력을 옮김 |
| `week8/colab_full_photo.txt` | 사진 전체 보호 방식 A·B·C·C': SimSwap·FaceID ArcFace 점수, 처리 시간, PSNR | `notebooks/simswap_robustness.ipynb`, `ipadapter_faceid_eval.ipynb` 셀 출력을 옮김 |
| `week8/facenet_full_photo.txt` | 사진 전체 보호 방식 A·B·C·C'로 FaceID가 생성한 이미지의 FaceNet 유사도 | `experiments/week7_facenet_eval.py` |

- FaceNet 채점에는 무작위 요소가 없어서, 같은 합성 결과를 다시 채점하면 같은 숫자가 나온다.
- ArcFace 점수는 Colab 노트북(`notebooks/simswap_eval.ipynb`, `notebooks/simswap_robustness.ipynb`)의 셀 출력에 남아 있다.
