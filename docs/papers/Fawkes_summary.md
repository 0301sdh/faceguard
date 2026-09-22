# [논문 리뷰] Fawkes: Protecting Privacy against Unauthorized Deep Learning Models

## 1. 논문 정보
- **저자:** Shawn Shan, Emily Wenger, Jiayun Zhang, Huiying Li, Haitao Zheng, Ben Y. Zhao
- **소속:** 시카고 대학교 (University of Chicago, Computer Science)
- **발표 연도:** 2020
- **링크:** https://people.cs.uchicago.edu/~ravenben/publications/pdf/fawkes-usenix20.pdf

## 2. 논문의 핵심 아이디어
- **Image Cloaking:** 사용자 사진에 눈에 띄지 않는 미세한 픽셀 변화(Cloak)를 추가하여 얼굴 인식 모델을 속임.
- **특징 공간(Feature Space) 왜곡:** 모델의 특징 추출기가 원본 사용자가 아닌 완전히 다른 사람(타겟)의 특징으로 인식하도록 유도함.
- **전이성(Transferability):** 사용자와 추적자(불법 AI)가 서로 다른 모델을 사용하더라도 보호 효과가 상당 부분 유지됨.

## 3. FaceGuard 프로젝트에 참고할 점
- **Face Embedding 공격 방향:** 우리 프로젝트에서도 단순히 픽셀을 망가뜨리는 것이 아니라, AI의 특징 추출기(Feature Extractor)를 거쳐 나온 'Face Embedding'이 원본과 달라지도록 조작하는 것을 핵심 목표로 삼는다.
- **평가 방식:** 원본과 노이즈가 추가된 이미지의 Embedding Similarity를 비교하는 방식을 우리 프로젝트의 주요 정량 지표로 활용할 수 있다.

## 4. 논문의 한계 및 우리 프로젝트와의 차별점
- **차별점:** Fawkes는 주로 데이터 포이즈닝(학습 방해)에 초점을 맞추었으나, 우리 프로젝트는 이 원리를 응용한 모델을 웹 서비스 형태로(Spring Boot + FastAPI) 구현하여 전체 라이프사이클을 완성하는 것에 집중한다.