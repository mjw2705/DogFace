# Dog Face Recognition

## Description
개 얼굴 인식 알고리즘 개발

## 모델 구축
### 1. Dataset
DogFaceNet의 dataset 사용
- id: 1393개 + class 2개 추가 ⇒ 1395개
- 총 이미지 : 8363개 (한 class 당 이미지 최소 2개) + 25개 이미지 추가 ⇒ 8388개

dlib의 dogHeadDetector를 사용하여 개 얼굴 cropping
