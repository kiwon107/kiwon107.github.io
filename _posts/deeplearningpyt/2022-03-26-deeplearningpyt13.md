---
layout: single
title: "[Deeplearning(pytorch)] 13. 딥러닝을 해킹하는 적대적 공격"
folder: "deeplearningpyt"
categories:
    - deeplearningpyt
tag:
    - [deep learning, pytorch, rnn]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "펭귄브로의 3분 딥러닝, 파이토치맛" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 13-1. 적대적 공격이란?
머신러닝 모델의 착시를 유도하는 입력을 **적대적 예제(Adversarial example)** 라고 한다. 적대적 예제를 생성해서 여러 가지 머신러닝 기반 시스템의 성능을 의도적으로 떨어뜨려 보안 문제를 일으키는 것을 **적대적 공격(Adversarial attack)** 이라고 한다.<br/>

최근 구글 브레인에서 진행한 연구에 따르면, 모든 머신러닝 분류기가 속임수에 의해 잘못된 예측을 할 수 있다고 한다. 자율주행, 은행의 비정상거래탐지, 의료영상분석 등 실수가 용납되지 않는 분야의 시스템에 신뢰도를 떨어뜨리는 약점이 존재할 수 있다는 것은 치명적이다. 딥러닝 모델 내부를 해석할 수 있다면 이러한 적대적 공격에 대해 보안 기능을 마련할 수 있을 것이다. 그러나 아직 이러한 기술은 초기 단계에 있어 최근 연구에서도 효과적인 방어법을 내놓지 못하고 있다.<br/>

입력 데이터가 신경망 모델을 타고 흐르면서 모델 내 가중치와 편향 값에 계속 변환이 일어난다. 각 변환은 입력의 특정 구조에 매우 예민하게 반응한다. 이처럼 모델이 예민하게 반응하는 부분을 공략하여 모델을 헷갈리게 할 수 있다.<br/>

## 13-2. 적대적 공격의 종류
적대적 공격은 적절한 잡음을 생성하여 사람의 눈에는 똑같이 보이지만 머신러닝 모델을 헷갈리게 만드는 적대적 예제를 생성하는 것이 핵심이다. 인식 오류를 일으키지만 원본과 차이가 가장 적은 잡음을 찾는 것이고, 결국 최적화 문제로 해석할 수 있다. 적대적 공격에선 오차를 줄이기보다는 극대화하는 쪽으로 잡음을 최적화한다.<br/>

잡음을 생성하는 방법은 많다. 다만 모델 정보가 필요한지, 우리가 원하는 정답으로 유도할 수 있는지, 여러 모델을 동시에 헷갈리게 할 수 있는지, 학습이 필요한지 등의 여부에 따라 종류가 나뉜다. 극단적으로 이미지 픽셀 하나만 건드려 분류기의 예측을 완전히 빗나가게 할 수도 있다.<br/>

적대적 예제에서 잡음의 생성 방법은 분류 기준이 무엇이냐에 따라 여러가지로 나뉜다.<br/>
1) 기울기와 같은 모델 정보가 필요한지에 따라, 모델 정보를 토대로 잡음을 생성하는 **화이트박스(White box)** 방법과 모델 정보 없이 생성하는 **블랙박스(Black box)**로 나뉜다.<br/>
2) 원하는 정답으로 유도할 수 있다면 **표적(Targeted)**, 아니라면 **비표적(Non-targeted)**으로 분류한다.<br/>
3) 잡음을 생성하기 위해 반복된 학습(최적화)이 필요하면 **반복(Iterative)**, 아니면 **원샷(One-shot)**으로 나눌 수 있다.<br/>
4) 한 잡음이 특정 입력에만 적용되는지, 모든 이미지에 적용될 수 있는 범용적인 잡음인지로 나눌 수 있다.<br/>

가장 강력한 공격 방법은 모델 정보가 필요 없고, 원하는 정답으로 유도할 수 있으며, 복잡한 학습이 필요하지 않고, 여러 모델에 동시 적용할 수 있는 방법이다. 그러나 각 특징에는 기회비용이 존재한다.

## 13-3. FGSM 공격
**FGSM(Fast gradient sign method)** 란 방법으로 적대적 예제를 생성해, 미리 학습된 딥러닝 모델을 공격해보자. FGSM은 반복된 학습 없이 잡음을 생성하는 원샷 공격이다. 입력 이미지에 대한 기울기의 정보를 추출하여 잡음을 생성한다. 그림 13-1 처럼 잡음이 눈에 보이지 않아야 하므로, 아주 작은 숫자를 잡음에 곱해서 희석한 후 원본 그림에 더한다. 최적화를 통해 더 정교한 잡음을 만들 수도 있다. FGSM은 공격 목표를 정할 수 없는 Non-targeted 방식이자, 대상 모델의 정보가 필요한 화이트박스 방식이다.
![그림 13-1. FGSM 예시](/assets/images/deeplearningpyt/13-1.png)
{: .align-center}
그림 13-1. FGSM 예시

이제 공격이 진행되는 방식을 단계별로 보자.
```python
# 관련 모듈 임포트
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import matplotlib.pyplot as plt
```

토치비전은 AlexNet, VGG, ResNet, SqueezeNet, DenseNet, Inception 등 여러 가지 학습된 모델을 제공한다. 대부분 이미지넷 데이터셋으로 학습된 모델이다. `models.<모델명>` 함수를 호출할 때 인수로 `pretrained=True`를 명시하면 학습이 완료된 모델을 사용할 수 있다. 이번 예제에서는 ResNet101을 사용해보자. 정확도를 더 끌어올리고 싶다면 DenseNet이나 Inception v3 같은 모델을 사용하고, 노트북처럼 컴퓨팅 성능이 떨어지는 환경이면 SqueezeNet 같이 가벼운 모델을 사용하자.
```python
# 학습된 모델 불러오기
model = models.resnet101(pretrained=True)
model.eval()
print(model)
```

위 모델의 예측값은 0~1000까지의 숫자를 뱉는다. 이 값은 이미지넷 데이터셋의 클래스를 가리키는 번호다. `imagenet_classes.json` 파일에 숫자와 클래스 제목의 매칭 정보가 담겨 있다. 이 파일을 딕셔너리로 만들어 번호를 레이블 이름으로 변환해주는 idx2class 리스트를 만들고, 이 매칭 정보를 언제든 이용할 수 있도록 하자.
```python
# 데이터셋 불러오기
CLASSES = json.load(open('./imagenet_samples/imagenet_classes.json'))
idx2class = [CLASSES[str(i)] for i in range(1000)]
```

이제 공격하고자 하는 이미지를 불러오자. 실제 공격은 학습용 데이터에 존재하지 않는 이미지로 가해진다. 우리도 데이터셋에 존재하지 않는 이미지를 새로 준비하자.
```python
# 이미지 불러오기
img = Image.open('imagenet_samples/corgie.jpg')

# 이미지를 이미지넷과 같은 크기로 변환후 텐서로 변환
img_transforms = transforms.Compose([
  transforms.Resize((224, 224), Image.BICUBIC),
  transforms.ToTensor(),
])
img_tensor = img_transforms(img)
img_tensor = img_tensor.unsqueeze(0) # 3x224x224 -> 1x3x224x224
print("이미지 텐서 모양: ", img_tensor.size())
```
    (결과) 이미지 텐서 모양:  torch.Size([1, 3, 224, 224])

원본 이미지 텐서를 시각화하기 위해 `squeeze()` 함수로 차원을 줄이고, `detach()` 함수로 원본 이미지 텐서와의 연결을 끊자. `transpose()` 함수로 뒤집힌 이미지를 제자리로 돌려놓고 맷플롯립과 호환되는 넘파이 행렬로 변환하자.
```python
# 시각화 위해 넘파이 행렬 변환
# 1x3x224x224 -> 3x224x224
original_img_view = img_tensor.squeeze(0).detach()
original_img_view = original_img_view.transpose(0, 2).transpose(0, 1).numpy() # transpose는 dim0과 dim1 파라미터 차원을 서로 바꿈
plt.imshow(original_img_view)
```
![그림 13-2. 코드 결과](/assets/images/deeplearningpyt/13-2.png)
{: .align-center}
그림 13-2. 코드 결과

공격을 하기 전, 준비한 학습용 데이터에 없는 이미지를 얼마나 잘 분류하는지 확인하기 위해 앞서 준비한 모델에 이미지를 통과시켜보자. `max()` 함수를 이용하여 확률이 높은 인덱스를 가져오자.
```python
output = model(img_tensor)
prediction = output.max(1, keepdim=False)[1] # keepdim을 True로 하면 output의 차원인 (1,1000)의 2개 차원을 유지시킨 (1, 1)로 값 출력됨. False면 값 하나만 나오고 shape은 (1)로 나옴. # 맨 끝 [0]은 값, [1]은 인덱스 값 출력. 첫번째 인자 0이면 열 기준, 1이면 행 기준으로 max값 찾음.

prediction_idx = prediction.item()
prediction_name = idx2class[prediction_idx]

print("예측된 레이블 번호: ", prediction_idx)
print("레이블 이름: ", prediction_name)
```
    (결과) 예측된 레이블 번호:  263
           레이블 이름:  Pembroke, Pembroke Welsh corgi

웰시코기로 잘 분류가 됐다!

FGSM 공격은 모델에서 입력 이미지에 대한 기울기 정보를 추출하고, 그것을 왜곡하여 원본 이미지에 더하여 이뤄진다. 기울기는 모델이 학습할 때 각 픽셀이 미치는 영향이라고 이해할 수 있다. 원본 이미지를 받아 적대적 예제를 생성하는 fgsm_attack() 함수를 정의해보자.

모델을 헷갈리게 하려면, 모델의 오차값을 극대화해야 한다. 띠리사 FGSM 에서는 딥러닝 모델 학습시 잡음이 기울기 방향으로 최적화하도록 하여 오차를 키운다. `sign()` 함수를 적용하여 기울기의 방향성을 알 수 있도록 하자. `sign()` 함수는 입력이 0보다 작으면 -1, 0보다 크면 1을 출력하는 단순한 함수이다. 그리고 작은 값인 엡실론을 곱해준다. 엡실론은 모델 학습 시 지정해줘야 하는 학습률과 같은 개념이다. 잡음이 너무 커지지 않고 사람의 눈에 보이지 않도록 제한하는 역할이다. 즉, 기울기 방향이 양수이면 엡실론만큼 값을 증가시키고, 음수인 곳은 엡실론만큼 값을 감소시킨다.
```python
def fgsm_attack(image, epsilon, gradient):
  # 기울기값의 원소 sign 값을 구함
  sign_gradient = gradient.sign()

  # 이미지의 각 픽셀값을 sign_gradient 방향으로 epsilon 만큼 조절
  perturbed_image = image + epsilon * sign_gradient

  # [0, 1] 범위를 벗어나는 값 조절
  perturbed_image = torch.clamp(perturbed_image, 0, 1) # 최소값, 최대값 범위 넘으면 지정한 최소값, 최대값으로 변환
  return perturbed_image
```

원본 이미지에 대한 기울기를 추출하려면 `requires_grad_(True)` 함수를 호출해 이미지에 대한 기울기를 보존하도록 해야한다! 그렇지 않으면 기울기가 생성되지 않는다. 평범하게 학습하듯 오차값과 기울기값을 구한다. 역전파를 마치면 img_tensor의 grad.data 변수에 입력 이미지의 기울기가 저장된다. 이 기울기값을 가지고 다음 앞서 정의한 fgsm_attack() 함수를 호출한다.
```python
# 이미지의 기울기를 구하도록 설정
img_tensor.requires_grad_(True)

# 이미지를 모델에 통과
output = model(img_tensor)

# 오차값 구하기
loss = F.nll_loss(output, torch.tensor([263]))

# 기울기 구하기
model.zero_grad()
loss.backward()

# 이미지 기울기 추출
gradient = img_tensor.grad.data

# FGSM 공격으로 적대적 예제 생성
epsilon = 0.03
perturbed_data = fgsm_attack(img_tensor, epsilon, gradient)

# 생성된 적대적 예쩨를 모델에 통과
output = model(perturbed_data)
```

이제 새로 만든 적대적 예제 이미지를 다시 똑같은 모델로 분류해보자.
```python
# 적대적 예제 성능 확인
perturbed_prediction = output.max(1, keepdim=True)[1]
perturbed_prediction_idx = perturbed_prediction.item() # 텐서를 원래 자료형으로 변환
perturbed_prediction_name = idx2class[perturbed_prediction_idx]

print("예측된 레이블 번호: ", perturbed_prediction_idx)
print("레이블 이름: ", perturbed_prediction_name)
```
    (결과) 예측된 레이블 번호:  172
           레이블 이름:  whippet

같은 모델임에도 생성된 적대적 예제를 휘핏으로 예측하였다.

딥러닝 모델을 헷갈리게 만든 적대적 예제를 시각화해보자.
```python
# 시각화를 위해 넘파이 행렬 변환
perturbed_data_view = perturbed_data.squeeze(0).detach()
perturbed_data_view = perturbed_data_view.transpose(0, 2).transpose(0, 1).numpy()
plt.imshow(perturbed_data_view)
```
![그림 13-3. 코드 결과](/assets/images/deeplearningpyt/13-3.png)
{: .align-center}
그림 13-3. 코드 결과

이미지 주변에 이상한 무늬가 생긴게 전부이다. 원본과 적대적 예제를 나란히 시각화해보자.
```python
f, a = plt.subplots(1, 2, figsize=(10, 10))

# 원본
a[0].set_title(prediction_name)
a[0].imshow(original_img_view)

# 적대적 예제
a[1].set_title(perturbed_prediction_name)
a[1].imshow(perturbed_data_view)
```
![그림 13-4. 코드 결과](/assets/images/deeplearningpyt/13-4.png)
{: .align-center}
그림 13-4. 코드 결과

적대적 예제를 생성하는 각종 방법에 대한 연구는 현재도 진행되고 있지만, 이것을 방어하는 연구 또한 활발히 진행되고 있다.