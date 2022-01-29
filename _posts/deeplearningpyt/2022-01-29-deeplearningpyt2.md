---
layout: single
title: "[Deeplearning(pytorch)] 2. 경사하강법으로 이미지 복원하기"
folder: "deeplearningpyt"
categories:
    - deeplearningpyt
tag:
    - [deep learning, pytorch, gradient descent, example, practice]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "펭귄브로의 3분 딥러닝, 파이토치맛" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 2-1. 오염된 이미지 문제와 복원 방법
오염된 이미지와 이미지 처리 함수 `weird_function()`을 이용하여 원본 이미지를 복원하는 문제이다.

다음과 같은 사고 과정을 거쳐보자!
1. 오염된 이미지와 같은 크기의 랜덤 텐서 생성
2. 랜덤 텐서를 `weird_function()` 함수에 입력하여 똑같이 오염된 이미지 생성. 이때 인위적으로 생성한 복사본 이미지를 가설이라고 함!
3. 가설과 오염된 이미지가 같다면, 무작위 이미지와 오염되기 전 원본 이미지도 같을 것!
4. 이에따라, `weird_function(random_tensor) = broken_image` 관계 성립되도록 만듬.

위 사고 과정을 실체화에 성공한다면, random_tensor는 오염되기 전 원본이미지와 거의 비슷한 형태가 될 것이다.<br/>

이를 구현하기 위해, 우리는 가설인 random_tensor와 오염되기 전 원본 이미지(`weird_function()` 들어가기 전) 사이의 거리 값을 오차로 두어, 이 오차값이 최솟값이 되도록 랜덤 텐서를 바꿔주어야 한다. 랜덤 텐서를 바꿔주는 것은 경사하강법 알고리즘을 사용한다.  `Autograd` 패키지를 이용하여 오차를 출력하는 함수의 기울기를 구하고, 이 기울기의 반대 방향으로 가면 오차값이 줄어든다. 이것을 계속 반복하여, 오차값이 최소가 되었을 때의 `random_tensor`값을 보면 오염되기 전 원본 이미지와 거의 비슷한 형태가 될 것이다.

## 2-2. 문제 해결과 코드 구현
파이토치, 맷플롯립을 임포트 한다. 오염된 이미지 파일 로딩하는데 사용할 피클 라이브러리도 임포트 한다. 피클은 파이썬 객체를 파이썬 형태로 저장할 때 쓰는 패키지로, 파이썬에서 기본적으로 제공한다. 오염된 이미지를 파이썬 텐서의 형태로 읽고 이들을 시각화해보자.

```python
import torch
import pickle
import matplotlib.pyplot as plt

broken_image = torch.FloatTensor( pickle.load(open('./broken_image_t_p', 'rb'), encoding='latin1'))
plt.imshow(broken_image.view(100, 100))
```
![그림 2-1. 코드 결과](/assets/images/deeplearningpyt/2-1.png)
{: .align-center}
그림 2-1. 코드 결과

`broken_image`는 이미지 행렬을 랭크 1의 벡터로 표현한 텐서 데이터이다. 10,000개의 원소를 [100, 100] 모양의 행렬이 되도록 변환시켜 이를 시각화 하였다.

이제 `weird_function()` 함수를 만들자. 저자는 함수를 아직 이해할 필요는 없다고하니, 가볍게 보고 넘어가보자.

```python
def weird_function(x, n_iter=5):
    h = x
    filt = torch.tensor([-1./3, 1./3, -1./3]) # 필터! 무슨 필터인지는 아직 모르겠다.
    for i in range(n_iter):
        zero_tensor = torch.tensor([1.0*0])
        h_l = torch.cat((zero_tensor, h[:-1]), 0)  # zero_tensor와 h[:-1]을 concatenate 한다
        h_r = torch.cat((h[1:], zero_tensor), 0)   # h[1:]와 zero_tensor를 concatenate 한다
        h = filt[0] * h + filt[2] * h_l + filt[1] * h_r
        if i % 2 == 0:
            h = torch.cat((h[h.shape[0]//2:], h[:h.shape[0]//2]), 0)
    
    return h
```

그 다음 무작위 텐서를 `werid_tensor()`에 입력하여 얻은 가설 텐서와 오염된 원본 이미지 간 오차를 구하는 함수를 만들 것이다.
```python
def distance_loss(hypothesis, broken_image):
  return torch.dist(hypothesis, broken_image)  # 두 텐서 사이의 거리 구하는 함수
```

이제 무작위 값 갖는 텐서 생성하고 경사하강법에 사용할 학습률을 설정해보자. 학습률은 경사하강법이 여러 번 반복될 때, 1회 반복에서 최솟점으로 얼마나 이동할지, 즉 학습을 얼마나 급하게 진행할 것인지 정하는 매개변수이다.
```python
random_tensor = torch.randn(10000, dtype = torch.float)
lr = 0.8
```

이제 경사하강법의 for 반복문을 구현해보자. 먼저 random_tensor가 미분 가능하도록 설정하고, 무작위 텐서를 `weird_function()` 함수에 통과시켜 가설을 구한다. 그 다음 가설과 오염된 원본 이미지의 오차를 계산하고 오차 함수를 random_tensor에 대해 미분한다. 마지막으로 직접 경사하강법을 구현할 것이기 때문에 파이토치의 자동 기울기 계산을 비활성화하고, `loss.backward()`에서 구한 loss의 기울기 방향의 반대쪽으로 random_tensor를 학습률만큼 이동시킨다. for문이 1,000번 반복될 때마다 오차를 출력하도록 할 것이다.
```python
for i in range(0, 20000):
  random_tensor.requires_grad_(True)
  hypothesis = weird_function(random_tensor)
  loss = distance_loss(hypothesis, broken_image)
  loss.backward()

  with torch.no_grad():
    random_tensor = random_tensor - lr * random_tensor.grad
  
  if i % 1000 == 0:
    print('loss at {} = {}'.format(i, loss.item()))
```

반복문이 다 돌았다면 random_tensor가 어떻게 바뀌었는지 확인해보자.
```python
plt.imshow(random_tensor.view(100, 100))
```
![그림 2-2. 코드 결과](/assets/images/deeplearningpyt/2-2.png)
{: .align-center}
그림 2-2. 코드 결과

원본 이미지 타임스퀘어 풍경이 잘 만들어졌다!