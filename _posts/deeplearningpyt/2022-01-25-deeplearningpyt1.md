---
layout: single
title: "[Deeplearning(pytorch)] 1. 텐서와 Autograd"
folder: "deeplearningpyt"
categories:
    - deeplearningpyt
tag:
    - [deep learning, pytorch, tensor, autograd]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "펭귄브로의 3분 딥러닝, 파이토치맛" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 1-1. 텐서의 차원 자유자재로 다루기
파이토치 임포트
```python
import torch
```

- 텐서: 파이토치에서 다양한 수식 계산하는데 사용하는 가장 기본적인 자료구조로써 숫자들을 특정한 모양으로 배열한 것
- 텐서는 '차원' 또는 '랭크' 라는 개념 가짐
  * 랭크0 텐서: 스칼라, Ex) 1, 모양은 `[]`
  * 랭크1 텐서: 벡터, Ex) [1, 2, 3], 모양은 `[3]`
  * 랭크2 텐서: 행렬, Ex) [[1, 2, 3]], 모양은 `[1, 3]`
  * 랭크3 텐서: 3차원 행렬, Ex) [[[1, 2, 3]]], 모양은 `[1, 1, 3]`

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)
print('Size: ', x.size())  # 텐서의 구체적 모양
print('Shape: ', x.shape)  # 텐서의 구체적 모양
print('랭크(차원): ', x.ndimension())
print('----------------------------')

x = torch.unsqueeze(x, 0)  # 랭크 2 텐서의 첫번째 자리에 1이라는 차원값 추가하여 [1, 3, 3] 모양의 랭크 3 텐서로 변경
print(x)
print('Size: ', x.size()) 
print('Shape: ', x.shape)  
print('랭크(차원): ', x.ndimension())
print('----------------------------')

x = torch.squeeze(x)  # 텐서의 랭크 중, 크기가 1인 랭크를 삭제하여 다시 랭크 2 텐서로 되돌림
print(x)
print('Size: ', x.size()) 
print('Shape: ', x.shape)  
print('랭크(차원): ', x.ndimension())
print('----------------------------')

x = x.view(9)  # 랭크 2의 [3, 3] 모양인 x를 랭크 1의 [9] 모양으로 바꿈. 텐서의 원소 개수는 바꿀 수 없으므로, 잘못 입력시 에러 발생됨(Ex. 8로 변경 불가능).
print(x)
print('Size: ', x.size()) 
print('Shape: ', x.shape)  
print('랭크(차원): ', x.ndimension())
```
    (결과) tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
           Size:  torch.Size([3, 3])
           Shape:  torch.Size([3, 3])
           랭크(차원):  2
           ----------------------------
           tensor([[[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]]])
           Size:  torch.Size([1, 3, 3])
           Shape:  torch.Size([1, 3, 3])
           랭크(차원):  3
           ----------------------------
           tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
           Size:  torch.Size([3, 3])
           Shape:  torch.Size([3, 3])
           랭크(차원):  2
           ----------------------------
           tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
           Size:  torch.Size([9])
           Shape:  torch.Size([9])
           랭크(차원):  1

## 1-2. 텐서를 이용한 연산과 행렬곱
- 행렬: 랭크 2인 텐서와 같은 개념. 숫자들을 네모꼴로 배치한 2차원 배열.
- 행렬의 높이를 '행', 너비를 '열' 이라 함
- A, B라는 두 행렬로 행렬곱 하려면 다음 조건 성립해야함
  * A의 열 수와 B의 행 수는 같아야 함
  * 행렬곱 A*B의 결과 행렬의 행 개수는 A와 같고, 열의 개수는 B와 같음

```python
import torch

w = torch.randn(5, 3, dtype=torch.float) # 정규분포에서 무작위로 실수값 뽑아 텐서 생성하는 randn함수로 5x3 shape의 텐서 생성
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) # 실수형 원소 값는 3x3 shape의 텐서
print("w size: ", w.size())
print("x size: ", x.size())
print("w:", w)
print("x:", x)
print('----------------------------')

b = torch.randn(5, 2, dtype=torch.float)
print("b size:", b.size())
print("b:", b)
print('----------------------------')

wx = torch.mm(w, x)
print("wx size:", wx.size())
print("wx:", wx)
print('----------------------------')

result = wx + b
print("result size:", result.size())
print("result:", result)
```
    (결과) w size:  torch.Size([5, 3])
           x size:  torch.Size([3, 2])
           w: tensor([[ 0.6106,  0.0583, -0.6274],
                   [ 0.0542,  0.3214,  0.2737],
                   [-1.4459,  0.4687,  1.3794],
                   [ 0.4403, -0.2277, -0.1737],
                   [-1.3437, -0.5235, -1.4958]])
           x: tensor([[1., 2.],
                   [3., 4.],
                   [5., 6.]])
           ----------------------------
           b size: torch.Size([5, 2])
           b: tensor([[-2.2417, -1.6182],
                   [ 0.6943,  0.4118],
                   [ 0.9479,  0.6748],
                   [ 0.9508, -0.8875],
                   [-0.3306,  0.2391]])
           ----------------------------
           wx size: torch.Size([5, 2])
           wx: tensor([[ -2.3514,  -2.3099],
                   [  2.3866,   3.0358],
                   [  6.8571,   7.2593],
                   [ -1.1116,  -1.0729],
                   [-10.3932, -13.7562]])
           ----------------------------
           result size: torch.Size([5, 2])
           result: tensor([[ -4.5931,  -3.9281],
                   [  3.0809,   3.4477],
                   [  7.8050,   7.9340],
                   [ -0.1609,  -1.9604],
                   [-10.7238, -13.5172]])

## 1-3. Autograd
직역하면 '자동 기울기'로, 수식의 기울기를 자동으로 계산한다는 뜻이 됨. 이건 왜 필요할까?<br/>
데이터에 대한 정답과 머신러닝 모델이 예측한 결과의 차이를 산출적으로 표현한 것을 **거리**라고 함. 그리고 학습 데이터로 계산한 거리들의 평균을 **오차**라고 부름.
오차가 작은 머신러닝 모델일수록 더 정확한 답을 낸다고 할 수 있음.<br/>
오차를 최소화 하는데 **경사하강법** 이라는 알고리즘을 주로 사용하며, 이는 오차를 수학함수로 표현하여 이를 미분해 함수의 기울기를 구하면, 이를 이용해 오차의 최소값이 있는 방향을 찾아내는 알고리즘임.<br/>
Autograd는 미분 계산을 자동화하여 경사하강법을 구현하는 수고를 덜어줌!

```python
w = torch.tensor(1.0, requires_grad=True)  # requires_grad를 True로 하면 파이토치의 autograd가 자동으로 계산할 때, w에 대한 미분값을 w.grad에 저장함
a = w * 3
l = a**2 # 9*(w^2) 과 동일
l.backward()
print('l을 w로 미분한 값은 {}'.format(w.grad)) #l에 backward() 함수 호출하여 w.grad가 w가 속한 수식을 w로 미분한 값 반환함.
```
    (결과) l을 w로 미분한 값은 18.0