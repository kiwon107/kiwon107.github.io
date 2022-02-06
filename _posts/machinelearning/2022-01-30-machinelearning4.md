---
layout: single
title: "[Machine learning] 4. k-최근접 이웃 회귀"
folder: "machinelearning"
categories:
    - machinelearning
tag:
    - [machine learning, regression, k-neighbor regressor, coefficient of determination, overfitting, underfitting]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 4-1. k-최근접 이웃 회귀
지도 학습 알고리즘은 크게 분류와 회귀로 나뉜다. 회귀는 임의의 어떤 숫자를 예측하는 문제이다. 예를 들면 배달이 도착할 시간 예측 같은 문제이다. 생선의 무게를 예측하는 것도 회귀 문제이다. k-최근접 이웃 알고리즘은 어떻게 회귀 문제에 적용될까?<br/>
분류 문제를 풀 때와 비슷하다. 예측하려는 샘플에 가장 가까운 샘플 k개를 선택한다. 그리고 이웃 수치들의 평균을 구한다. 이 평균값이 예측 타깃값이 된다.

## 4-2. 데이터 준비
농어의 길이와 무게에 대한 데이터를 준비하자.

```python
import numpy as np

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
```

이제 이 데이터를 산점도로 그려보자.
```python
import matplotlib.pyplot as plt

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
![그림 4-1. 코드 결과](/assets/images/machinelearning/4-1.png)
{: .align-center}
그림 4-1. 코드 결과

농어의 길이가 커질 수록 무게도 늘어남을 확인할 수 있다. 우리는 농어의 길이를 보고 무게를 예측하는 문제를 풀 것이다. 훈련 세트와 테스트 세트로 나누어보자.

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
```

`train_input`은 1개의 특성만 가지므로 1차원 배열이다. 하지만 훈련 세트는 `(데이터 개수, 특성 개수)` 형태를 가져야한다. 1차원 배열을 어떻게 2차원 배열로 바꿀수 있을까?
```python
test_array = np.array([1, 2, 3, 4])
print(test_array.shape)
test_array = test_array.reshape(2, 2)
print(test_array.shape)
```
    (결과) (4,)
           (2, 2)

이 처럼 `reshape()` 메소드를 활용하여 배열의 크기를 지정할 수 있다. 참고로 바꾸고자 하는 원소 개수만큼 크기를 변환해야 한다. 원소 개수가 맞지 않으면 `ValueError`가 발생한다.<br/>
이제 `(42,)` 형태의 1차원 배열을 `(42,1)` 형태의 2차원 배열로 변환하자.
```python
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape)
```
    (결과) (42, 1) (14, 1)

## 4-3. 결정계수(R^2)
사이킷런에서 k-최근접 이웃 회귀 알고리즘 구현할 때 `KNeighborsRegressor` 클래스를 사용한다.
```python
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))
```
    (결과) 0.992809406101064

분류에서는 `score()` 함수로 정확도를 출력하였다. 회귀에서는 정확한 숫자를 예측한다는 것이 사실 거의 불가능하다. 이에, 조금 다른 값으로 평가를 하는데 이 점수를 **결정계수**라고 한다. 또한 **$R^{2}$** 라고도 한다. 이 값은 다음 산출식에 의해 나타난다.

$R^{2} = 1 - \frac{(타깃-예측)^{2}의 합}{(타깃-평균)^{2}의 합}$

만약 타깃의 평균 정도를 예측하는 수준이라면 $R^{2}$ 는 0에 가까워 진다. 반면 예측이 타깃에 아주 가까워지면 1에 가까운 값이 된다.<br/>
이 값으로는 직감적으로 얼마나 좋은 수치인건지 이해하기가 어렵다. 타깃과 예측한 값 사이의 차이를 구해본다면 어느 정도 예측이 벗어난 건지 알 수 있지 않을까? 사이킷런은 `sklearn.metrics` 패키지 아래 여러 측정 도구를 제공한다. `mean_absolute_error`는 타깃과 예측의 절대값 오차를 평균하여 반환한다.

```python
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
```
    (결과) 19.157142857142862

평균적으로 예측이 19g 정도 타깃갓과 다르다는 것을 확인할 수 있다.

지금까지 우리는 훈련 세트를 사용해 모델을 훈련하고, 테스트 세트로 모델을 평가하였다. 훈련 세트만으로는 모델을 평가까지 할 수 없을까?

## 4-4. 과대적합 vs 과소적합
한번 `score()` 메소드에 훈련 세트를 넣어보자.

```python
print(knr.score(train_input, train_target))
```
    (결과) 0.9698823289099254

테스트 세트를 넣었을 때는 0.99가 나왔다. 왜 훈련 세트가 더 낮게 나왔을까?<br/>
사실 일반적으로는 훈련 세트의 점수가 더 높게 나온다. 훈련 세트로 모델을 훈련했을 것이기 때문이다. 만약 훈련 세트에서 점수가 굉장히 좋았는데, 테스트 세트에서 점수가 굉장히 낮았다면, 모델이 훈련 세트에 **과대적합** 되었다고 한다. 즉, 훈련 세트에 대해서만 잘 맞는 모델인 것이다. 반대로 훈련 세트보다 테스트 세트의 점수가 높거나 두 점수가 너무 낮게 나왔다면, 모델이 훈련 세트에 **과소적합** 되었다고 한다. 즉, 모델이 너무 단순하여 아직 훈련 세트에 적절히 훈련되지 않은 것이다. 또한 과소적합은 훈련 세트와 테스트 세트의 크기가 매우 작아서 발생될 수도 있다. 데이터가 작으면 테스트 세트가 훈련 세트의 특징을 따르지 못할 수도 있다.

현재 score는 테스트 세트일 때가 훈련 세트일 때 보다 더 높으므로 과소적합이라 할 수 있다. 훈련 세트에 더 잘 맞게 만들어 모델을 더 복잡하게 만들자. k-최근접 이웃 알고리즘으로 모델을 더 복잡하게 만드는 방법은 이웃의 개수 k를 더 줄이는 것이다. 이웃의 개수를 줄이면 훈련 세트에 있는 국지적인 패턴에 민감해지고, 이웃의 개수를 늘리면 데이터 전반에 있는 일반적인 패턴을 따른다.
```python
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))
```
    (결과) 0.9804899950518966
           0.9746459963987609

테스트 세트의 점수가 훈련 세트보다 낮아졌고, 두 점수의 차이가 크지 않다. 딱 좋아 보인다.
