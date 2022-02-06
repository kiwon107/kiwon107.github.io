---
layout: single
title: "[Machine learning] 3. 데이터 전처리"
folder: "machinelearning"
categories:
    - machinelearning
tag:
    - [machine learning, data preprocessing, standard score, broadcasting]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 3-1. 넘파이로 데이터 준비하기
2장에서 썼던 도미와 빙어 데이터를 그대로 활용해보자. 넘파이까지 임포트 할 거다.

```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import numpy as np
```

넘파이의 `column_stack()` 함수는 전달받은 리스트를 일렬로 세우고 이들을 나란히 연결한다. 다음 예를 보자

```python
np.column_stack(([1, 2, 3], [4, 5, 6]))
```
    (결과) array([[1, 4],
                  [2, 5],
                  [3, 6]])

두 리스트를 일렬로 세우고 튜플로 전달한다. 이제 `fish_length`와 `fish_weight`를 합쳐보자.

```python
fish_data = np.column_stack((fish_length, fish_weight))
print(fish_data[:5])
```
    (결과) [[ 25.4 242. ]
            [ 26.3 290. ]
            [ 26.5 340. ]
            [ 29.  363. ]
            [ 29.  430. ]]

이제 타깃 데이터를 만들어보자. 2장에서는 원소가 하나인 리스트 `[1]`과 `[0]` 을 여러 번 곱해서 타깃 데이터를 만들었다. 이번에는 `np.ones()`와 `np.zeros()` 함수를 쓸 것이다. `np.concatenate()` 함수를 사용하여 타깃 데이터를 만들어보자.

```python
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
print(fish_target)
```
    (결과) [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

2장에서 작성한 코드와 차이점은 이전에는 파이썬 리스트를 사용해 수동으로 만들었지만 이번에는 넘파이를 사용했다는 것이다. 넘파이 배열은 핵심 부분이 C, C++ 같은 저수준 언어로 개발되어 빠르고, 데이터 과학 분야에 알맞게 최적화되어 있다고 한다.

## 3-2. 사이킷런으로 훈련 세트와 테스트 세트 나누기
이번에는 사이킷런을 사용할 것이다. 사이킷런은 머신러닝 모델을 위한 알고리즘뿐만 아니라 다양한 유틸리티 도구도 제공한다. `train_test_split()` 함수는 사이킷런의 `model_selection` 모듈 아래 있다. 해당 함수를 임포트 하자.

```python
from sklearn.model_selection import train_test_split
```

이전에는 데이터를 무작위로 섞기 전 np.random.seed() 함수를 사용하였다. `train_test_split()` 함수는 자체적으로 랜덤 시드를 지정할 수 있는 `random_state` 매개변수가 있다.
```python
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
```

위와 같이 코딩하면 전달한 2개의 배열 `fish_data`, `fish_target`이 나뉘어 4개의 배열로 반환된다. 처음 2개는 입력 데이터 `train_input`, `test_input` 나머지 2개는 `train_target`, `test_target`이다. 이 함수는 기본적으로 25%를 테스트 세트로 떼어낸다.

```python
print(train_input.shape, test_input.shape)
print(train_target.shape, test_target.shape)
```
    (결과) (36, 2) (13, 2)
           (36,) (13,)

넘파이 배열의 shape 속성으로 입력 데이터 크기를 출력해보니 맞게 나뉘었다. 도미와 빙어가 잘 섞였는지 테스트 데이터를 출력해보자.

```python
print(test_target)
```
    (결과) [1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

아쉽게도 도미가 10개, 빙어가 3개 들어있다. 빙어의 비율이 조금 모자르다. 전체 데이터에서는 도미와 빙어 비율이 약 2.5:1 였는데, 테스트 데이터에는 비율이 3.3:1 이다.<br/>
무작위로 데이터를 나누면 샘플이 이처럼 골고루 섞이지 않을 수 있다. 특히 일부 클래스 개수가 적으면 더욱 그렇다. `train_test_split()` 함수는 이런 문제를 해결할 수 있다. `stratify` 매개변수에 타깃데이터를 전달하면 클래스 비율에 맞게 데이터를 나눈다. 훈련 데이터가 작거나 특정 클래스의 샘플 개수가 적을 때 특히 유용하다.

```python
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
print(test_target)
```
    (결과) [0. 0. 1. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1.]

빙어가 하나 더 늘어나 2.25:1이 되었다.

## 3-3. 수상한 도미 한 마리
준비한 데이터로 K-최근접 이웃을 훈련해보자.
```python
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
```
    (결과) 1.0

이 모델에 새로운 도미 데이터를 넣어보자.
```python
print(kn.predict([[25, 150]]))
```
    (결과) [0.]

도미로 분류해야하는데 빙어로 분류했다. 결과가 잘못 산출되었다. 산점도를 그려보자.

```python
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
![그림 3-1. 코드 결과](/assets/images/machinelearning/3-1.png)
{: .align-center}
그림 3-1. 코드 결과

K-최근접 이웃은 주변의 샘플 중 다수인 클래스를 예측으로 사용한다 하였다. `KNeighborsClassifier` 클래스의 이웃 개수인 `n_neighbors`의 기본값은 5이다. 즉 5개의 이웃이 반환된다. 한번 확인해보자.

```python
distances, indexes = kn.kneighbors([[25, 150]])
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker = '^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')  # 마름모 꼴로 산점도 표시
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
![그림 3-2. 코드 결과](/assets/images/machinelearning/3-2.png)
{: .align-center}
그림 3-2. 코드 결과

확인해보면 가장 가까운 이웃 5개를 살폈을 때  4개가 빙어 데이터로 분류되어 있다. 참고로 위에서 뽑은 `distances` 배열에는 이웃 샘플까지의 거리가 담겨있다.

```python
print(distances)
```
    (결과) [[ 92.00086956 130.48375378 130.73859415 138.32150953 138.39320793]]

## 3-4. 기준을 맞춰라
산점도랑 실제 거리를 비교해보면 뭔가 이상하다. 산점도에서 130대 거리는 92 거리보다 몇배는 더 길어보이는데 130 밖에 안된다니?<br/>
이상해보이는 이유는 산점도의 x축은 범위가 10~40인 반면, y축 범위는 0~1000 이기 때문이다. 즉, y축으로 거리가 조금만 멀어져도 아주 큰 값으로 `distances` 배열 값이 계산된다는 것이다. x축 범위를 0~1000으로 확장하여 확인해보자.

```python
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
![그림 3-3. 코드 결과](/assets/images/machinelearning/3-3.png)
{: .align-center}
그림 3-3. 코드 결과

이렇게 확인해보니 좀 더 와닿지 않는가?<br/>
이렇듯, 두 특성의 값이 놓인 범위가 매우 다르다. 이를 두 특성의 스케일이 다르다고 말한다. 특성 간 스케일은 당연히 다를 수 있다. 그러나 데이터를 표현하는 기준이 다르면 알고리즘은 이를 올바르게 예측할 수 없다. 특히 거리 기반으로 무언가를 판단하는 알고리즘일수록 더욱 그렇다. 이런 알고리즘들은 샘플 간의 거리에 영향을 많이 받으므로 제대로 사용하기 위해서는 특성값을 일정한 기준으로 맞춰주어야 한다. 이런 작업을 **데이터 전처리** 라 한다.<br/>
가정 널리 사용하는 전처리 방법 중 하나는 **표준점수**이다. 표준점수는 각 특성값이 0에서 표준편차의 몇 배 만큼 떨어져 있는지를 나타낸다.
```python
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
print(mean, std)
```
    (결과) [ 27.29722222 454.09722222] [  9.98244253 323.29893931]

`np.mean()` 함수는 평균을 계산하고, `np.std()` 함수는 표준편차를 계산한다.`axis=0`을 지정하여, 행을 따라 각 열의 통계값이 계산되도록 하였다.<br/>
이제 표준점수로 변환해보자.
```python
train_scaled = (train_input - mean) / std
```

넘파이는 `train_input`의 모든 행에서 `mean`에 있는 두 평균값을 빼준다. 그리고 `std`에 있는 두 표준편차를 다시 모든 행에 적용한다. 이러한 기능을 **브로드캐스팅** 이라고 한다.

## 3-5. 전처리 데이터로 모델 훈련하기
새로운 데이터 `[25, 150]` 도 표준점수를 구하여야 한다.
```python
new = ([25, 150] - mean) / std
```

위에서 구한 표준점수를 산점도로 표현해보자.
```python
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
![그림 3-4. 코드 결과](/assets/images/machinelearning/3-4.png)
{: .align-center}
그림 3-4. 코드 결과

이제 이 데이터로 다시 K-최근접 이웃 모델을 훈련시켜보자.
```python
kn.fit(train_scaled, train_target)
```

이제 테스트 세트로 평가를 할 것이다. 주의할 점은, 테스트 세트도 표준점수를 구해줘야 한다는 것인데 이때 훈련 세트로 구한 평균과 표준편차를 이용하여 표준점수를 구해야한다는 것이다. 그렇지 않으면 데이터의 스케일이 같아지지 않으므로 훈련한 모델이 쓸모없게 된다.
```python
test_scaled = (test_input - mean) / std
print(kn.score(test_scaled, test_target))
print(kn.predict([new]))
```
    (결과) 1.0
           [1.]

이번에는 새로운 데이터에 대해 분류가 잘 되었다. 한번 최근접 이웃 샘플 5개에 대한 데이터를 산출하여 산점도로 나타내보자.

```python
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
![그림 3-5. 코드 결과](/assets/images/machinelearning/3-5.png)
{: .align-center}
그림 3-5. 코드 결과