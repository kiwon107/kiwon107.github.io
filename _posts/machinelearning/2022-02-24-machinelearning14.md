---
layout: single
title: "[Machine learning] 14. 주성분 분석"
folder: "machinelearning"
categories:
    - machinelearning
tag:
    - [machine learning, dimensionality reduction, principal components analysis, explained variance]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 14-1. 차원과 차원 축소
13장에서 과일 사진은 10.000개의 픽셀이 있었다. 이는 10,000개의 특성이 있는것과 같다. 이런 특성을 머신러닝 에서는 **차원(Dimension)**이라고 한다. 이 차원을 줄일 수 있다면 저장 공간을 크게 절약할 수 있다. 참고로 다차원 배열에서 차원은 배열의 축 개수이다. 예를 들어 2차원 배열에서는 행과 열이 차원이 된다. 그러나 1차원 배열에서는 원소의 개수가 차원이다.<br/>

이제 비지도 학습 작업 중 하나인 **차원 축소(Dimensinoality reduction)**를 알아보자. 특성이 많으면 선형 모델의 성능이 높아지고 훈련 데이터에 쉽게 과적합된다. 차원 축소는 데이터를 가장 잘 나타내는 일부 특성을 선택하여 데이터의 크기를 줄이고 지도 학습 모델의 성능은 향상시킬 수 있는 방법이다. 물론 줄어든 차원에서 다시 원본 차원으로 손실을 최대한 줄이며 복원할 수도 있다.

## 14-2. 주성분 분석 소개
**주성분 분석(Principal component analysis)**은 데이터의 분산이 큰 방향을 찾는 것으로 이해할 수 있다. 분산이란 데이터가 널리 퍼져있는 정도를 말한다. 분산이 큰 방향은 데이터로 잘 표현하는 벡터라 생각할 수 있다. 이렇게 분산이 큰 방향의 벡터를 **주성분(Principal component)**라고 한다. 주성분 벡터는 원본 데이터에 있는 어떤 방향이다. 주성분은 원본 차원과 같으며, 원본 데이터는 주성분을 사용하여 차원을 줄일 수 있다.<br/>
주성분은 가장 분산이 큰 방향이므로 주성분에 투영하여 바꾼 데이터는 원본이 가지고 있는 특성을 가장 잘 나타낸다. 첫 번째 주성분을 찾고 이 벡터에 수직이며 분산이 가장 큰 다음 방향을 찾으면, 이 벡터가 두 번째 주성분이 된다. 참고로 기술적인 이유 때문에 주성분은 원본 특성의 개수와 샘플 개수 중 작은 값만큼 찾을 수 있다. 비지도 학습은 일반적으로 대량의 데이터에서 수행하므로 원본 특성의 개수만큼 찾을 수 있다고 한다.

## 14-3. PCA 클래스
```python
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.decomposition import PCA
pca = PCA(n_components=50)  # n_components에 주성분 개수 지정
pca.fit(fruits_2d)
print(pca.components_.shape)  # pca가 찾은 주성분은 components_ 속성에 저장됨
```
    (결과) (50, 10000)

보다시피 `pca.components_` 배열의 첫 번째 차원이 50이다. 즉 50개의 주성분을 찾은 것이다. draw_fruits() 함수를 만들고, 주성분들의 그림을 그려보자
```python
def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
    
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
                axs[i, j].axis('off')
    plt.show()

import matplotlib.pyplot as plt

draw_fruits(pca.components_.reshape(-1, 100, 100))
```
![그림 14-1. 코드 결과](/assets/images/machinelearning/14-1.png)
{: .align-center}
그림 14-1. 코드 결과

원본 데이터에서 가장 분산이 큰 방향을 순서대로 표시했다. 주성분을 찾았으니, 원본 데이터를 주성분에 투영하여 특성 개수를 10,000개에서 50개로 줄일 수 있다.
```python
print(fruits_2d.shape)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
```
    (결과) (300, 10000)
           (300, 50)

(300, 10000) 크기의 배열이 50개의 주성분을 찾은 PCA 모델을 사용하여 (300, 50) 크기의 배열로 변환됐다.

## 14-4. 원본 데이터 재구성
10,000개의 특성이 50개로 줄었으니 어느 정도 손실이 발생할 수밖에 없다. 그러나 분산이 큰 방향으로 데이터를 투영했으므로 원본 데이터를 상당 부분 재구성할 수 있다. `PCA`클래스는 `inverse_transform()` 메소드를 제공한다. 앞서 50개의 차원으로 축소한 `fruits_pca` 데이터를 전달하여 10,000개 특성을 복원하겠다.
```python
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)
```
    (결과) (300, 10000)

예상대로 10,000개의 특성이 복원되었다. 이제 100 x 100 크기로 바꾸어 그림을 그려보자.
```python
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print('\n')
```
![그림 14-2. 코드 결과](/assets/images/machinelearning/14-2.png)
{: .align-center}
그림 14-2. 코드 결과
![그림 14-3. 코드 결과](/assets/images/machinelearning/14-3.png)
{: .align-center}
그림 14-3. 코드 결과
![그림 14-4. 코드 결과](/assets/images/machinelearning/14-4.png)
{: .align-center}
그림 14-4. 코드 결과

모든 과일이 잘 복원되었다. 일부 흐리고 번진 부분도 있지만 이정도면 양호하다.

## 14-5. 설명된 분산
주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값을 **설명된 분산(Explained variance)**라고 한다. `explained_variance_ratio_`에 각 주성분의 설명된 분산 비율이 기록되어 있다. 첫 번째 주성분의 설명된 분산이 가장 크다. 이 분산 비율을 모두 더하면 50개의 주성분으로 표현하고 있는 총 분산 비율을 얻을 수 있다.
```python
print(np.sum(pca.explained_variance_ratio_))
```
    (결과) 0.9214529386197033

설명된 분산의 비율을 그래프로 그려보면 적절한 주성분의 개수를 찾는데 도움이 된다.
```python
plt.plot(pca.explained_variance_ratio_)
plt.show()
```
![그림 14-5. 코드 결과](/assets/images/machinelearning/14-5.png)
{: .align-center}
그림 14-5. 코드 결과

그래프를 보면 거의 10개의 주성분이 대부분의 분산을 표현하고 있다.

## 14-6. 다른 알고리즘과 함께 사용하기
과일 사진 원본 데이터와 PCA로 축소한 데이터를 지도 학습에 적용해 보고 어떠한 차이가 있는지 보자.
```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
target = np.array([0]*100 + [1]*100 + [2]*100)

from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
```
    (결과) 0.9966666666666667
           0.6728623867034912

교차 검증의 점수는 0.997 정도로 매우 높다. 특성이 10,000개나 되므로 300개의 샘플에서는 금방 과적합된 모델을 만들기 쉽다. `fit_time` 항목에는 각 교차 검증 폴드의 훈련 시간이 기록된다. 0.67초 정도 걸렸다. 이번에는 PCA로 축소한 데이터를 넣어보자.
```python
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
```
    (결과) 1.0
           0.03979372978210449

50개의 특성만 사용했음에도 불구하고 정확도는 100%, 훈련 시간은 0.03초 밖에 안걸렸다. PCA로 훈련 데이터의 차원을 축소하면 저장 공간뿐만 아니라 머신러닝 모델의 훈련 속도도 높일 수 있다. `PCA` 클래스의 `n_components` 매개변수에 주성분 개수가 아닌 설명된 분산의 비율을 넣을 수도 있다. 이는 지정된 비율에 도달할 때 까지 자동으로 주성분을 찾는다.
```python
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
print(pca.n_components_)
```
    (결과) 2

이제 해당 PCA 모델로 원본 데이터를 변환해보자.
```python
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
```
    (결과) (300, 2)
           c:\users\lg\appdata\local\programs\python\python37\lib\site-packages\sklearn\linear_model\_logistic.py:765: ConvergenceWarning: lbfgs failed to converge (status=1):
           STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
           Increase the number of iterations (max_iter) or scale the data as shown in:
               https://scikit-learn.org/stable/modules/preprocessing.html
           Please also refer to the documentation for alternative solver options:
               https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
           extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
           c:\users\lg\appdata\local\programs\python\python37\lib\site-packages\sklearn\linear_model\_logistic.py:765: ConvergenceWarning: lbfgs failed to converge (status=1):
           STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
           0.9966666666666667
           0.06479973793029785

로지스틱 회귀 모델이 완전히 수렴하지 못하여 반복 횟수를 증가하라는 경고(ConvergenceWarning: lbfgs failed to converge)가 출력됐지만, 교차 검증의 결과가 충분이 좋게 나왔다.

이번에는 차원 축소된 데이터를 k-평균 알고리즘에 적용해보자.
```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))

for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")
```
    (결과) (array([0, 1, 2]), array([110,  99,  91], dtype=int64))
![그림 14-6. 코드 결과](/assets/images/machinelearning/14-6.png)
{: .align-center}
그림 14-6. 코드 결과
![그림 14-7. 코드 결과](/assets/images/machinelearning/14-7.png)
{: .align-center}
그림 14-7. 코드 결과
![그림 14-8. 코드 결과](/assets/images/machinelearning/14-8.png)
{: .align-center}
그림 14-8. 코드 결과

훈련 데이터를 줄일 때, 3개 이하로 차원을 줄일 경우 화면에 출력하기가 쉽다. fruits_pca 데이터가 2개의 특성을 가지므로, 2차원 평면에 산점도로 표현할 수 있다.
```python
for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['pineapple', 'banana', 'apple'])
plt.show()
```
![그림 14-9. 코드 결과](/assets/images/machinelearning/14-9.png)
{: .align-center}
그림 14-9. 코드 결과
