---
layout: single
title: "[Machine learning] 12. 군집 알고리즘"
folder: "machinelearning"
categories:
    - machinelearning
tag:
    - [machine learning, unsupervised learning, histogram, cluster]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 12-1. 타깃을 모르는 비지도 학습
**비지도 학습(Unsupervised learning)**은 타깃이 없을 때 사용하는 머신러닝 알고리즘이다. 사람이 가르쳐 주지 않아도 데이터에 있는 무언가를 학습한다.

## 12-2. 과일 사진 데이터 준비하기
`!wget https://bit.ly/fruits_300_data -O fruits_300.npy` 를 코랩에 쳐서 npy 파일을 다운로드 한다. npy 파일은 넘파일 배열의 기본 저장 포맷이다. `!`문자로 시작하면 코랩은 이후 명령을 리눅스 셸 명령으로 이해한다. `wget` 명령은 원격 주소에서 데이터를 다운로드하여 저장한다. `-O` 옵션은 저장할 파일 이름을 지정한다.
```python
import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')
print(fruits.shape)
print(fruits[0, 0, :])
```
    (결과) (300, 100, 100)  # 샘플 개수, 이미지 높이, 이미지 너비
           [ 1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   2   1
             2   2   2   2   2   2   1   1   1   1   1   1   1   1   2   3   2   1
             2   1   1   1   1   2   1   3   2   1   3   1   4   1   2   5   5   5
             19 148 192 117  28   1   1   2   1   4   1   1   3   1   1   1   1   1
             2   2   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
             1   1   1   1   1   1   1   1   1   1]

```python
plt.imshow(fruits[0], cmap='gray')  # 흑백 이미지이므로 cmap 매개변수를 'gray'로 지정
plt.show()
```
![그림 12-1. 코드 결과](/assets/images/machinelearning/12-1.png)
{: .align-center}
그림 12-1. 코드 결과

보통 흑백 샘플 이미지는 바탕이 밝고 물체가 짙다. 그런데 그림 12-1은 그 반대다. 왜 그럴까?<br/>
이 흑백 이미지는 사진으로 찍은 이미지를 넘파이 배열로 변환할 때 반전 시킨 것이다. 우리의 관심 대상은 사과이지만 컴퓨터는 처음 이미지가 생성될 때 255에 가까운 값을 가진 바탕에 집중한다. 따라서 바탕을 검게 만들고, 사과를 밝게 만든 것이다.<br/>
cmap 매개변수를 `gray_r`로 지정하여 다시 반전 시켜보자.
```python
plt.imshow(fruits[0], cmap='gray_r')  # 흑백 이미지를 반전시킴
plt.show()
```
![그림 12-2. 코드 결과](/assets/images/machinelearning/12-2.png)
{: .align-center}
그림 12-2. 코드 결과

이 데이터는 사과, 바나나, 파인애플이 각각 100개씩 들어있다.
```python
fig, axs = plt.subplots(1, 2)  # 하나의 행과 2개의 열 지정
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()
```
![그림 12-3. 코드 결과](/assets/images/machinelearning/12-3.png)
{: .align-center}
그림 12-3. 코드 결과

맷플롯립의 `subplots()` 함수를 사용하면 여러 개의 그래프를 배열처럼 쌓을 수 있도록 도와준다. `axs`는 2개의 서브 그래프를 담고 있는 배열이다.

## 12-3. 픽셀값 분석하기
사용하기 쉽게 fruits 데이터를 사과, 파인애플, 바나나로 각각 나누어 보자. 넘파이 배열을 나눌 때 100 x 100 이미지를 펼쳐서 길이가 10,000 인 1차원 배열로 만들자.
```python
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
print(apple.shape)
```
    (결과) (100, 10000)
  
이제 넘파이 `mean()` 메소드를 사용하자. 샘플마다 픽셀의 평균값을 계산할 것이다. `axis=0`으로 하면 첫 번째 축인 행을 따라 계산하고, `axis=1`로 지정하면 두 번째 축인 열을 따라 계산한다. 샘플을 모두 가로로 나열했으므로, `axis=1`로 지정하여 평균을 계산하자.
```python
print(apple.mean(axis=1))
```
    (결과) [ 88.3346  97.9249  87.3709  98.3703  92.8705  82.6439  94.4244  95.5999
             90.681   81.6226  87.0578  95.0745  93.8416  87.017   97.5078  87.2019
             88.9827 100.9158  92.7823 100.9184 104.9854  88.674   99.5643  97.2495
             94.1179  92.1935  95.1671  93.3322 102.8967  94.6695  90.5285  89.0744
             97.7641  97.2938 100.7564  90.5236 100.2542  85.8452  96.4615  97.1492
             90.711  102.3193  87.1629  89.8751  86.7327  86.3991  95.2865  89.1709
             96.8163  91.6604  96.1065  99.6829  94.9718  87.4812  89.2596  89.5268
             93.799   97.3983  87.151   97.825  103.22    94.4239  83.6657  83.5159
             102.8453  87.0379  91.2742 100.4848  93.8388  90.8568  97.4616  97.5022
             82.446   87.1789  96.9206  90.3135  90.565   97.6538  98.0919  93.6252
             87.3867  84.7073  89.1135  86.7646  88.7301  86.643   96.7323  97.2604
             81.9424  87.1687  97.2066  83.4712  95.9781  91.8096  98.4086 100.7823
             101.556  100.7027  91.6098  88.8976]

히스토그램을 그려보면 평균값이 어떻게 분포되어 있는지 한눈에 잘 볼 수 있다. 히스토그램은 값이 발생한 빈도를 그래프로 표시한 것이다. 보통 x 축이 값의 구간(계급)이고, y축은 발생 빈도(도수)이다. `alpha` 매개변수에 1보다 작은 값을 주어 투명하게 만들 수 있다. 투명도를 조절하여 겹친 부분도 보일 수 있도록 만들자. 또한 맷플롯립의 `legend()` 함수를 사용하여 어떤 과일의 히스토그램인지 범례를 만들자.
```python
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()
```
![그림 12-4. 코드 결과](/assets/images/machinelearning/12-4.png)
{: .align-center}
그림 12-4. 코드 결과

바나나 사진의 평균은 40 아래에 집중되어 있다. 사과와 파인애플은 90~100 사이에 많이 보인다. 바나나는 확실히 픽셀 평균값만으로 구분이 되는 반면 사과와 파인애플은 그렇지 않다. 겹쳐 보이는 구간이 꽤 있다. 만약 샘플의 평균값이 아닌 픽셀별 평균값을 비교해 보면 어떨까?
```python
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()
```
![그림 12-5. 코드 결과](/assets/images/machinelearning/12-5.png)
{: .align-center}
그림 12-5. 코드 결과

사과, 파인애플, 바나나 마다 각 이미지의 느낌이 막대그래프에 잘 나타나 있다.<br/>
이제 픽셀 평균값을 100 x 100 크기로 변환하여 이미지처럼 출력해보자.
```python
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()
```
![그림 12-6. 코드 결과](/assets/images/machinelearning/12-6.png)
{: .align-center}
그림 12-6. 코드 결과

## 12-4. 평균값과 가까운 사진 고르기
평균값인 apple_mean과 가까운 사진을 골라보자. 절대값 오차를 사용하여 찾을 것이다.
```python
abs_diff = np.abs(fruits - apple_mean)  # 절대값 반환! np.absolute()와는 다름
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)
```
    (결과) (300,)

그 다음, 오차값이 작은 순서대로 100개 고르자.
```python
apple_index = np.argsort(abs_mean)[:100]  # 작은 것에서 큰 순서대로 나열한 abs_mean 배열의 인덱스를 반환함
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
    axs[i, j].axis('off')
plt.show()
```
![그림 12-7. 코드 결과](/assets/images/machinelearning/12-7.png)
{: .align-center}
그림 12-7. 코드 결과

확인해보니 모두 사과로 잘 골랐다. 이렇게 비슷한 샘플끼리 그룹으로 모으는 작업을 **군집(Clustering)**이라 한다. 군집은 대표적인 비지도 학습 작업 중 하나이다. 군집 알고리즘에서 만든 그룹을 **클러스터(Cluster)**라고 한다. 비지도 학습에서 타깃값을 모르기 때문에 원래는 샘플의 평균값을 미리 구할 수 없다.