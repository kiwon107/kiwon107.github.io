---
layout: single
title: "[Machine learning] 13. K-평균"
folder: "machinelearning"
categories:
    - machinelearning
tag:
    - [machine learning, k-means, cluster center, elbow method]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 13-1. K-평균 알고리즘 소개
k-평균 알고리즘 작동 방식은 다음과 같다.<br/>
 1). 무작위로 k개의 클러스터 중심을 정한다.<br/>
 2). 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정한다.<br/>
 3). 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경한다.<br/>
 4). 클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복한다.<br/>

평균값이 클러스터의 중심에 위치하므로 **클러스터 중심(Cluster center)** 또는 **센트로이드(Centroid)** 라고 한다.

## 13-2. KMeans 클래스
3차원 배열을 2차원 배열 형태로 변경하자.
```python
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)  # n_clusters로 클러스터 개수 지정
km.fit(fruits_2d)
print(km.labels_)
```
    (결과) [2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
           2 2 2 2 2 0 2 0 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 0 0 2 2 2 2 2 2 2 2 0 2
           2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0
           0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
           0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
           0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
           1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
           1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
           1 1 1 1]

레이블 0, 1, 2로 모은 샘플의 개수를 확인해보자.
```python
print(np.unique(km.labels_, return_counts=True))
```
    (결과) (array([0, 1, 2]), array([111,  98,  91], dtype=int64))

각 클러스터가 어떤 이미지를 나타냈는지 그림으로 출력하기 위한 함수를 만들어보자.
```python
import matplotlib.pyplot as plt

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
```

이제 불리언 인덱싱을 적용하여 True인 위치의 원소를 모두 추출하자.<br/>
```python
draw_fruits(fruits[km.labels_==0])
```
![그림 13-1. 코드 결과](/assets/images/machinelearning/13-1.png)
{: .align-center}
그림 13-1. 코드 결과

```python
draw_fruits(fruits[km.labels_==1])
```
![그림 13-2. 코드 결과](/assets/images/machinelearning/13-2.png)
{: .align-center}
그림 13-2. 코드 결과

```python
draw_fruits(fruits[km.labels_==2])
```
![그림 13-3. 코드 결과](/assets/images/machinelearning/13-3.png)
{: .align-center}
그림 13-3. 코드 결과

레이블이 0인 클러스터는 파인애플, 1인 클러스터는 바나나, 2인 클러스터는 사과로 이루어져 있다. 레이블이 0인 클러스터는 사과도 섞여 있는 것으로 보인다. 그럼에도 불구하고 비슷한 샘플들이 꽤 잘 모아졌다.

## 13-3. 클러스터 중심
`KMeans` 클래스가 최종적으로 찾은 클러스터 중심은 `cluster_centers_` 속성에 저장된다.
```python
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
```
![그림 13-4. 코드 결과](/assets/images/machinelearning/13-4.png)
{: .align-center}
그림 13-4. 코드 결과

`KMeans` 클래스는 훈련 데이터 샘플에서 클러스터 중심까지 거리로 변환해주는 `transform()` 메소드를 가진다.
```python
print(km.transform(fruits_2d[100:101]))
```
    (결과) [[3393.8136117  8837.37750892 5267.70439881]]

보다시피 반환된 배열은 크기가 (1, 클러스터 개수)인 2차원 배열이다. 첫 번째 클러스터까지의 거리가 가장 작은 것으로 보아 이 샘플은 레이블 0에 속한 것으로 보인다.
```python
print(km.predict(fruits_2d[100:101]))
```
    (결과) [0]

역시 짐작대로 레이블 0으로 예측했다.
```python
draw_fruits(fruits[100:101])
```
![그림 13-5. 코드 결과](/assets/images/machinelearning/13-5.png)
{: .align-center}
그림 13-5. 코드 결과

k-평균 알고리즘은 반복적으로 클러스터 중심을 옮기면서 최적의 클러스터를 찾는다. 알고리즘이 반복한 횟수는 `KMeans` 클래스의 `n_iter_` 속성에 저장된다.
```python
print(km.n_iter_)
```
    (결과) 4

우리는 `n_clusters=3` 이라는 걸 알고 지정했지만 실제로는 이러한 사실 조차 몰라야 한다. 어떻게 최적의 `n_clusters`를 찾을 수 있을까?

## 13-4. 최적의 k 찾기
적절한 클러스터 개수를 찾기 위한 대표적인 방법으로 **엘보우(Elbow)** 방법이 있다. 앞에서 우리는 `KMeans` 클래스의 `transform()` 메소드를 이용하여 클러스터 중심과 클러스터에 속한 샘플 사이의 거리를 잴 수 있었다. 이 거리의 제곱 합을 **이너셔(Inertia)** 라고 부른다.<br/>
이너셔는 클러스터에 속한 샘플이 얼마나 가깝게 모여있는지를 나타내는 값이다. 일반적으로 클러스터 개수가 늘어나면 클러스터 개개의 크기는 줄어드므로 이너셔도 줄어든다. 엘보우 방법은 클러스터 개수를 늘려가면서 이너셔의 변화를 관찰해 최적의 클러스터 개수를 찾는 방법이다.<br/>
클러스터 개수를 증가시키면서 이너셔를 그래프로 그리면 감소하는 속도가 꺾이는 지점이 있다. 이 지점부터 클러스터 개수를 늘려도 클러스터에 잘 밀집된 정도가 크게 개선되지 않는다. 즉 이너셔가 크게 줄어들지 않게 되는 것이다! 이 지점이 팔꿈찌 모양이어서 엘보우 방법이라 부르는 것이다.
```python
inertia=[]

for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)  # inertia_ 속성에 이너셔값 저장됨

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
```
![그림 13-6. 코드 결과](/assets/images/machinelearning/13-6.png)
{: .align-center}
그림 13-6. 코드 결과