---
layout: single
title: "[Machine learning] 9. 결정 트리"
folder: "machinelearning"
categories:
    - machinelearning
tag:
    - [machine learning, decision tree, impurity, information gain, feature importance]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 9-1. 로지스틱 회귀로 와인 분류하기
와인 데이터를 한번 봐보자.
```python
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
wine.head()
```
    (결과) alcohol	sugar	 pH	   class
        0	9.4 	1.9	    3.51	0.0
        1	9.8	    2.6	    3.20	0.0
        2	9.8	    2.3	    3.26	0.0
        3	9.8	    1.9	    3.16	0.0
        4	9.4	    1.9	    3.51	0.0

도수, 당도, PH를 가지고 레드와인인지 화이트와인인지 맞춰야하는 문제이다.

info() 메소드를 사용하면 데이터프레임의 각 열 데이터 타입과 누락된 데이터가 있는지 확인하는데 유용하다.
```python
wine.info()
```
    (결과) <class 'pandas.core.frame.DataFrame'>
           RangeIndex: 6497 entries, 0 to 6496
           Data columns (total 4 columns):
           alcohol    6497 non-null float64
           sugar      6497 non-null float64
           pH         6497 non-null float64
           class      6497 non-null float64
           dtypes: float64(4)
           memory usage: 203.2 KB
    
와인 데이터를 보면 Non-Null Count가 6497이고, 총 샘플 개수도 6497개라고 나온다. 이에따라, 누락된 값은 없는것으로 보인다. 만약 누락된 값이 있으면, 그 데이터를 버리거나 평균값으로 채운 후 사용할 수 있다. 어떤 방식이 최선인지는 알기 어려우니 두 가지 모두 시도해 보는게 좋다! 만약 평균값으로 채워 넣는다고 하면, 테스트 세트의 누락된 값은 훈련 세트의 평균값으로 채워야 한다!

describe() 메소드를 사용하면 열데 대한 간략한 통계를 출력할 수 있다.
```python
wine.describe()
```
    (결과)      alcohol	    sugar	        pH	      class
        count	6497.000000	6497.000000	6497.000000	6497.000000
        mean	10.491801	5.443235	3.218501	0.753886
        std	    1.192712	4.757804	0.160787	0.430779
        min	    8.000000	0.600000	2.720000	0.000000
        25%	    9.500000	1.800000	3.110000	1.000000
        50%	    10.300000	3.000000	3.210000	1.000000
        75%	    11.300000	8.100000	3.320000	1.000000
        max	    14.900000	65.800000	4.010000	1.000000

평균, 표준편차, 최소, 최대값을 볼 수 있다. 중간값(50%), 1사분위수(25%), 3사분위수(75%)도 알려준다.
사분위수는 데이터를 일렬로 늘어놓고, 데이터를 순서대로 4등분 한 값이다. 만약 데이터 개수가 짝수개 이면 중앙값은 가운대 2개의 값의 평균을 사용한다고 한다.
로지스틱 회귀로 와인을 분류해보자!

```python
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
print(train_input.shape, test_input.shape)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```
    (결과) (5197, 3) (1300, 3)
           0.7808350971714451
           0.7776923076923077

훈련 세트, 테스트 세트 모두 점수가 낮은걸 보면 과소 적합인 것으로 보인다.

## 9-2. 설명하기 쉬운 모델과 어려운 모델
로지스틱 회귀가 학습한 계수와 절편을 출력해보자.
```python
print(lr.coef_, lr.intercept_)
```
    (결과) [[ 0.51270274  1.6733911  -0.68767781]] [1.81777902]

저 가중치와 편향이 어떤 것이라는건 알고있지만, 왜 저 값이 나왔는지 이해하기는 어렵다. 뭔가 왜 이런 결과가 나왔는지 쉽게 설명 가능한 모델은 없을까?

## 9-3. 결정 트리
**결정 트리(Decision Tree)**는 스무고개와 같다. 질문을 하나씩 던져 정답과 맞춰가는 것이 바로 결정 트리의 원리이다. 데이터를 잘 나눌 수 있는 질문을 찾으면 계속 질문을 추가해서 분류 정확도를 높일 수 있다. 사이킷런의 `DecisionTreeClassifier` 클래스를 사용하여 결정 트리 모델을 훈련할 수 있다. 결정 트리 알고리즘은 노드에서 최적의 분할을 찾기 전에 특성의 순서를 섞는다. 약간의 무작위성이 들어가기 때문에 실행할 때마다 점수가 조금씩 달라질 수 있다.
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
```
    (결과) 0.996921300750433
           0.8592307692307692

약간 과대적합된 것으로 보인다. 과연 이 모델은 어떻게 쉽게 설명할 수 있을까? `plot_tree()` 함수를 사용하면 확인할 수 있다.
```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 7))
plot_tree(dt)
plt.show()
```
![그림 9-1. 코드 결과](/assets/images/machinelearning/9-1.png)
{: .align-center}
그림 9-1. 코드 결과

보다시피 엄청난 트리가 만들어졌다. 맨 위의 노드를 루트 노드라고 하고, 맨 아래 끝 노드를 리프 노드라고 한다. 노드는 결정 트리를 구성하는 핵심 요소이다. 노드는 훈련 데이터의 특성에 대한 테스트를 표현한다. 가지는 테스트의 결과를 나타내고 일반적으로 하나의 노드는 2개의 가지를 가진다. 한번 트리의 길이를 제한해서 다시 출력해보자. `plot_tree()` 함수는 몇가지 매개변수를 갖는다. `max_depth`를 1로 주면 루트 노드를 제외하고 하나의 노드를 더 확장하여 그린다. `filled` 매개변수에서 클래스에 맞게 노드의 색을 칠할 수도 있다. `feature_names`는 특성의 이름을 전달할 수 있다.
```python
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```
![그림 9-2. 코드 결과](/assets/images/machinelearning/9-2.png)
{: .align-center}
그림 9-2. 코드 결과

그림 9-2를 보자. 박스의 가장 위에 있는건 테스트 조건을 말한다. 테스트 조건을 충족하면 왼쪽, 아니면 오른쪽 가지로 간다. 박스의 세번째 값은 총 샘플수를 의미한다. 박스의 마지막 값은 양성 클래스(화이트 와인)와 음성 클래스(레드 와인) 개수를 의미한다. 루트 노드를 지나 두번째 노드들중 왼쪽을 보면 양성 클래스의 비율이 크게 줄었다. 오른쪽 노드는 반대로 양성 클래스 비율이 높다. `filled=True`로 지정했으므로, 어떤 클래스의 비율이 높아지면 점점 진한색으로, 반대면 연한색으로 변하게 된다.<br/>
결정 트리에서는 리프 노드에서 가장 많은 클래스가 예측 클래스가 된다. 참고로, 결정 트리를 회귀 문제에 적용하면, 리프 노드에 도달한 샘플의 타깃을 평균하여 예측값으로 사용하게 된다.<br/>
박스 안의 두번째 값은 불순도라는 것이다.

## 9-4. 불순도
**지니 불순도(gini impurity)**는 `DecisionTreeClassifier` 클래스의 `criterion` 매개변수을 기본값인 `gini`로 설정할 때 나온다. `criterion` 매개변수는 노드에서 데이터를 분할할 기준을 정하는 것이다. 지니 불순도는 다음 식으로 구할 수 있다.<br/>
$지니불순도=1-(음성클래스비율^{2}+양성클래스비율^{2})$<br/>
루트 노드를 기준으로 구하면 지니 불순도는 $1-((1258/5197)^{2}+(3939/5197)^{2}=0.367)$ 이다. 만약 클래스 비율이 양성 음성 모두 절반이라는 최악의 상황이 되면, 지니 불순도는 0.5가 나온다. 반대로 하나의 클래스에만 완전히 쏠려있다면 지니 불순도는 0이 된다.<br/>
결정 트리 모델은 부모 노드와 자식 노드의 불순도 차이가 가능한 크도록 트리를 성장 시킨다. 부모 노드와 자식 노드의 불순도 차이는 어떻게 계산될까? 먼저 자식 노드의 불순도를 샘플 개수에 비례하여 모두 더하고, 그 다음 부모 노드의 불순도에서 빼면 된다!
$부모불순도-(왼쪽노드샘플수/부모샘플수) \times 왼쪽노드불순도-(오른쪽노드샘플수/부모샘플수) \times 오른쪽노드불순도 = 0.367-(2922/5197) \times 0.481 - (2275/5197) \times 0.069 = 0.066$<br/>
부모와 자식 노드 사이의 불순도 차이를 바로 **정보 이득(information gain)**이라고 한다. 즉, 결정 트리 모델은 정보 이득이 최대가 되도록 데이터를 나눈다! 정보 이득을 구하기 위해 지니 불순도를 사용한다.<br/>
`criterion='entropy'`로 설정하면 엔트로피 불순도를 사용하게 된다. 엔트로피 불순도는 노드 클래스 비율을 사용하지만, 지니 불순도의 제곱이 아니고 밑이 2인 로그를 사용한다.<br/>
$ 엔트로피불순도=-음성클래스비율 \times log_{2}(음성클래스비율)-양성클래스비율 \times log_{2}(양성클래스비율) $ <br/>
루트 노드를 기준으로 구하면 엔트로피 불순도는 $ -(1258/5197) \times log_{2}(1258/5197)-(3939/5197) \times log_{2}(3939/5197) = 0.798 $<br/>
지니 불순도와 엔트로피 불순도의 결과 차이는 크지 않다. 결정 트리가 불순도 기준을 사용하여 정보 이득이 최대가 되도록 노드를 분할한다는 것을 알았다. 노드를 순수하게 나눌 수록 정보 이득은 커지게 된다. 마지막에 도달한 노드의 클래스 비율을 보고 예측을 만들게 된다!


## 9-5. 가지치기
결정 트리도 과대적합이 될 수 있다. 이에 따라 일반화를 해주어야 한다. 일반화 방법으로 가지치기가 있다. 가지치기를 하지 않으면 트리는 무작정 끝까지 자라나게 된다. `DecisionTreeClassifier` 클래스의 `max_depth`를 3으로 지정하여 모델을 만들어보자. 이렇게 하면 루트 노드 아래로 최대 3개의 노드까지만 성장할 수 있다.
```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
```
    (결과) 0.8454877814123533
           0.8415384615384616

`plot_tree()` 함수로 그려보자
```python
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```
![그림 9-3. 코드 결과](/assets/images/machinelearning/9-3.png)
{: .align-center}
그림 9-3. 코드 결과

그림을 보면 각 노드마다 어떤 특성을 기준으로 나누었는지 확인할 수 있다.<br/>
결정트리는 불순도를 기준으로 샘플을 나눈다. 불순도는 클래스 비율을 가지고 계산한다. 로지스틱 분류 방식 처럼 정규화를 해줘야할까? 결정 트리는 특성값의 스케일에 영향을 받지 않는다. 이것은 결정 트리의 장점 중에 하나라 할 수 있다. 그림 9-3에 당도가 -값을 갖는다는게 말이 되는가? 정규화를 안하면 이런 괴이한 현상을 보지 않아도 된다.
```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```
    (결과) 0.8454877814123533
           0.8415384615384616
![그림 9-4. 코드 결과](/assets/images/machinelearning/9-4.png)
{: .align-center}
그림 9-4. 코드 결과

마지막으로 결정 트리는 어떤 특성이 유용한지 나타내는 특성 중요도를 계산해준다. 특성 중요도는 각 노드의 정보 이득과 전체 샘플에 대한 비율을 곱한 후 특성별로 더하여 계산한다. 특성 중요도는 결정 트리 모델의 `feature_importances_` 속성에 저장되어 있다.
```python
print(dt.feature_importances_)
```
    (결과) [0.12345626 0.86862934 0.0079144 ]