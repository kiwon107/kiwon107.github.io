---
layout: single
title: "[Machine learning] 6. 특성 공학과 규제"
folder: "machinelearning"
categories:
    - machinelearning
tag:
    - [machine learning, multiple regression, feature engineering]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 6-1. 다중 회귀
5장에서 하나의 특성(생선 길이)을 사용하여 선형 회귀 모델을 훈련시켰다. 여러 개의 특성을 사용한 선형 회귀를 **다중 회귀**라고 한다.
1개의 특성을 사용하면 직선을 학습한다. 2개의 특성을 사용하면 선형 회귀는 평면을 학습한다. 그 이상의 개수에 대한 특성을 사용하면 시각적으로 표현이 불가능하다. 그러나 특성이 많은 고차원에서는 선형 회귀가 매우 복잡한 모델을 표현할 수 있다.<br/>
이번 예제에서는 '농어 길이', '농어 높이', '농어 두께'도 함께 사용해보자. 그리고 '농어 길이 x 농어 높이', '농어 길이 x 농어 두께', '농어 높이 x 농어 두께' 특성도 추가할 것이다. 이렇게 기존의 특성을 사용하여 새로운 특성을 뽑아내는 작업을 **특성 공학**이라고 부른다.

## 6-2. 데이터 준비
농어의 특성이 3개로 늘어났다. 일일이 데이터를 복붙하는것도 이제 번거로워졌다. 엑셀파일에 저장된 녀석을 긁어오는 방법이 있다면 얼마나 좋을까? **판다스**는 유명한 데이터 분석 라이브러리로, 데이터프레임이라는 핵심 데이터 구조를 제공한다. 넘파이 배열과 비슷하게 다차원 배열을 다룰 수 있고 넘파이보다 더 많은 기능을 제공한다. 데이터프레임은 넘파이 배열로 쉽게 바꿀 수도 있다. 판다스의 `read_csv()` 함수에 주소를 넣어보자. 이 함수로 데이터프레임을 만들고 `to_numpy()` 메소드를 이용하여 데이터프레임을 넘파이 배열로 바꿔보자.

```python
import pandas as pd
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)
```
    (결과) [[ 8.4   2.11  1.41]
           [13.7   3.53  2.  ]
           ...
           [44.   12.49  7.6 ]]

타깃값은 하드코딩으로 입력하자.
```python
import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
```

그리고 훈련 세트와 테스트 세트로 전체 데이터를 나누자.
```python
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
```

## 6-3. 사이킷런의 변환기
사이킷런은 특성을 만들거나 전처리하기 위한 다양한 클래스를 제공한다. 이런 클래스를 **변환기** 라고 부르는데, 사이킷런의 모델 클래스와 비슷하게 변환기 클래스도 모두 `fit()`, `transform()` 메소드를 제공한다. 한번 `PolynomialFeatures` 클래스를 활용해보자.
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
```
    (결과) [[1. 2. 3. 4. 6. 9.]]

`PolynomialFeatures`의 경우 `fit`해야 `transform`이 가능하다. 이를 하나로 붙인 `fit_transform` 메소드도 제공하니 참고하자! 어째됐건 `[[2, 3]]` 2개 특성에 대해 변환시켰더니 6개의 특성이 생겼다. 4, 6, 9는 대충 이해가 가는데, 1은 무엇일까? 이건 절편에 해당하는 녀석이라 할 수 있다. 하지만 사이킷런의 선형 모델은 자동으로 절편을 추가한다. 굳이 이렇게 만들필요가 없다. `include_bias=False`로 설정하여 다시 변환하자.
```python
poly = PolynomialFeatures(include_bias=False)
print(poly.fit_transform([[2, 3]]))
```
    (결과) [[2. 3. 4. 6. 9.]]

이제 실제 데이터로 변환을 해보자.
```python
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)  # 특성의 조합을 준비하기만 하고 별도 통계 값을 구하지는 않음. 따라서 테스트 세트 따로 변환해도 됨.
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input) # 훈련 세트를 기준으로 테스트 세트 변환!
print(train_poly.shape)
print(poly.get_feature_names())
```
    (결과) (42, 9)
           ['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2']

## 6-4. 다중 회귀 모델 훈련하기
이제 다중 회귀 모델을 훈련시켜보자. `LinearRegression` 클래스를 임포트하여 훈련시키자.
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
```
    (결과) 0.9903183436982124
           0.9714559911594199

상당히 높은 점수가 나왔고, 과소적합 경향은 전혀 안보인다. 특징을 제곱말고, 3제곱, 4제곱항까지 넣으면 어떻게 될까? 5제곱까지 특성을 만들어서 출력하고 훈련까지 시켜 성능을 확인해보자.
```python
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
```
    (결과) (42, 55)
           0.9999999999997522
           -144.40564434202673

특성 개수가 55개로 엄청 늘어났다. 훈련 세트에 대한 스코어는 거의 1에 육박할 정도로 엄청 높게 나왔다. 하지만 테스트 세트에 대한 스코어는 마이너스이다. 왜 그럴까?<br/>
특성의 개수를 크게 늘리면 선형 모델은 매우 강력해진다. 훈련 세트에 대해 완벽하게 학습이 가능하다. 하지만 이는 훈련 세트에 대한 과대적합으로 이어져, 테스트 세트에 대해서는 형편없는 점수를 만들게 된다. 과대적합을 줄이려면 어떻게 해야할까?

## 6-5. 규제
규제는 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하다록 훼방하는 것을 말한다. 선형 회귀 모델에서는 특성에 곱해지는 계수의 크기를 작게 만든다.<br/>
규제에 대한 예를 보여주기 전에, 정규화를 먼저 수행하여, 계수 값의 크기가 서로 많이 다르지 않도록 만들자! 정규화되지 않으면, 곱해지는 계수들 사이의 값 차이도 상당히 나게 될 것이다. 이번에는 사이킷런에서 제공하는 StandardScaler 클래스를 사용할 것이다.
```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```
참고로 훈련 세트에서 학습한 평균과 표준편차는 StandardScaler 클래스 객체의 `mean_`, `scale_` 속성에 저장된다. 특성마다 평균과 표준편차를 계산하기 때문에 55개의 평균과 표준편차가 들어있을 것이다.

## 6-6. 릿지 회귀
릿지 회귀와 라쏘 회귀는 모두 `sklearn.linear_model` 패키지 안에 있다.<br/>
릿지는 계수를 제곱한 값을 기준으로 규제를 적용한다. 참고로 라쏘보다는 릿지를 조금 더 선호한다.
```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
```
    (결과) 0.9896101671037343
           0.9790693977615391

결과가 상당히 좋게 나왔다. 많은 특성을 사용했지만, 훈련 세트에 너무 과적합되지 않아 테스트 세트에서도 좋은 성능을 냈다.

릿지와 라쏘 모델을 사용시, 규제의 양을 조절할 수 있다. 모델 객체를 만들 때, alpha 매개변수로 규제의 강도를 조절한다. alpha 값이 크면 규제 강도가 세져 계수 값을 더 줄이고 조금 더 과소적합되도록 유도한다. alpha 값이 작으면 계수를 줄이는 역할이 줄어들고 선형 회귀 모델과 유사해져서 과대적합될 가능성이 크다. 참고로 alpha값은 우리가 지정해줘야하는 값으로, 이러한 사람이 알려줘야 하는 파라미터를 하이퍼파라미터라고 한다. 머신러닝 라이브러리에서는 보통 클래스와 메소드의 매개변수로 하이퍼파라미터가 표현된다.<br/>

그럼 적절한 alpha 값을 어떻게 찾을 수 있을까? 한가지 방법은 $R^{2}$ 값의 그래프를 그려보는 것이다. 훈련 세트와 테스트 세트의 점수가 가장 가까운 지점이 최적 alpha 값이 된다.
```python
import matplotlib.pyplot as plt

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```
![그림 6-1. 코드 결과](/assets/images/machinelearning/6-1.png)
{: .align-center}
그림 6-1. 코드 결과


참고로 0.001 부터 10배씩 늘렸기 때문에 이대로 그래프를 그리면 왼쪽이 너무 촘촘해져서 로그 함수를 적용하여 지수로 표현하였다. 그래프를 보면 왼쪽은 훈련 세트에는 잘 맞지만 테스트 세트에는 형편없는 성능을 보인 것으로 보아, 과대적합임이 확실하다. 오른쪽은 훈련 세트와 테스트 세트의 점수가 모두 낮아지는 과소적합 모습이다.
적절한 alpha 값은 0.1이다
```python
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
```
    (결과) 0.9903815817570367
           0.9827976465386834

## 6-6. 라쏘 회귀
라쏘는 계수의 절대값을 기준으로 규제를 적용한다. 라쏘의 경우 계수를 아예 0으로 만들 수 있다.
```python
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```
    (결과) 0.989789897208096
           0.9800593698421883

라쏘 모델도 alpha 매개변수로 규제의 강도를 조절할 수 있다.
```python
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel(alpha)
plt.ylabel('R^2')
plt.show()
```
![그림 6-2. 코드 결과](/assets/images/machinelearning/6-2.png)
{: .align-center}
그림 6-2. 코드 결과

참고로 라쏘 모델 훈련시 다음과 같은 경고가 발생할 수 있다
```python
c:\users\lg\appdata\local\programs\python\python37\lib\site-packages\sklearn\linear_model\_coordinate_descent.py:532: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 18778.697957792876, tolerance: 518.2793833333334
  positive)
c:\users\lg\appdata\local\programs\python\python37\lib\site-packages\sklearn\linear_model\_coordinate_descent.py:532: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 12972.821345404844, tolerance: 518.2793833333334
  positive)
```

사이킷런의 라쏘 모델은 최적의 계수를 찾기 위해 반복적인 계산을 수행하는데, 지정한 반복 횟수가 부족할 때 이런 경고가 발생한다고 한다. 이 반복 횟수를 충분히 늘리기 위해 max_iter 매개변수 값을 10000으로 지정했다고 하니 참고하자!

그래프를 보면 왼쪽은 과대적합 모습을 보이고, 오른쪽으로 갈수록 훈련 세트와 테스트 세트의 점수가 좁혀지고 있다. 가장 오른쪽은 갑자기 점수가 크게 떨어지는 모양새다. 이 지점은 과소적합 되는 모습이다. 해당 라쏘 모델에서 최적의 alpha 값은 10이다.
```python
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```
    (결과) 0.9888067471131867
           0.9824470598706695

특성을 많이 사용했지만, 릿지 모델처럼 라쏘 모델도 과대적합을 잘 억제하고, 테스트 세트의 성능을 크게 높혔다. 라쏘 모델의 계수를 확인해보자.
```python
print(lasso.coef_)
```
    (결과) [ 0.          0.          0.         12.14852453 55.44856399 42.23100799
             0.          0.         13.70596191  0.         43.2185952   5.7033775
             47.46254536  7.42309425 11.85823365  0.          0.         13.53038193
             21.22111356  0.          0.          0.          0.          0.
             0.         18.66993032  0.          0.          0.         15.81041778
             0.          0.          0.          0.          0.          0.
             0.          0.          0.          0.          0.          0.
             0.          0.         18.14672268  0.          0.          0.
             0.         15.51272953  0.          0.          0.          0.
             0.        ]

정말 많은 계수가 0이 되었다. 앞서 라쏘 모델은 계수 값을 아예 0으로 만들 수 있다고 했는데 정말로 사용한 특성은 15개 밖에 되지 않는다. 라쏘 모델은 유용한 특성을 골라내는 용도로도 사용할 수 있다.