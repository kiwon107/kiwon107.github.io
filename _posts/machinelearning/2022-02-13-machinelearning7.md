---
layout: single
title: "[Machine learning] 7. 로지스틱 회귀"
folder: "machinelearning"
categories:
    - machinelearning
tag:
    - [machine learning, logistic regression, multi-class classification, sigmoid function, softmax function]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 7-1. 럭키백의 확률
럭키백에 들어간 생선의 크기, 무게 등이 주어졌을 때, 7개 생선에 대한 확률을 출력해야 한다고 하자. 길이, 높이, 두께, 대각선 길이, 무게를 특성으로 사용할 수 있다고 한다. 사이킷런의 K-최근접 이웃 분류기로 클래스 확률을 계산할 수는 있다. 한번 확률을 출력해보자.

## 7-2. 데이터 준비
```python
import pandas as pd
fish = pd.read_csv('http://bit.ly/fish_csv_data')
fish.head()  # 처음 5개 행 출력
```
    (결과)  	Species	Weight	Length	Diagonal	Height	Width
            0	Bream	242.0	25.4	30.0	    11.5200	4.0200
            1	Bream	290.0	26.3	31.2	    12.4800	4.3056
            2	Bream	340.0	26.5	31.1	    12.3778	4.6961
            3	Bream	363.0	29.0	33.5	    12.7300	4.4555
            4	Bream	430.0	29.0	34.0	    12.4440	5.1340

판다스의 `raed_csv()` 함수는 csv 파일을 데이터프레임으로 변환한다. 데이터프레임은 판다스에서 제공하는 표 형식의 주요 데이터 구조이다. 넘파이 행렬처럼 열과 행으로 이루어져 있다. 데이터프레임은 통계와 그래프를 위한 메소드를 풍부하게 제공하며, 넘파이와 사이킷런과의 호환성이 좋다.

이제 Species 열에서 고유한 값을 추출해보자.
```python
print(pd.unique(fish['Species']))  # unique() 함수는 데이터의 고유값들에 어떤 종류가 있는지 알고 싶을 때 사용!
```
    (결과) ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']

Species 열을 타깃으로 만들고 나머지 5개를 열을 입력 데이터로 사용하자.
```python
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
print(fish_input[:5])
```
    (결과) [[242.      25.4     30.      11.52     4.02  ]
            [290.      26.3     31.2     12.48     4.3056]
            [340.      26.5     31.1     12.3778   4.6961]
            [363.      29.      33.5     12.73     4.4555]
            [430.      29.      34.      12.444    5.134 ]]

그 다음, 데이터를 훈련 세트와 테스트 세트로 나누고, 각 세트를 표준화 전처리 하자.
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

## 7-3. K-최근접 이웃 분류기의 확률 예측
최근접 이웃 개수인 k를 지정하여 K-최근접 이웃 분류기가 예측한 확률을 구해보자.
```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
```
    (결과) 0.8907563025210085
           0.85

일단 결과는 이렇게 나왔는데, 참고만 하고 점수에 대해서는 잊자.<Br/>
타깃 데이터 2개 이상의 클래스가 포함된 문제를 **다중 분류(Multi-class classification)** 라고 한다. 이진 분류를 사용했을 때는 양성 클래스와 음성 클래스를 각각 1과 0으로 지정하여 타깃 데이터를 만들었다. 다중 분류에서도 타깃값을 숫자로 바꾸어 입력할 수 있다. 그러나 사이킷런에서는 문자열로 된 타깃값을 그대로 사용할 수 있다! 이 때, 타깃값을 그대로 사이킷런 모델에 전달하면 순서가 자동으로 알파벳 순 정렬된다. 즉, `pd.unique(fish['Species'])`로 출력했던 순서와 다르다!
```python
print(kn.classes_)
```
    (결과) ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

테스트 세트에 있는 처음 5개 샘플의 타깃을 예측해보자.
```python
print(kn.predict(test_scaled[:5]))
print(test_target[:5])
```
    (결과) ['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']
           ['Perch' 'Smelt' 'Pike' 'Whitefish' 'Perch']

사이킷런의 분류 모델은 `predict_proba()` 메소드로 클래스별 확률값을 반환할 수 있다.
```python
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))  # decimals 매개변수로 유지할 소수점 아래 자릿수 지정 가능!
```
    (결과) [[0.     0.     1.     0.     0.     0.     0.    ]
            [0.     0.     0.     0.     0.     1.     0.    ]
            [0.     0.     0.     1.     0.     0.     0.    ]
            [0.     0.     0.6667 0.     0.3333 0.     0.    ]
            [0.     0.     0.6667 0.     0.3333 0.     0.    ]]

위 5개 데이터 중, 4번째 데이터와 가까운 이웃 샘플 3개에 대한 타겟을 보면 위 코드 결과가 맞다는 것을 확인할 수 있다.
```python
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])
```
    (결과) [['Roach' 'Perch' 'Perch']]

K-최근접 이웃은 결국 이웃 샘플 개수를 몇개로 설정하느냐에 따라 확률값이 고정되어 출력된다. 다른 방법은 없을까?

## 7-4. 로지스틱 회귀
**로지스틱 회귀**는 선형 회귀와 동일하게 선형 방정식을 학습하는 **분류 모델**이다.<br/>
$ z = a\times (Weight) + b\times (length) + c\times (Diagonal) + d\times (Height) + e\times (Width) + f $
여기에서 $a, b, c, d, e$는 가중치이다. $z$는 어떤 값도 가능하지만, 확률이 되려면 0~1 사이 값이 되어야 한다. $z$가 아주 큰 음수이면 0, 아주 큰 양수이면 1이 되도록 바꾸는 기능을 담당하는 함수가 바로 **시그모이드 함수(Sigmoid function)** 또는 **로지스틱 함수(Logistic function)**이다.<br/>
$ \phi = \frac{1}{1+e^{-z}} $ <br/>
z가 무한하게 큰 움수이면 이 함수는 0에 가까워지고, z가 무한하게 큰 양수가 되면 1에 가까워진다. z가 0이 될 때는 0.5가 되며, z가 어떤 값이 되더라도 $ \phi $는절대 0~1 사이의 범위를 벗어날 수 없다. 그래프를 시각화 하면 다음과 같다.
```python
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
```
![그림 7-1. 코드 결과](/assets/images/machinelearning/7-1.png)
{: .align-center}
그림 7-1. 코드 결과

이진 분류를 해보자. 시그모이드 함수 출력이 0.5보다 크면 양성 클래스, 0.5보다 작으면 음성 클래스로 판단하도록 만들자.

## 7-5. 로지스틱 회귀로 이진 분류 수행하기
넘파이 배열은 True, False 값을 전달하여 행을 선택할 수 있다. 이를 **불리언 인덱싱(Boolean indexing)** 라고 한다.
```python
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])
```
    (결과) ['A' 'C']

이를 이용해서, 도미와 빙어 클래스 데이터 및 라벨만 골라내도록 하자.
```python
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
```

이제 로지스틱 회귀 모델을 훈련하여 테스트셋 일부 결과를 출력해보자.

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))
print(lr.classes_)
print(lr.coef_, lr.intercept_)
```
    (결과) ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']
           [[0.99759855 0.00240145]
            [0.02735183 0.97264817]
            [0.99486072 0.00513928]
            [0.98584202 0.01415798]
            [0.99767269 0.00232731]]
           ['Bream' 'Smelt']
           [[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]

위 코드 결과에 따라 로지스틱 회귀 모델이 학습한 방정식은 다음과 같다.
$ z = -0.404\times (Weight) + -0.576\times (length) + -0.663\times (Diagonal) + 1.013\times (Height) + -0.732\times (Width) + -2.161 $<br/>

처음 5개 샘플에 위 식을 대입하여 나온 z값을 출력해보고, 이를 시그모이드 함수에도 넣어보자!
```python
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit  # 사이파이 모듈에서 제공하는 시그모이드 함수
print(expit(decisions))
```
    (결과) [-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]
           [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]

## 7-6. 로지스틱 회귀로 다중 분류 수행하기
LogisticRegression 클래스는 기본적으로 반복적인 알고리즘을 사용한다. max_iter 라는 매개변수에서 반복횟수를 지정하고, 기본값은 100이다. 또한, 릿지 회귀와 같은 계수의 제곱을 규제한다. 앞서 배운 alpha와 동일한 기능을 가진 규제를 제어하는 매개변수가, LogisticRegression 클래스에서는 C로 제공된다. 그러나 C는 alpha와는 반대로 값이 작을수록 규제가 커진다. C의 기본값은 1이다. 이 값을 조절하여 모델을 만들어보자!
```python
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```
    (결과) 0.9327731092436975
           0.925

처음 5개 샘플에 대한 예측과 예측확률도 출력해보자.
```python
print(lr.predict(test_scaled[:5]))
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))  # 소수점 네번째 자리에서 반올림
print(lr.classes_)
print(lr.coef_.shape, lr.intercept_.shape)
```
    (결과) ['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']
           [[0.    0.014 0.841 0.    0.136 0.007 0.003]
            [0.    0.003 0.044 0.    0.007 0.946 0.   ]
            [0.    0.    0.034 0.935 0.015 0.016 0.   ]
            [0.011 0.034 0.306 0.007 0.567 0.    0.076]
            [0.    0.    0.904 0.002 0.089 0.002 0.001]]
            ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
           (7, 5) (7,)

데이터의 특성이 5개이고, 클래스가 7개라 `(7, 5)` shape이 나왔다. 다중 분류는 클래스마다 z값을 하나씩 계산한다. 이진분류에서는 시그모이드 함수를 사용하여 z를 0과 1 사이의 값으로 변환했다. 다중 분류는 **소프트맥스(Softmax)** 함수를 사용하여 7개의 z값을 확률로 변환한다. 시그모이드 함수가 하나의 선형 방정식 출력값을 0~1 사이로 압축한다면, 소프트맥스 함수는 여러 개의 선형 방정식의 출력값을 0~1 사이로 압축하고 전체 합이 1이 되도록 만든다. 다음과 같은 식에 의해 말이다!

$ esum = e^{z1} + e^{z2} + e^{z3} + e^{z4} + e^{z5} + e^{z6} + e^{z7} $
$ s1=\frac{e^{z1}}{esum}, s2=\frac{e^{z2}}{esum}, ..., s7=\frac{e^{z7}}{esum} $

테스트 세트의 처음 5개 샘플에 대한 z1~z7을 구하고 소프트맥스 함수를 적용해보자!
```python
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

from scipy.spacial import sofrtmax
proba = softmax(decision, axis=1) # axis=1 로 해야 각 행에 대해 softmax 계산함
print(np.round(proba, decimals=3))
```
    (결과) [[ -6.5    1.03   5.16  -2.73   3.34   0.33  -0.63]
            [-10.86   1.93   4.77  -2.4    2.98   7.84  -4.26]
            [ -4.34  -6.23   3.17   6.49   2.36   2.42  -3.87]
            [ -0.68   0.45   2.65  -1.19   3.26  -5.75   1.26]
            [ -6.4   -1.99   5.82  -0.11   3.5   -0.11  -0.71]]
           [[0.    0.014 0.841 0.    0.136 0.007 0.003]
            [0.    0.003 0.044 0.    0.007 0.946 0.   ]
            [0.    0.    0.034 0.935 0.015 0.016 0.   ]
            [0.011 0.034 0.306 0.007 0.567 0.    0.076]
            [0.    0.    0.904 0.002 0.089 0.002 0.001]]