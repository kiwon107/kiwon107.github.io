---
layout: single
title: "[Machine learning] 8. 확률적 경사 하강법"
folder: "machinelearning"
categories:
    - machinelearning
tag:
    - [machine learning, stochastic gradient descent, loss function, epoch]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 8-1. 점진적인 학습
훈련데이터가 항상 한 번에 준비되면 너무 좋을 것이다. 그러나 현실은 녹록치 않다. 만약 데이터가 조금씩 꾸준히 전달되는 경우라면 어떻게 해야할까?<br/>
새로운 데이터를 추가할 때 마다 이전 데이터를 버려서 훈련 데이터 크기를 일정하게 유지한다면? 이 경우, 버린 데이터 안에 중요한 데이터가 들어있다면 큰일이 아닐 수 없다. 데이터를 버리지 않고 새로운 데이터에 대해 조금씩 더 훈련하는 방법! 이것이 바로 **점진적 학습** 또는 **온라인 학습**이라고 한다. 대표적인 점진적 학습 알고리즘은 **확률적 경사 하강법(Stochastic Gradient Descent)**이다.

## 8-2. 확률적 경사 하강법
경사는 기울기를 말한다. 하강법은 내려가는 방법을 말한다. 즉, 경사 하강법은 경사를 따라 내려가는 방법을 말한다. 경사 하강법에서 중요한 것은, 가장 가파른 길을 찾아서 조금씩 내려오는 것이다. 한번에 걸음이 너무 크면, 경사를 따라 내려가지 못하고 오히려 올라가는 경우가 발생할 수 있기 때문이다.<br/>
그럼 확률적이란 말은 무엇일까? 훈련 세트를 사용하여 모델을 훈련하므로, 경사 하강법도 당연히 훈련 세트를 사용해서 가장 가파른 길을 찾는다. 이때, 전체 샘플을 사용하지 않고 딱 하나의 샘플을 훈련 세트에서 랜덤하게 골라 가장 가파른 깅를 찾는 것도 한가지 방법이 될 수 있다. 이것이 바로 **확률적 경사 하강법**이다!

**확률적 경사 하강법**은 훈련 세트에서 랜덤하게 하나의 샘플을 선택하여 가파른 경사를 조금 내려간다. 그 다음 훈련세트에서 랜덤하게 또 다른 샘플을 하나 선택하여 경사를 조금 내려간다. 이런식으로 전체 샘플을 모두 사용할 때까지 계속한다. 모든 샘플을 다 사용했는데, 산을 못내려왔다면? 그럼, 훈련 세트의 모든 샘플을 다시 채워 넣는다. 그리고 다시 랜덤하게 하나의 샘플을 선택하여 계속 경사를 내려간다. 이렇게 훈련 세트를 한 번 모두 사용하는 과정을 **에포크(Epoch)** 라고 부른다. 보통 경사 하강법은 수십, 수백번 이상 에포크를 수행한다.

무작위로 샘플을 선택해서 산을 내려가는데, 과연 잘 찾아갈 수 있을까? 확률적 경사 하강법은 꽤 잘 작동한다. 하나의 샘플보다 여러 개의 샘플을 사용해서 경사 하강법을 수행하는 방식도 있다. 이러한 방식을 **미니배치 경사 하강법(Minibatch gradient descent)** 라고 한다. 실전에서 가장 많이 사용하는 방법이다.

극단적으로 한 번 경사로를 따라 이동하기 위해 전체 샘플을 사용하는 경우도 있다. 이를 **배치 경사 하강법(Batch gradient descent)** 라고 부른다. 전체 데이터를 사용하므로 가장 안정적이나, 그만큼 컴퓨터 자원을 많이 사용하게 된다. 심지어, 메모리가 무족해서 한 번에 전체 데이터를 모두 읽을 수 없는 경우도 발생한다.

이제 훈련 데이터가 모두 준비되어 있지 않고 매일매일 업데이트 되는 방식으로 제공되어도, 계속 학습을 이어나갈 수 있다. 즉, 산꼭대기에서부터 다시 시작할 필요가 없는 것이다!

이제 경사 하강법은 이해했다. 그렇다면 어떻게, 어디서 가파른 길을 찾아 내려가야할까? 바로 손실함수를 통해 알 수 있다!

## 8-3. 손실 함수
**손실 함수(Loss function)**는 머신러닝 알고리즘이 어떤 문제에 대해 얼마나 엉터리인지를 측정하는 기준이다! 이 손실 함수 값은 작을수록 좋다. 이 최적의 손실 함수 값을 찾기 위해 가파른 길로 산을 조금씩 이동하는 것이라 보면 된다. 참고로 손실 함수는 샘플 하나에 대한 손실을 의미한다. 반면, 비용 함수(Cost function)는 훈련 세트에 있는 모든 샘플에 대한 손실 함수의 합을 말한다.

이제 분류 기준에서 손실함수를 바라보자. 분류에서 손실함수는 정답을 못맞히는 것이다. 맞혔다, 못맞혔다 2가지 경우의 수로 손실함수를 정할 수는 없다. 경사 하강법으로 조금씩 산을 내려가려면 산의 경사면이 연속적이어야 한다! 그럼 손실 함수도 연속적이여야 한다는 의미가 된다! 맞혔다, 못맞혔다를 확률 0~1 사이의 값으로 나타내기 위해 **로지스틱 손실 함수**를 이용할 수 있다!

## 8-4. 로지스틱 손실 함수
**로지스틱 손실 함수(Logistic loss function)**는 다음과 같이 정의된다.
타깃=1 일 때, $로지스틱 손실 함수 = -log(예측 확률)$<br/>
타깃=0 일 때, $로지스틱 손실 함수 = -log(1-예측 확률)$<br/>
양성 클래스(타깃=1)일 때, 확률이 1에서 멀어질수록 손실은 아주 큰 양수가 된다. 그리고 음성 클래스(타깃=0)일 때, 예측 확률이 0에서 멀어질수록 손실은 아주 큰 양수가 된다.
다중 분류도 매우 비슷한 손실 함수를 사용한다. 다중 분류에서 사용하는 손실 함수를 **크로스엔트로피 손실 함수(Cross-entropy loss function)** 라고 부른다.<br/>
참고로, 회귀의 손실 함수로는, 평균 절대값 오차 또는 평균 제곱 오차를 많이 사용한다.

## 8-5. SGDClassifier
```python
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

이제 사이킷런에서 확률적 경사 하강법을 제공하는 분류 클래스 SGDClassifier를 사용하자. SGDClassifier 객체 생성시, 2개의 매개변수를 저장한다. loss는 손실 함수의 종류를 지정한다. loss는 'log'로 지정해보자. max_iter는 에포크 횟수를 의미한다. 10으로 지정하자.

```python
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```
    (결과) 0.773109243697479
           0.775
           c:\users\lg\appdata\local\programs\python\python37\lib\site-packages\sklearn\linear_model\_stochastic_gradient.py:577: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
           ConvergenceWarning)

친절하게도, 사이킷런이 모델이 충분히 수렴하지 않았다는 ConvergenceWarning 경고를 보내줬다. SGDClassifier 객체를 다시 만들지 않고, 이미 훈련시킨 모델 sc를 추가로 더 훈련해보자. 모델을 이어서 훈련할 때는 `partial_fit()` 메소드를 사용한다. fit 메소드와 사용법은 같지만, 호출할 때마다 1에포크씩 이어서 훈련할 수 있다.
```python
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```
    (결과) 0.8151260504201681
           0.825

아직 점수가 낮지만, 에포크를 추가로 수행하니, 정확도가 향상되었다. 얼마나 더 훈련시켜야 만족할만한 정확도를 얻을 수 있을까? 어떤 기준이 있으면 좋을 것 같은데?

참고로 SGDClassifier는 객체에 한 번에 훈련 세트 전체를 전달했지만, 훈련 세트에서 1개씩 샘플을 꺼내어 경사 하강법 단계를 수행한다. SGDClassifier는 미니배치 경사 하강법이나, 배치 경사 하강법을 제공하지 않는다.

## 8-6. 에포크와 과대/과소적합
확률적 경사 하강법을 사용한 모델은 에포크 횟수에 따라서 과소적합이나 과대적합이 될 수 있다. 에포크 횟수가 적으면, 모델이 훈련 세트를 덜 학습한다. 반대로 에포크 횟수가 충분히 많으면 훈련 세트를 완전히 학습할 것이다. 훈련 세트에 아무 잘 맞는 모델이 만들어지는 것이다. 즉, 적은 에포크 횟수 동안 훈련한 모델은 과소적합된 모델일 가능성이 높다. 반면 많은 에포크 횟수 동안 훈련한 모델은 훈련 세트에 너무 잘 맞아 과대적합된 모델일 가능성이 높다. 훈련 세트 점수는 에포크가 진행될 수록 계속 증가하지만, 테스트 세트 점수는 어느 순간 감소하기 시작한다. 이 지점이 바로 과대적합되기 시작하는 곳이다. 과대적합이 시작하기 전 훈련을 멈추는 것이 바로 **조기 종료(Early stopping)**이다.
```python
import numpy as np
sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(0, 300): ## _는 사용하지 않고, 그냥 버리는 값을 넣어두는 용도로 사용함! 300번의 에포크
    sc.partial_fit(train_scaled, train_target, classes=classes)  # partial_fit 메소드만 사용하려면, 해당 메소드에 전체 클래스의 레이블 전달해야함!
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```
![그림 8-1. 코드 결과](/assets/images/machinelearning/8-1.png)
{: .align-center}
그림 8-1. 코드 결과

그림을 보면, 백 번째 에포크 이후에는 훈련 세트와 테스트 세트 점수가 조금씩 벌어지는 것을 보인다. 반복 횟수를 100에 맞춰서 다시 모델을 훈련시켜보자.
```python
sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```
    (결과) 0.957983193277311
           0.925

`tol` 매개변수에서 향상될 최솟값을 지정하면, 일정 에포크 동안 성능이 최솟값 만큼 향상되지 않을 경우, 더 훈련하지 않고 자동으로 멈춘다. 여기서는 None으로 설정하여 멈추지 않고 에포크 100 횟수만큼 무조건 반복 훈련하도록 하였다. 최종 점수는 높아 보인다. 참고로 SGDRegressor은 확률적 경사 하강법을 사용한 회귀 알고리즘이다.

## 8-7. 힌지 손실 / 서포트 벡터 머신
사실 SGDClassifier 클래스의 loss 매개변수의 기본값은 'hinge' 이다. **힌지 손실(Hinge loss)** 또는 **서포트 벡터 머신(Support vector machine)**라 불리는 손실 함수이다. 저자가 이에 대한 추가 설명은 하지 않았지만 서포트 벡터 머신이 널리 사용된다는 점과 SGDClassifier가 여러 종류의 손실 함수를 loss 매개변수에 지정한다는 점을 강조하였다.
```python
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```
    (결과) 0.9495798319327731
           0.925