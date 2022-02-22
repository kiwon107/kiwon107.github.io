---
layout: single
title: "[Machine learning] 11. 트리의 앙상블"
folder: "machinelearning"
categories:
    - machinelearning
tag:
    - [machine learning, ensemble learning, random forest extra tree, gradient boosting]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 11-1. 정형 데이터와 비정형 데이터
csv, DB, 혹은 엑셀 같이 특성별로 정리된 형태의 데이터를 **정형 데이터(Structured data)** 라고 한다. 정형 데이터와 반대되는 데이터를 **비정형 데이터(Unstructured data)** 라고 한다. 책의 글과 같은 텍스트 데이터나 사진, 디지털 음악 등이 여기에 해당한다. 참고로 텍스트나 사진 같은 비정형 데이터도 DB에 저장할 수는 있다. NoSQL DB가 그 예이다.<br/>
두가지 유형의 데이터 중, 정형 데이터를 다루는데 뛰어난 성과를 내는 알고리즘이 바로 **앙상블 학습(Ensemble learning)**이다. 이 알고리즘은 대부분 결정 트리를 기반으로 만들어져 있다. 비정형 데이터는 규칙성을 찾기 어려워 신경망 알고리즘을 활용해야 한다.

## 11-2. 랜덤 포레스트
**랜덤 포레스트(Random forest)**는 결정 트리를 랜덤하게 만들어 결정 트리의 숲을 만든다. 그리고 각 결정 트리의 예측을 사용해 최종 예측을 만든다. 먼저 랜덤 포레스트는 각 트리를 훈련하기 위한 데이터를 랜덤하게 만든다. 이 데이터를 만들 때는, 입력한 훈련 데이터에서 랜덤하게 샘플을 추출하여 훈련 데이터를 만든다. 이 때 한 샘플이 중복되어 추출될 수도 있다! 이렇게 중복으로 랜덤하게 일정 개수를 뽑은 샘플을 **부트스트랩 샘플(Bootstrap sample)**라고 한다. 기본적으로 부트스트랩 샘플은 훈련 세트의 크기와 같게 만든다. 다만 중복하여 샘플을 뽑으므로 모든 샘플이 활용되지는 않는다. 참고로 부트스트랩 방식은 데이터 세트에서 중복을 허용하여 데이터를 샘플링하는 방식을 의미한다.<br/>
노드 분할은 어떻게 이루어질까? 노드를 분할할 때, 전체 특성 중 일부 특성을 무작위로 고르고 이 중 최선의 분할을 찾는다. `RandomForestClassifier`는 기본적으로 전체 특성 개수의 제곱근만큼의 특성을 선택한다. 즉, 4개의 특성이 있다면 노드마다 2개를 랜덤하게 선택하여 사용하는 것이다. 다만 `RandomForestRegressor`는 전체 특성을 사용한다. 사이킷런의 랜덤 포레스트는 기본적으로 100개의 결정 트리를 이런 방식으로 훈련한다. 그 다음, 분류일 경우 각 트리의 클래스별 확률을 평균하여 가장 높은 확률을 가진 클래스를 예측으로 삼는다. 회귀일 경우 각 트리의 예측을 평균한다.<br/>
랜덤 포레스트는 랜덤하게 선택한 샘플과 특성을 사용하므로 훈련 세트에 과적합되는 것을 막아주고 검증 세트와 테스트 세트에서 안정적인 성능을 얻을 수 있다.
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
wine = pd.read_csv('http://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, random_state=42)  # 기본으로 100개 트리 사용. n_jobs=-1 지정하여 모든 CPU 코어 사용!
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```
    (결과) 0.9973541965122431 0.8905151032797809

참고로 `return_train_score` 매개변수를 True로 지정하면 검증 점수뿐만 아니라 훈련 세트에 대한 점수도 같이 반환한다.
출력된 결과를 보면 다소 과대적합된 것으로 보인다.<br/>
랜덤 포레스트는 결정 트리의 앙상블이므로 `DecisionTreeClassifier`가 제공하는 중요한 매개변수를 모두 제공한다. `criterion`, `max_depth`, `max_features`, `min_samples`, `min_samples_split`, `min_impurity_decrease`, `min_samples_leaf` 등이 바로 그것이다. 또한 특성 중요도를 계산하여 제공한다. 랜덤 포레스트의 특성 중요도는 각 결정 트리의 특성 중요도를 취합한 것이다.
```python
rf.fit(train_input, train_target)
print(rf.feature_importances_)
```
    (결과) [0.23167441 0.50039841 0.26792718]

참고로 9장 결정 트리에서 산출한 특성 중요도는 [0.12345626 0.86862934 0.0079144 ] 였다. 특성 중요도가 달라진 이유는 랜덤 포레스트가 특성의 일부를 랜덤하게 선택하여 결정 트리를 훈련하였기 때문이다. 그 결과 하나의 특성에 과도하게 집중하지 않고 좀 더 많은 특성이 훈련에 기여할 기회를 얻었다. 이는 과대 적합을 줄이고 일반화 성능을 높이는데 도움이 된다.

`RandomForestClassifier`에는 재미있는 기능이 있다. 부트스트랩 샘플에 포함되지 않고 남은 샘플을 활용하여, 부트스트랩 샘플로 훈련한 결정 트리를 평가할 수 있다. 부트스트랩 샘플에 포함되지 않고 남는 샘플을 **OOB(Out of bag)** 라고 하는데, 마치 검증 세트의 역할을 하게 된다. 이 점수를 얻으려면 `oob_score` 매개변수를 True로 지정해야 한다. 이렇게 하면 랜덤 포레스트가 각 결정 트리의 OOB 점수를 평균하여 출력한다.
```python
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print(rf.oob_score_)
```
    (결과) 0.8934000384837406

## 11-3. 엑스트라 트리
**엑스트라 트리(Extra trees)**는 랜덤 포레스트와 마찬가지로 기본 100개의 결정 트리를 훈련한다. 또한 랜덤 포레스트와 동일한 매개변수를 지원한다. 전체 특성 중, 일부 특성을 랜덤하게 선택하여 노드를 분할하는 것도 동일하다. 다만, 부트스트랩 샘플을 사용하지 않는 다는 것이 유일한 차이이다. 즉, 결정 트리를 만들 때 전체 훈련 세트를 사용한다는 것이다. 대신 노드를 분할할 때는 가장 좋은 분할을 찾는 것이 아닌 무작위 분할을 수행한다. 하나의 결정 트리에서 특성을 무작위로 분할하면 성능은 낮아진다. 하지만 많은 트리를 앙상블 하므로 과대적합을 막고 검증 세트의 점수를 높힐 수 있다.
```python
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```
    (결과) 0.9974503966084433 0.8887848893166506

보통 엑스트라 트리가 무작위성이 좀 더 크므로, 랜덤 포레스트보다 더 많은 결정 트리를 훈련해야 한다. 하지만 랜덤하게 노드를 분할하므로 빠른 연산 속도를 제공한다는 것이 엑스트라 트리의 장점이라 할 수 있겠다. 참고로 걸정 트리는 최적의 분할을 찾는데 많은 시간이 소요된다. 고려해야 할 특징 개수가 많을 수록 더욱 그렇다. 만약 무작위로 나눈다면 훨씬 빨리 트리를 구성할 수 있을 것이다.<br/>
엑스트라 트리도 특성 중요도를 제공한다.
```python
et.fit(train_input, train_target)
print(et.feature_importances_)
```
    (결과) [0.20183568 0.52242907 0.27573525]

## 11-4. 그레이디언트 부스팅
**그레이디언트 부스팅(Gradient boosting)**은 깊이가 얕은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블 하는 방법이다. `GradientBoostingClassifier`는 기본적으로 깊이가 3인 결정 트리를 100개 사용한다. 깊이가 얕은 결정 트리를 사용하므로 과대적합에 강하고 일반적으로 높은 일반화 성능을 기대할 수 있다. 이 녀석은 경사 하강법을 이용하여 트리를 앙상블에 추가한다. 이미 배웠듯이 경사 하강법은 손실 함수가 가장 낮은 곳을 찾아 내려오는 방법이다. 분류에서는 로지스틱 손실 함수를 사용하며, 회귀에서는 평균 제곱 오차 함수를 사용한다. 그래디언트 부스팅은 결정 트리를 계속 추가하면서 손실 함수가 가장 낮은 곳을 찾아 이동한다. 손실 함수가 낮은 곳으로 천천히 조금씩 이동해야하므로 깊이가 얕은 트리를 사용한다. 또한 학습률 매개변수로 속도를 조절한다.
```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```
    (결과) 0.8881086892152563 0.8720430147331015

보다시피 거의 과대적합이 되지 않았다. 그레이디언트 부스팅은 결정 트리의 개수를 늘려도 과대적합에 매우 강하다. 학습률을 증가시키고 트리의 개수도 늘려 성능을 조금 더 향상시켜보자.
```python
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```
    (결과) 0.9464595437171814 0.8780082549788999

결과를 보면 결정 트리 개수를 5배나 늘렸음에도 과대적합을 잘 억제하고 있음을 확인할 수 있다.<br/>
그래디언트 부스팅도 특성 중요도를 제공한다.
```python
gb.fit(train_input, train_target)
print(gb.feature_importances_)
```
    (결과) [0.15872278 0.68011572 0.16116151]

그레이디언트 부스팅에는 특이한 매개변수가 있다. 바로 `subsample` 이라는 녀석인데, 트리 훈련에 사용할 훈련 세트의 비율을 정하는 매개변수이다. 기본값은 1.0이며, 이렇게 설정되면 전체 룬련 세트를 사용한다. `subsample`이 1보다 작으면 훈련 세트의 일부를 사용한다. 마치 경사 하강법 단계마다 일부 샘플을 랜덤하게 선택하여 진행하는 미니배치 경사 하강법과 비슷하다 할 수 있겠다.

일반적으로 그레이디언트 부스팅이 랜덤 포레스트보다 조금 더 높은 성능을 보여준다. 그러나 순서대로 트리를 추가하기 때문에 훈련속도가 느리다는 단점이 있다. 이에 따라, `GradientBoostingClassifier`에는 `n_jobs` 매개변수가 없다.

## 11-5. 히스토그램 기반 그레이디언트 부스팅
**히스토그램 기반 그레이디언트 부스팅(Histogram-based Gradient boosting)**은 정형 데이터를 다루는 머신러닝 알고리즘 중에 가장 인기가 높은 알고리즘이다. 히스토그램 기반 그레이디언트 부스팅은 먼저 입력 특성을 256개의 구간으로 나누어 노드 분할 시 최적의 분할을 매우 빠르게 찾을 수 있도록 한다. 256개 구간 중 하나를 떼어 놓고 누락된 값을 위해 사용한다. 따라서 입력에 누락된 특성이 있어도 이를 따로 전처리할 필요가 없다. `HistGradientBoostingClassifier`는 기본 매개변수에서 안정적인 성능을 얻을 수 있다. `n_estimators` 대신 `max_iter`로 부스팅 반복 횟수를 지정하여 트리의 개수를 정한다.
```python
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```
    (결과) 0.9321723946453317 0.8801241948619236

과대적합은 잘 억제하면서 그레이디언트 부스팅보다 조금 더 높은 성능을 보임을 확인할 수 있다. `permutation_importance()` 함수를 사용하면 특성 중요도도 확인할 수 있다. 이 함수는 특성을 하나씩 랜덤하게 섞어서 모델의 성능이 변화하는것을 관찰하여 어떤 특성이 중요한지를 계산한다. 훈련 세트 뿐만 아니라 테스트 세트에도 적용할 수 있고 사이킷런에서 제공하는 추정기 모델에 모두 사용할 수도 있다. `n_repeats` 매개변수는 랜덤하게 섞을 횟수를 의미한다. 기본값은 5이지만 이걸 10으로 지정해보자.
```python
from sklearn.inspection import permutation_importance
hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)
```
    (결과) [0.08876275 0.23438522 0.08027708]

`permutation_importance()` 함수가 반환하는 객체는 특성 중요도, 평균, 표준편차를 담고 있다. 테스트 세트에서 특성 중요도 평균을 출력해보자.
```python
result=permutation_importance(hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)
```
    (결과) [0.05969231 0.20238462 0.049     ]

`HistGradientBoostingClassifier`를 사용하여 테스트 세트에서의 성능을 최종적으로 확인해보자.
```python
hgb.score(test_input, test_target)
```
    (결과) 0.8723076923076923

사이킷런 외에도 히스토그램 기반 그레이디언트 부스팅 알고리즘을 구현한 라이브러리가 있다. 대표적인 라이브러리가 `XGBoost`인데, 사이킷런의 `cross_validate()` 함수와 함께 사용이 가능하다. `XGBoost`는 다양한 부스팅 알고리즘을 지원한다. `tree_method` 매개변수를 'hist'로 지정하면 히스토그램 기반 그레이디언트 부스팅을 사용할 수 있다.
```python
from xgboost import XGBClassifier
xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```
    (결과) 0.9555033709953124 0.8799326275264677

또한 마이크로소프트에서 만든 `LightGBM` 이라는 라이브러리도 있다. 빠르고 최신 기술을 많이 적용하여 인기가 점점 높아지고 있다.

```python
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```
    (결과) 0.935828414851749 0.8801251203079884

참고로 사이킷런의 히스토그램 기반 그레이디언트 부스팅은 LightGBM 영향을 많이 받았다고 한다.