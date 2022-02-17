---
layout: single
title: "[Machine learning] 10. 교차 검증과 그리드 서치"
folder: "machinelearning"
categories:
    - machinelearning
tag:
    - [machine learning, validation set, cross validation, grid search, random search]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 10-1. 검증 세트
테스트 세트를 사용하지 않으면 모델이 과대적합인지 과소적합인지 판단하기가 어렵다. 테스트 세트를 사용하지 않고 이를 측정하려면 어떻게 해야 할까? 바로 훈련 세트를 나누는 것이다! 훈련 세트로부터 분리된 데이터를 **검증 세트(Validation set)** 라고 한다. 전체 데이터 중, 20%를 테스트 세트, 나머지 80%를 훈련 세트로 만든다. 그리고 이 훈련 세트 중, 다시 20%를 떼어 내어 검증 세트로 만든다. 훈련 세트에서 모델을 훈련하고 검증 세트로 모델을 평가한다. 그리고 나서 테스트하고 싶은 매개변수를 바꿔가며 가장 좋은 모델을 고른다. 그 다음, 해당 매개변수가 괜찮으면, 훈련 세트와 검증 세트를 합쳐 전체 훈련 데이터에서 모델을 다시 훈련한다. 마지막에 테스트 세트에서 최종 점수를 평가한다.
```python
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
print(sub_input.shape, val_input.shape)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
```
    (결과) (4157, 3) (1040, 3)
           0.9971133028626413
           0.864423076923077

위 코드에서는 훈련 세트에서 과대적합이 된 것으로 보인다. 매개변수를 바꿔서 더 좋은 모델을 찾아야한다.

## 10-2. 교차 검증
검증 세트를 만드느라 훈련 세트가 줄었다. 검증 세트를 조금만 떼어 놓자니, 검증 데어터가 부족해 검증 점수는 들쭉날쭉하고 불안정할 것이다. 이때 **교차 검증(Cross validation)**을 이용하면 안정적인 검증 점수를 얻고 훈련에 더 많은 데이터를 사용할 수 있다. 교차 검증은 훈련 세트에서 검증 세트를 여러 번 떼어 내어 평가하는 과정을 반복한다. 그리고 이 반복하여 얻은 점수들을 평균하여 최종 점수를 얻는다. 3-폴드 교차 검증을 예로 들면, 훈련 세트를 3부분으로 나누어 1부분씩 검증 세트로 만들고 3번의 검증 점수를 얻어 평균하면 된다. **k-폴드 교차 검증(K-fold cross validation)**은 이 훈련 세트를 k부분으로 나누어 점수를 k번 내고, 평균하여 점수를 얻는 것이다. 사이킷런에 `cross_validate()` 함수를 이용하여 교차 검증을 수행할 수 있다. 기본적으로 5-폴드 교차 검증을 수행한다. 참고로 `cross_val_score()` 함수도 있는데, 이 녀석은 `cross_validate()`의 결과에서 `test_score`값만 반환한다.
```python
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)
```
    (결과) {'fit_time': array([0.01099277, 0.00800276, 0.01399684, 0.01100063, 0.00998425]), 'score_time': array([0.00099945, 0.00099945, 0.0010035 , 0.00100112, 0.00099993]), 'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}

`fit_time`과 `score_time`은 모델을 훈련하는 시간과 검증하는 시간을 의미한다. `test_score`는 각 교차 검증의 점수이며, 이를 평균하면 최종 교초 검증 점수를 얻을 수 있다.
```python
import numpy as np
print(np.mean(scores['test_score']))
```
    (결과) 0.855300214703487

주의해야할 점은, `cross_validate()` 함수 적용시, 훈련 세트를 다시 훈련 세트와 검증 세트로 나눠서는 안된다는 것이다. 훈련 세트만 넣어주면 알아서 검증 세트를 계속 분할하여 테스트 해준다. 만약 교차 검증할 때 훈련 세트를 한번 섞어주고 싶다면, **분할기(Splitter)**를 지정해야 한다. 회귀 모델일 경우, **KFold 분할기**를 사용하고, 분류 모델일 경우, **StratifiedKFold**를 사용하여 타깃 클래스를 골고루 나눈다.
```python
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
```
    (결과) 0.855300214703487

만약 10-폴드 교차 검증을 수행하고 싶다면 다음과 같이 작성하자.
```python
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
```
    (결과) 0.8574181117533719

이제 테스트 세트를 사용하지 않고 교차 검증으로 좋은 모델을 고르자!

## 10-3. 하이퍼파라미터 튜닝
하이퍼파라미터는 모델이 학습할 수 없어서 사용자가 지정해야만 하는 파라미터를 의미한다. 보통 하이퍼파라미터는 모두 클래스나 메소드의 매개변수로 표현된다. 하이퍼파라미터를 튜닝할 때는 먼저 라이브러리가 제공하는 기본값을 그대로 사용하여 모델을 훈련한다. 그 다음 검증 세트의 점수나 교차 검증을 통해서 매개변수를 조금씩 바꿔나간다. 매개변수를 바꿔가며 모델을 훈련하고 교차 검증을 수행한다. 참고로 `AutoML` 이라는 기술이 있는데, 이는 사람의 개입 없이 하이퍼파라미터 튜닝을 자동으로 수행하는 기술을 말한다.

결정 트리를 예로 보자. `max_depth`를 최적값으로 고정하고 `min_samples_split`을 바꿔가며 최적의 값을 찾는다. 이렇게 한 매개변수의 최적값을 찾고 다른 매개변수의 최적값을 찾아도 될까? 아쉽게도 `max_depth`의 최적값은 `min_samples_split` 매개변수가 바뀔 경우 함께 달라진다. 즉, 두 매개변수를 동시에 바꿔가며 최적값을 찾아야 하는 것이다. 매개변수가 많을수록 문제는 더 복잡해진다. 사이킷런에서는 이런 경우를 대비해 **그리드 서치(Grid search)**를 제공한다. `GridSearchCV` 클래스는 하이퍼파라미터 탐색과 교차 검증을 한 번에 수행한다. `cross_validate()` 함수를 호출할 필요 없이 말이다.
```python
from sklearn.model_selection import GridSearchCV
params={'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}  # min_impurity_decrease 값 바꿔가며 총 5번 실행
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)  # cv 매개변수 기본값은 5, n_jobs로 병렬 실행에 사용할 CPU 코어 수 지정! 기본값은 1이며 -1 설정하면 사용 가능한 모든 코어 사용
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
```
    (결과) 0.9615162593804117

`best_params_` 속성에 최적의 매개변수 값 들어있다. 각 매개변수에서 수행한 교차 검증의 평균 점수는 `cv_results_` 속성의 'mean_test_score'키에 저장되어 있다.
```python
print(gs.best_params_)
print(gs.cv_results_['mean_test_score'])
```
    (결과) {'min_impurity_decrease': 0.0001}
           [0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]

`cv_results_` 속성의 'params' 키에는 각 매개변수의 값이 들어있다. 최고 교차 검증의 평균 점수를 낸 분류기의 파라미터 값을 확인해보자.
```python
best_index=np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
```
    (결과) {'min_impurity_decrease': 0.0001}

이제 과정을 정리해보자.<br/>
1). 탐색할 매개변수를 지정한다.<br/>
2). 훈련 세트에서 그리드 서치를 수행한다.<br/>
3). 최상의 평균 검증 점수가 나온 매개변수 조합을 찾는다.<br/>
4). 이 조합을 그리드 서치 객체에 저장한다.<br/>
5). 최상의 매개변수 값을 활용하여 전체 훈련 세트에 대한 최종 모델 훈련을 한다.<br/>
6). 이 모델도 그리드 서치 객체에 저장한다.<br/>

참고로 결정 트리에서 찾으려는 매개변수 `min_impurity_decrease`는 노드를 분할하기 위한 불순도 감소 최소량을 의미한다! `max_depth`는 트리의 최대 깊이이며, `min_samples_split`은 노드를 나누기 위한 최소 샘플 수이다. 이 3개의 매개변수 최적 조합도 찾아보자.
```python
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1), # range는 정수만 사용 가능
          'min_samples_split': range(2, 100, 10)
          } # 총 9 * 15 * 10 = 1350개 조합이며, 5-폴드 교차 검증도 하면 6750개임!

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))  # 교차 검증 점수
```
    (결과) {'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}
           0.8683865773302731

교차 검증은 원하는 매개변수 값을 나열하면 자동으로 교차 검증을 수행하여 최상의 매개변수를 찾아준다. 그런데 이렇게 매개변수에 간격을 둔 것은 사실 근거가 없다. 좀 더 넓은 간격으로 시도해 볼 순 없을까?

## 10-4. 랜덤 서치
매개변수 값이 수치이면 값의 범위나 간격을 정하기 어렵다. 매개 변수 조건도 많아 그리드 서치 수행 시간이 오래 걸릴 수도 있다. **랜덤 서치(Random search)**는 매개 변수 값 목록을 전달하지 않고 매개변수를 샘플링 할 수 있는 확률 분포 객체를 전달한다. 싸이파이로 2개의 확률 분포 클래스를 임포트 해보자. 참고로 싸이파이(Scipy)는 적분, 보간, 선형 대수, 확률 등 포함한 수치 계산 전용 라이브러리이다.
```python
from scipy.stats import uniform, randint  # 주어진 범위에서 각각 실수값, 정수값 고르게 뽑음
rgen = randint(0, 10)
rgen.rvs(10)
```
    (결과) array([4, 0, 6, 7, 2, 2, 7, 6, 9, 6])

숫자를 1000개 샘플링해서 각 숫자 개수를 세어보자
```python
np.unique(rgen.rvs(1000), return_counts=True)
```
    (결과) (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
           array([101,  96, 106,  92, 106,  94, 102,  91, 104, 108], dtype=int64))

uniform 클래스 사용법도 동일하다. 0~1 사이에서 10개 실수 추출해보자.
```python
ugen = uniform(0, 1)
ugen.rvs(10)
```
    (결과) array([0.8400478 , 0.29386554, 0.39775206, 0.77627505, 0.16128134,
       0.8968112 , 0.14322257, 0.38157404, 0.33622736, 0.0414059 ])

이제 싸이파이의 확률 분포 모듈을 활용하여 랜덤 서치를 수행해보자. 기존의 결정 트리 매개변수에 `min_samples_leaf`를 추가할 것이다. `min_samples_leaf`는 리프 노드가 되기 위한 최소 샘플의 개수를 의미한다. 즉 노드를 분할하여 자식 노드가 생길 때, 자식 노드가 갖는 샘플 개수가 이 개수보다 작으면 분할하지 않는다.
```python
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25)
          }
```

이제 사이킷런의 `RandomizedSearchCV` 클래스를 사용하여 랜덤 서치를 수행해보자.
```python
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42) # n_iter는 샘플링 횟수 의미
gs.fit(train_input, train_target)
```

총 100번 샘플링하여 교차 검증을 수행하고 최적의 매개변수 조합을 찾는다. 그리드 서치보다 교차 검증 수는 줄이면서 많은 영역을 효과적으로 탐색할 수 있다.
```python
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))
```
    (결과) {'max_depth': 39, 'min_impurity_decrease': 0.00034102546602601173, 'min_samples_leaf': 7, 'min_samples_split': 13}
           0.8695428296438884

최적 모델은 이미 전체 훈련 세트로 훈련되어 `best_estimator_` 속에 저장되어 있다. 이제 이 최적 모델로 테스트 세트의 성능을 확인해보자.
```python
dt = gs.best_estimator_
print(dt.score(test_input, test_target))
```
    (결과) 0.86