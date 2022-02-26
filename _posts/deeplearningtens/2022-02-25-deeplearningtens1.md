---
layout: single
title: "[Deeplearning(Tensorflow)] 1. 인공 신경망"
folder: "deeplearningtens"
categories:
    - deeplearningtens
tag:
    - [deep learning, artificial neural network, tensorflow, dense layer, one-hot encoding]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 1-1. 패션 MNIST
머신러닝에서 붓꽃 데이터셋이 유명하다면, 딥러닝에서는 MNIST 데이터셋이 유명하다. 이 데이터는 손으로 쓴 0~9까지의 숫자로 이루어져 있다. 텐서플로를 사용하여 이 데이터를 불러올 수 있다. 텐서플로의 케라스 패키지를 임포트하고 패션 MNIST 데이터를 다운로드하자.
```python
from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)
```
    (결과) (60000, 28, 28) (60000,)
           (10000, 28, 28) (10000,)

어떤 이미지인지 시각화하여 확인해보자.
```python
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()
```
![그림 1-1. 코드 결과](/assets/images/deeplearningtens/1-1.png)
{: .align-center}
그림 1-1. 코드 결과

각 그림의 타겟이 무엇인지도 확인해보자.
```python
print([train_target[i] for i in range(10)])
```
    (결과) [9, 0, 0, 3, 0, 2, 7, 2, 5, 5]

패션 MNIST의 타깃은 0~9까지의 숫자 레이블로 구성된다. 10개 레이블의 의미는 다음과 같다.<br/>
 - 0: 티셔츠<br/>
 - 1: 바지<br/>
 - 2: 스웨터<br/>
 - 3: 드레스<br/>
 - 4: 코트<br/>
 - 5: 샌달<br/>
 - 6: 셔츠<br/>
 - 7: 스니커즈<br/>
 - 8: 가방<br/>
 - 9: 앵클 부츠<br/>

각 레이블 당 샘플 개수는 다음과 같다.
```python
import numpy as np
print(np.unique(train_target, return_counts=True))
```
    (결과) (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))

## 1-2. 로지스틱 회귀로 패션 아이템 분류하기
훈련 샘플이 60,000개나 되므로, 전체 데이터를 한꺼번에 사용하기보다, 샘플을 하나씩 꺼내서 모델을 훈련시키는게 더 효율적이다. `SGDClassifier` 클래스의 loss 매개변수를 'log'로 지정하여 로지스틱 손실 함수를 최소화하는 확률적 경사 하강법 모델을 만들자. 특성마다 값의 범위를 동일하게 하여 올바르게 손실 함수의 경사를 내려올 수 있도록 0~1 사이의 값으로 정규화를 하자. 그 다음, 2차원 형태를 1차원으로 변환하자.
```python
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
print(train_scaled.shape)
```
    (결과) (60000, 784)

이제 `SGDClassifier` 클래스와 `cross_validate` 함수로 이 데이터에서 교차 검증 성능을 확인해보자.
```python
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))
```
    (결과) 0.8192833333333333

뭔가 결과가 기대치에 못미치는듯 하다.

로지스틱 회귀 공식에 따라 패션 MNIST 데이터에 맞게 변형한다면 다음과 같을 것이다.
$z_티셔츠 = w1 \times (픽셀1) + w2 \times (픽셀2) + ... + w784 \times (픽셀784) + b$<br/>
두번째 레이블은 다음과 같을 것이다<br/>
$z_바지 = w1' \times (픽셀1) + w2' \times (픽셀2) + ... + w784' \times (픽셀784) + b'$<br/>
보다시피 바지에 대한 출력을 계산하기 위해 가중치와 절편은 다른 값을 사용하였다. 티셔츠와 같은 가중치를 사용하면 바지와 티셔츠를 구분할 수 없을 것이기 때문이다. 10개의 방정식에 대한 모델 파라미터를 찾고나서, z_티셔츠와 z_바지와 같이 10개의 클래스에 대한 선형 방정식을 모두 계산한다. 그리고 소프트 맥스 함수를 통과하여 각 클래스에 대한 확률을 얻는다.

## 1-3. 인공 신경망
가장 기본적인 **인공 신경망은(Artificial Neural Network)** 확률적 경사하강법을 사용하는 로지스틱 회귀와 같다. $z_{1}$ ~ $z_{10}$ 을 계산하고 이를 바탕으로 클래스를 예측한다. **출력층(Output layer)**이라고 불리우는 곳에서 신경망의 최종 값을 만들어진다. 인공 신경망에서는 z값을 계산하는 단위를 **뉴런(Neuron)** 이라고 부른다. 하지만 뉴런에서 일어나는 일은 선형 계산이 전부이다. 인공 신경망의 입력인 $x_{1}$ ~ $x_{784}$ 까지를 **입력층(Input layer)**라고 부른다. 여기서 입력층은 픽셀값 자체를 의미하며, 특별한 계산을 수행하지는 않는다. 절편은 뉴런마다 하나씩이므로 순서대로 $b1$과 $b2$로 나타낸다.<br/>

생물학적 뉴런은 수상 돌기로부터 신호를 받아 세포체에 모은다. 신호가 어떤 임곗값에 도달하면 축삭 돌기를 통하여 다른 세포에 신호를 전달한다. 인공 신경망은 이러한 인공 뉴런과 굉장히 비슷하다 할 수 있다.

## 1-4. 텐서플로와 케라스
텐서플로는 구글이 2015년 11월 오픈소스로 공개한 딥러닝 라이브러리이다. 2019년 9월, 텐서플로 2.0이 릴리스 되었다. 다음과 같이 간단히 임포트하여 사용가능하다.
```python
import tensorflow as tf
```

텐서플로는 저수준 API와 고수준 API가 있다. 케라스가 텐서플로의 고수준 API이다. 케라스는 2015년 3월 프랑소와 숄레가 만든 딥러닝 라이브러리이다. 딥러닝 라이브러리가 머신러닝 라이브러리와 다른점은, GPU를 사용하여 인공 신경망을 훈련한다는 것이다. GPU는 벡터와 행렬 연산에 매우 최적화 되어있어, 곱셈과 덧셈이 많이 수행되는 인공 신경망에 큰 도움이 된다.<br/>
케라스 라이브러리는 직접 GPU 연산을 수행하지 않고, GPU 연산을 수행하는 다른 라이브러리를 백엔드로 사용한다. 텐서플로가 케라스의 백엔드 중 하나이다. 씨아노, CNTK와 같은 여러 딥러닝 라이브러리를 케라스 백엔드로 사용할 수 있다. 프랑소와가 구글에 합류하면서 텐서플로 라이브러리에 케라스 API가 내장되었다. 텐서플로 2.0부터 케라스 API를 남기고 나머지 고수준 API를 모두 정리했다. 다양한 백엔드를 지원했던 멀티-백엔드 케라스는 2.3.1 버전 이후 더 이상 개발되지 않았다.
```python
from tensorflow import keras
```

## 1-4. 인공 신경망으로 모델 만들기
로지스틱 회귀에서는 교차 검증을 사용하여 모델을 평가했다. 그러나 인공 신경망에서는 교차 검증을 잘 사용하지 않고 검증 세트를 별도로 덜어내어 사용한다. 이렇게 하는 이유는 딥러닝 분야의 데이터 셋이 충분이 커서 검증 점수가 안정적이기 때문이다. 또한 교차 검증을 수행하기에는 훈련 시간이 너무 오래 걸리기도 하다. 패션 MNIST 데이터셋이 그만큼 크지는 않지만, 관례를 따라 검증 세트를 나누어보자.
```python
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)
```
    (결과) (48000, 784) (48000,)
           (12000, 784) (12000,)

이제 10개의 패션 아이템을 분류하기 위해 10개의 뉴런으로 한 층을 구성해보자. 케라스의 레이어(`keras.layers`) 패키지 안에는 다양한 층이 있다. 가장 기본이 되는 층은 **밀집층(Dense layer)**이다. 이런 층을 양쪽의 뉴런이 모두 연결하므로 **완전 연결층(Fully connected layer)**라고도 한다.
```python
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
```

첫번째 매개변수는 뉴런 개수이다. 10개로 지정하여 10개의 패션 아이템이 분류되도록 했다. 10개의 뉴런에서 출력되는 값을 확률로 바꾸기 위해 소프트맥스 함수를 사용한다. 케라스 층에서는 `activation` 매개변수에 이 함수를 지정한다. 이제 이 밀집층을 가진 신경망 모델을 만들어야한다. 케라스의 `Sequential` 클래스를 사용한다.
```python
model = keras.Sequential(dense)
```

Sequential 클래스의 객체를 만들 때 앞에서 만든 밀집층의 객체 dense를 전달했다. 이제 model 객체가 신경망 모델이다! 절편이 뉴런마다 더해진다는 점은 꼭 기억하자! 참고로 소프트맥스처럼 뉴런의 선형 방정식 계산 결과에 적용되는 함수를 **활성화 함수(Activation function)**라고 한다.

## 1-5. 인공 신경망으로 패션 아이템 분류하기
케라스 모델은 훈련하기 전에 설정 단계가 있다. model 객체의 `compile()` 메소드에서 수행한다. 꼭 지정해야할 것은 손실 함수의 종류이다.
```python
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
```

이진 분류에서는 이진 크로스 엔트로피(`binary_crossentropy`), 다중 분류에서는 크로스 엔트로피(`categorical_crossentropy`) 손실 함수를 사용한다. 이진 분류에서는 출력 뉴런이 오직 양성 클래스에 대한 확률(a)만 출력한다. 음성 클래스에 대한 확률은 간단히 1-a로 구할 수 있다. 다중 분류에서는 n개 클래스에 대한 확률을 먼저 출력한다. 각 클래스에 대한 확률이 모두 출력되므로, 타깃에 해당하는 확률만 남기고 나머지 확률에는 모두 0을 곱한다. 이처럼 타깃값을 해당 클래스만 1로 남기고 나머지는 모두 0인 배열로 만드는 작업이 **원-핫 인코딩(One-hot encoding)**라고 부른다. 따라서 다중 분류에서 크로스 엔트로피 손실 함수를 사용하려면 0, 1, 2와 같은 정수로 된 타깃값을 원-핫 인코딩으로 변환해야 한다.
```python
print(train_target[:10])
```
    (결과) [7 3 5 8 6 9 3 3 9 9]

모두 정수로 되어있는데, 텐서플로에서는 정수로 된 타깃값을 원-핫 인코딩으로 바꿀 필요는 없다. 정수로된 타깃값을 사용하여 크로스 엔트로피 손실을 계산하는 것이 `sparse_categorical_crossentropy`이다. 빽빽한 배열이 아닌 정수값만 사용한다는 의미에서 `sparse`라는 이름이 붙었다. 타깃값을 원-핫 인코딩으로 준비했다면 `compile()` 메소드에서 손실 함수를 `loss=categorical_crossentropy`로 지정하면 된다.<br/>
정확도도 함께 출력하기 위해, `metrics` 매개변수에 정확도 지표를 의미하는 `accuracy`를 지정하자. 이제 준비는 다 됐다. 모델을 훈련하자.
```python
model.fit(train_scaled, train_target, epochs=5)
```
    (결과) Epoch 1/5
           1500/1500 [==============================] - 2s 1ms/step - loss: 0.6098 - accuracy: 0.7927
           Epoch 2/5
           1500/1500 [==============================] - 3s 2ms/step - loss: 0.4805 - accuracy: 0.8399
           Epoch 3/5
           1500/1500 [==============================] - 3s 2ms/step - loss: 0.4574 - accuracy: 0.8459
           Epoch 4/5
           1500/1500 [==============================] - 2s 2ms/step - loss: 0.4451 - accuracy: 0.8528
           Epoch 5/5
           1500/1500 [==============================] - 3s 2ms/step - loss: 0.4369 - accuracy: 0.8537
           <tensorflow.python.keras.callbacks.History at 0x1f2356a50f0>

케라스에서 성능을 평가하는 메소드는 `evaluate()` 이다.
```python
model.evaluate(val_scaled, val_target)
```
    (결과) [0.447228342294693, 0.8525833487510681]
