---
layout: single
title: "[Deeplearning(Tensorflow)] 3. 신경망 모델 훈련"
folder: "deeplearningtens"
categories:
    - deeplearningtens
tag:
    - [deep learning, dropout, callback, early stopping]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 3-1. 손실 곡선
`fit()` 메소드로 모델을 훈련할 때 훈련 과정이 상세하게 출력되었다. 에포크, 횟수, 손실, 정확도 등이 있었다. 출력의 마지막에 다음과 같은 메세지도 있었다.<br/>
<tensorflow.python.keras.callbacks.History at 0x18340a10e10><br/>
`fit()` 메소드는 `History` 클래스 객체를 반환한다. `History` 객체에는 손실과 정확도 값이 저장되어 있다. 이 값을 사용하여 그래프를 그려 보자.
```python
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
```

우선 패션 MNIST 데이터셋을 훈련 세트와 검증 세트로 나눴다. 그 다음 모델을 만드는 간단한 함수를 정의하자.
```python
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

model = model_fn()
model.summary()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=5, verbose=0) # verbose=1로 하면 에포크마다 진행 막대와 손실 등의 지표 출력, verbose=2면 진행 막대 빼고 출력함. verbose=0은 훈련 과정 안나타냄.
```
    (결과) Model: "sequential"
           _________________________________________________________________
           Layer (type)                 Output Shape              Param #   
           =================================================================
           flatten (Flatten)            (None, 784)               0         
           _________________________________________________________________
           dense (Dense)                (None, 100)               78500     
           _________________________________________________________________
           dense_1 (Dense)              (None, 10)                1010      
           =================================================================
           Total params: 79,510
           Trainable params: 79,510
           Non-trainable params: 0
           _________________________________________________________________
           
만약 `model_fn()` 함수에 케라스 층을 추가하면 은닉층 뒤에 또 하나의 층을 추가할 수 있도록 했다. history 객체에는 훈련 측정값이 담겨 있는 history 딕셔너리가 들어있다.
```python
print(history.history.keys())
```
    (결과) dict_keys(['loss', 'accuracy'])

보면 손실과 정확도가 포함되어 있다는걸 확인할 수 있다. 케라스는 기본적으로 에포크마다 손실을 계산한다. 정확도는 `compile()` 메소드에서 `metrics` 매개변수에 'accuracy'를 추가했기 때문에 history 속성에 포함되어 있다. 그래프로 손실값을 나타내보자.
```python
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```
![그림 3-1. 코드 결과](/assets/images/deeplearningtens/3-1.png)
{: .align-center}
그림 3-1. 코드 결과

정확도도 그래프로 나타내보자,
```python
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```
![그림 3-2. 코드 결과](/assets/images/deeplearningtens/3-2.png)
{: .align-center}
그림 3-2. 코드 결과

에포크마다 손실이 감소하고 정확도가 향상되었다는걸 확인할 수 있다. 이번에는 에포크를 20으로 늘려서 손실 그래프를 그려보자.
```python
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```
![그림 3-3. 코드 결과](/assets/images/deeplearningtens/3-3.png)
{: .align-center}
그림 3-3. 코드 결과

## 3-2. 검증 손실
에포크에 따른 과대적합과 과소적합을 파악하려면 훈련 세트에 대한 점수 뿐만 아니라, 검증 세트에 대한 점수도 필요하다. 참고로 신경망 모델이 최적화 하는 대상은 정확도가 아닌 손실 함수이다. 손실 함수에 비례하여 정확도가 높아지지 않는 경우도 있다. 따라서 모델이 잘 훈련되었는지 확인하려면 정확도보다는 손실 함수의 값을 확인하는 것이 더 낫다. `validation_data` 매개변수에 검증에 사용할 입력과 타깃값을 튜플로 만들어 전달해보자.
```python
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
print(history.history.keys())
```
    (결과) dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

검증 세트에 대한 손실값은 'val_loss'에 들어있다. 정확도는 'val_accuracy'에 있다. 그래프로 이 둘을 나타내보자.
```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```
![그림 3-4. 코드 결과](/assets/images/deeplearningtens/3-4.png)
{: .align-center}
그림 3-4. 코드 결과

전형적인 과대적합 모델이 만들어졌다. 검증 손실이 상승하는 시점을 뒤로 늦추면 검증 세트에 대한 손실이 줄어들고 정확도도 증가할 것이다. 옵티마이저 하이퍼파라미터를 조정하여 과대적합을 완화시켜보자. 옵티마이저를 `RMSprop`에서 `Adam`으로 변경해보자.
```python
model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```
![그림 3-5. 코드 결과](/assets/images/deeplearningtens/3-5.png)
{: .align-center}
그림 3-5. 코드 결과

과대적합이 훨씬 줄었다. 이는 `Adam` 옵티마이저가 이 데이터셋에 잘 맞는다는 것이다. 더 나은 손실 곡선을 얻으려면 학습률을 조정해서 다시 시도해 볼 수 있다.

## 3-3. 드롭아웃
**드롭아웃(Dropout)**은 일부 뉴런을 랜덤하게 꺼서 과대적합을 막는다. 뉴런은 랜덤하게 드롭아웃되고 얼마나 많은 뉴런을 드롭할지는 우리가 정해야 할 또 다른 하이퍼파라미터이다. 드롭아웃은 일부 뉴런이 랜덤하게 꺼지게 만들어 특정 뉴런에 과대하게 의존하는 것을 줄이고 모든 입력에 대해 주의를 기울이도록 한다. 케라스에는 드롭아웃을 `keras.layers` 패키지 안에 `Dropout` 클래스로 제공한다. 어떤 층의 뒤에 드롭아웃을 두어 이 층의 출력을 랜덤하게 0으로 만드는 것이다. 드롭아웃이 층처럼 사용되지만, 훈련되는 모델 파라미터는 없다.
```python
model = model_fn(keras.layers.Dropout(0.3))
model.summary()
```
    (결과) Model: "sequential_4"
           _________________________________________________________________
           Layer (type)                 Output Shape              Param #   
           =================================================================
           flatten_4 (Flatten)          (None, 784)               0         
           _________________________________________________________________
           dense_8 (Dense)              (None, 100)               78500     
           _________________________________________________________________
           dropout (Dropout)            (None, 100)               0         
           _________________________________________________________________
           dense_9 (Dense)              (None, 10)                1010      
           =================================================================
           Total params: 79,510
           Trainable params: 79,510
           Non-trainable params: 0
           _________________________________________________________________

보다시피 은닉층 뒤 추가된 드롭아웃 층은 훈련되는 모델 파라미터가 없다. 또한 입력과 출력의 크기도 동일하다. 일부 뉴런의 출력을 0으로 만들지만 전체 출력 배열의 크기를 바꾸지는 않는다. 여기서 중요한 점은 훈련이 끝난 뒤 평가나 예측을 수행할 때는 드롭아웃을 적용하지 말아야 한다는 것이다. 훈련된 모든 뉴런을 사용해야 올바른 예측이 가능하다. 텐서플로와 케라스는 모델을 평가와 예측에 사용시 자동으로 드롭아웃을 적용하지 않는다.
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```
![그림 3-6. 코드 결과](/assets/images/deeplearningtens/3-6.png)
{: .align-center}
그림 3-6. 코드 결과

과대적합이 확실히 줄었다는 걸 확인할 수 있다.

## 3-4. 모델 저장과 복원
```python
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=10, verbose=0, validation_data=(val_scaled, val_target))
```

케라스 모델은 훈련된 모델의 파라미터를 저장하는 간편한 `save_weights()` 메소드를 제공한다. 텐서플로의 체크포인트 포맷으로 저장하지만 파일의 확장자가 '.h5'일 경우 HDF5 포맷으로 저장한다.
```python
model.save_weights('model-weights.h5')
```

또한 모델 구조와 모델 파라미터를 함께 저장하는 `save()` 메소드도 제공한다. 텐서플로의 SavedModel 포맷으로 저장하지만 파일의 확장자가 '.h5'일 경우 HDF5 포맷으로 저장한다.
```python
model.save('model-whole.h5')
```

두 가지 테스트를 해보자. 첫 번째는 훈련하지 않은 새로운 모델을 만들고, model-weights.h5 파일에서 훈련된 모델 파라미터를 읽어서 사용한다. 두 번째는 model-whole.h5 파일에서 새로운 모델을 만들어 바로 사용한다.
```python
model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model-weights.h5')

import numpy as np
val_labels = np.argmax(model.predict(val_scaled), axis=-1)  # 배열의 마지막 차원을 따라 최대값 고름
print(np.mean(val_labels==val_target))
```
    (결과) 0.88225

`predict()` 메소드는 샘플마다 10개의 클래스에 대한 확률을 반환한다. 10개 확률 중 가장 큰 값의 인덱스를 골라 타깃 레이블과 비교하여 정확도를 계산한 결과가 바로 위 코드이다.
```python
model = keras.models.load_model('model-whole.h5')
model.evaluate(val_scaled, val_target)
```
    (결과) 375/375 [==============================] - 1s 2ms/step - loss: 0.3247 - accuracy: 0.8823
           [0.32472777366638184, 0.8822500109672546]

`load_model()` 함수는 모델 파라미터 뿐만 아니라 모델 구조와 옵티마이저 상태까지 모두 복원한다. 그러므로 `evaluate()` 메소드를 사용할 수 있다. 보다시피 `load_weights`로 부른 모델과 동일한 정확도를 보여준다.

## 3-5. 콜백
**콜백(Callback)**은 훈련 과정 중간에 어떤 작업을 수행할 수 있게 하는 객체로 `keras.callbacks` 패키지 아래에 있는 클래스들이다. 여기서 사용할 `ModelCheckpoint` 콜백은 최상의 검증 점수를 만드는 모델을 저장한다. 저장될 파일 이름을 'best-model.h5'로 지정하여 콜백을 적용해보자.
```python
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb])
```
    (결과) <tensorflow.python.keras.callbacks.History at 0x276f91c17f0>

`ModelCheckpoint` 클래스의 객체 `checkpoint_cb`를 만들고 `fit()` 메소드의 `callbacks` 매개변수에 리스트로 감싸서 전달했다. 모델이 훈련한 후, best-model.h5에 최상의 검증 점수를 낸 모델이 자동 저장된다. 이 함수를 다시 읽어서 예측을 수행해보자.
```python
model = keras.models.load_model('best-model.h5')
model.evaluate(val_scaled, val_target)
```
    (결과) 375/375 [==============================] - 1s 2ms/step - loss: 0.3265 - accuracy: 0.8837
           [0.32645875215530396, 0.8836666941642761]

`ModelCheckpoint` 콜백이 가장 낮은 검증 점수의 모델을 자동으로 저장해주었다. 20번 에포크 전체를 거치지 않고 검증 점수가 상승하기 시작할 때 훈련을 중지하도록 하여 컴퓨터 자원과 시간을 아끼는 방법은 없을까? 이렇게 과대적합이 시작되기 전 훈련을 미리 중지하는 것을 **조기 종료(Early stopping)** 라고 한다. 조기 종료는 훈련 에포크 횟수를 제한하는 역할이지만 모델이 과적합되는 것을 막아주어 규제 방법 중 하나로 볼 수 있다. 케라스는 `EarlyStopping` 콜백을 제공해준다.
```python
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
print(early_stopping_cb.stopped_epoch)
```
    (결과) 10

`patience=2`로 지정하여 2번 연속 검증 점수가 향상되지 않으면 훈련을 중지하도록 한다. 또한 `restore_best_weights=True`로 지정하여 가장 낮은 검증 손실의 모델을 파일에 저장하고 검증 손실이 다시 상승할 때 훈련을 중지할 수 있다. 훈련 중지 후 현재의 모델 파라미터를 최상의 파라미터로 되돌린다. 훈련을 마치고 몇 번째 에포크에서 훈련이 중지되었는지는 `early_stopping_cb` 객체의 `stopped_epoch` 속성에서 확인가능 하다.

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```
![그림 3-7. 코드 결과](/assets/images/deeplearningtens/3-7.png)
{: .align-center}
그림 3-7. 코드 결과

조기 종료 기법을 사용하면 안심하고 에포크 횟수를 크게 지정해도 좋다. 조기 종료로 얻은 모델을 사용하여 검증 세트에 대한 성능을 확인해 보자.
```python
model.evaluate(val_scaled, val_target)
```
    (결과) 375/375 [==============================] - 1s 2ms/step - loss: 0.3241 - accuracy: 0.8811
           [0.3240998089313507, 0.8810833096504211]