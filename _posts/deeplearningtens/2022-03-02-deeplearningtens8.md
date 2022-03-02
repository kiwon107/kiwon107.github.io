---
layout: single
title: "[Deeplearning(Tensorflow)] 8. 순환 신경망으로 IMDB 리뷰 분류하기"
folder: "deeplearningtens"
categories:
    - deeplearningtens
tag:
    - [deep learning, corpus, token, one-hot encoding, word embedding]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "혼자 공부하는 머신러닝+딥러닝" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 8-1. IMDB 리뷰 데이터셋
IMDB 리뷰 데이터셋은 유명한 인터넷 영화 데이터베이스인 imdb.com 에서 수집한 리뷰를 감상평에 따라 긍정과 부정으로 분류해 놓은 데이터셋이다. 참고로 자연어 처리는 컴퓨터를 사용해 인간의 언어를 처리하는 분야인데, 음성 인식, 기계 번역, 감성 분석 등이 자연어 처리 분야의 대표적인 세부 분야이다. IMDB 리뷰를 감상평에 따라 분류하는 것은 감성 분석에 해당하며, 자연어 처리 분야에서 훈련 데이터를 종종 **말뭉치(Corpus)**라고 부른다. IMDB 리뷰 데이터셋은 하나의 말뭉치이다.

컴퓨터에서 처리하는 모든 것은 어떤 숫자 데이터이므로, 텍스트 자체를 신경망에 전달하지는 않는다. 텍스트 데이터의 경우 단어를 숫자 데이터로 바꾸는 일반적인 방법은 등장하는 단어마다 고유한 정수를 부여하는 것이다. 예를 들어, 'He follows the cat. He loves the cat.' 이라는 문장이 있을 경우 He: 10, follows: 11, the: 12, cat: 13, loves: 14 로 부여하는 것이다. 단어에 매핑되는 정수는 단어의 의미나 크기와 관련이 없다. 이 정수값 사이에는 어떠한 관계도 없다. 일반적으로 영어 문장은 모두 소문자로 바꾸고 구둣점을 삭제한 다음 공백을 기준으로 분리한다. 이렇게 분리된 단어를 **토큰(Token)**이라고 부른다. 하나의 샘플은 여러 개의 토큰으로 이루어져 있고 1개의 토큰이 하나의 타임스탬프에 해당한다. 참고로 한글은 조사가 발달되어 있어 공백으로 나누는 것만으로는 부족하다. 일반적으로 한글은 형태소 분석을 통해 토큰을 만든다.

토큰에 할당하는 정수 중 몇 개는 특정한 용도로 예약되어 있는 경우가 있다. 0: 패딩, 1: 문장의 시작, 2: 어휘 사전에 없는 토큰 이 그 예이다. 훈련 세트에서 고유한 단어를 뽑아 만든 목록을 **어휘 사전**이라고 하는데, 테스트 세트 안에 어휘 사전에 없는 단어가 있다면 2로 변환하여 신경망 모델에 주입한다. 실제 IMDB 리뷰 데이터셋은 영어로 된 문장이지만, 편리하게도 텐서플로에는 이미 정수로 바꾼 데이터가 포함되어 있다.
```python
from tensorflow.keras.datasets import imdb
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
print(train_input.shape, test_input.shape)
```
    (결과) (25000,) (25000,)

IMDB 리뷰 텍스트는 길이가 제각각이다. 따라서 고정 크기의 2차원 배열에 담기 보다는 리뷰마다 별도의 파이썬 리스트로 담아야 메모리를 효율적으로 사용할 수 있다. 그렇기 때문에 배열이 1차원인 것이다. 이 데이터는 개별 리뷰를 담은 파이썬 리스트 객체로 이루어진 넘파이 배열이다. 넘파이 배열은 정수나 실수 외에도 파이썬 객체를 담을 수 있다.
```python
print(type(train_input))
print(type(train_input[0]))
print(len(train_input[0]))
print(len(train_input[1]))
```
    (결과) <class 'numpy.ndarray'>
           <class 'list'>
           218
           189

보다시피 리뷰마다 각각 길이가 다르다. 여기서는 하나의 리뷰가 하나의 샘플이 된다.
```python
print(train_input[0])
```
    (결과) [1, 14, 22, 16, 43, 2, 2, 2, 2, 65, 458, 2, 66, 2, 4, 173, 36, 256, 5, 25, 100, 43, 2, 112, 50, 2, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 2, 2, 17, 2, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2, 19, 14, 22, 4, 2, 2, 469, 4, 22, 71, 87, 12, 16, 43, 2, 38, 76, 15, 13, 2, 4, 22, 17, 2, 17, 12, 16, 2, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2, 2, 16, 480, 66, 2, 33, 4, 130, 12, 16, 38, 2, 5, 25, 124, 51, 36, 135, 48, 25, 2, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 2, 15, 256, 4, 2, 7, 2, 5, 2, 36, 71, 43, 2, 476, 26, 400, 317, 46, 7, 4, 2, 2, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2, 56, 26, 141, 6, 194, 2, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 2, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 2, 88, 12, 16, 283, 5, 16, 2, 113, 103, 32, 15, 16, 2, 19, 178, 32]

앞서 설명한대로 IMDB 리뷰 데이터는 이미 정수로 변환되어 있다. 앞서 `num_words=500`으로 지정하였기 때문에 어휘 사전에는 500개의 단어만 들어가 있다. 어휘 사전에 없는 단어는 모두 2로 표시되어 나타난다. 500개 선정 기준은 등장 횟수 순서대로 나열한 다음 가장 많이 등장한 500개의 단어를 선택한다.
```python
print(train_target[:20])
```
    (결과) [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]

타겟 데이터는 리뷰가 긍정인지 부정인지에 대한 정답이 나온다. 0(부정)과 1(긍정)로 나누어진다. 이제 훈련 세트의 20%를 검증 세트로 떼어 놓자.
```python
from sklearn.model_selection import train_test_split
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
```

이제 평균적인 리뷰의 길이와 가장 중간 리뷰 길이를 확인해보자.
```python
import numpy as np
lengths = np.array([len(x) for x in train_input])
print(np.mean(lengths), np.median(lengths))
```
    (결과) 239.00925 178.0

lengths 배열에 대한 히스토그램도 그려보자.
```python
import matplotlib.pyplot as plt
plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()
```
![그림 8-1. 코드 결과](/assets/images/deeplearningtens/8-1.png)
{: .align-center}
그림 8-1. 코드 결과

히스토그램을 보면 대부분의 리뷰 길이가 300 미만이라는 것을 알 수 있다. 평균이 중간보다 높은 이유는 오른쪽 끝에 아주 큰 데이터가 있기 때문이다. 리뷰는 대부분 짧으므로, 중간값보다 훨씬 짧은 100개의 단어만 사용해보자. 물론 여전히 100개 단어보다 작은 리뷰가 있다. 이런 리뷰들은 길이를 100에 맞추도록 패딩이 필요하다. 보통 패딩을 나타내는 토큰으로는 0을 사용한다. 케라스는 시퀀스 데이터 길이를 맞추는 `pad_sequences()` 함수를 제공한다. 이 함수로 train_input 길이를 100으로 맞춰보자.
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen=100)
print(train_seq.shape)
print(train_seq[0])
```
    (결과) (20000, 100)
           [ 10   4  20   9   2 364 352   5  45   6   2   2  33 269   8   2 142   2
   5   2  17  73  17 204   5   2  19  55   2   2  92  66 104  14  20  93
  76   2 151  33   4  58  12 188   2 151  12 215  69 224 142  73 237   6
   2   7   2   2 188   2 103  14  31  10  10 451   7   2   5   2  80  91
   2  30   2  34  14  20 151  50  26 131  49   2  84  46  50  37  80  79
   6   2  46   7  14  20  10  10 470 158]

train_input[0]의 길이는 100개가 넘는다. 과연 앞에 데이터가 잘렸을까? 뒤에 데이터가 잘렸을까?
```python
print(train_input[0][-10:])
```
    (결과) [6, 2, 46, 7, 14, 20, 10, 10, 470, 158]

위 코드를 보다시피, train_input 데이터의 뒤에 10개를 출력해봤더니, train_seq의 뒤에 10개와 같다. 이로보아 앞에 데이터가 잘린것으로 보인다. `pad_sequences()` 함수는 기본으로 `maxlen`보다 긴 시퀀스의 앞부분을 자른다. 이렇게 하는 이유는 일반적으로 시퀀스의 뒷부분 정보가 더 유용하리라 기대하기 때문이다. 영화 리뷰 데이터를 생각해보자. 보통 리뷰 끝에 뭔가 결정적인 소감을 말할 가능성이 높아 보이지 않는가? 만약 시퀀스 뒷부분을 잘라내고 싶다면 `pad_sequences()` 함수의 `truncating` 매개변수 값을 기본 `pre`가 아닌 `post`로 바꾸면 된다.
```python
print(train_seq[5])
```
    (결과) [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   1   2 128  74  12   2 163  15   4   2   2   2   2  32  85
 156  45  40 148 139 121   2   2  10  10   2 173   4   2   2  16   2   8
   4 226  65  12  43 127  24   2  10  10]

다른 데이터를 보면 앞부분에 0이 있는 녀석도 있다. 이 샘플의 길이는 100개 안됐을 것이다. 패딩 토큰이 시퀀스 앞부분에 추가되었다. 만약 시퀀스 뒷부분에 패딩 토큰을 추가하고 싶다면 `pad_sequences()` 함수의 `padding` 매개변수 값을 기본 `pre`를 `post`로 바꾸면 된다.

이제 검증 세트 길이도 100으로 맞춰보자.
```python
val_seq = pad_sequences(val_input, maxlen=100)
```

## 8-2. 순환 신경망 만들기
IMDB 리뷰 분류 문제는 이진 분류이다. 따라서 마지막 출력층은 1개 뉴런을 가지고 시그모이드 활성화 함수를 사용해야 한다.
```python
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```

SimpleRNN 클래스를 보자. `input_shape`에 입력 차원을 (100, 500)으로 지정했다. 첫 번째 차원이 100인 것은 샘플 길이를 100으로 지정했기 때문이다. 그럼 500은 무엇일까? 이전에 만든 train_seq와 val_seq에는 한 가지 문제가 있다. 토큰을 정수로 변환한 이 데이터를 신경망에 주입하면 큰 정수가 큰 활성화 출력을 만든다는 것이다. 정수 사이에는 분명 어떠한 관련도 없다. 즉 20번 토큰이 10번 토큰보다 더 중요하다는 뜻이 아니라는 것이다. 따라서 단순한 정수값을 입력하려면 다른 방식을 찾아야 한다. 각 정수를 고유하게 표현하는 방법은 바로 **원-핫 인코딩**이다. train_seq의 첫 번째 토큰인 10을 원-핫 인코딩으로 바꾸면 열 번째 원소만 1이고 나머지는 모두 0인 배열이 된다. 이 배열의 길이는 `imbdb.load_data()` 함수에서 500개의 단어만 사용하도록 지정했으므로 500이 된다.

케라스에는 원-핫 인코딩을 위한 유틸리티를 제공한다. `keras.utils` 패키지 아래 `to_categorical()` 함수를 이용하면 된다. 정수 배열을 입력하면 자동으로 원-핫 인코딩된 배열을 반환해준다.
```python
train_oh = keras.utils.to_categorical(train_seq)
print(train_oh.shape)
```
    (결과) (20000, 100, 500)

train_oh의 첫 샘플의 첫 토큰이 10였었다. 잘 인코딩 되었는지 보자.
```python
print(train_oh[0][0][:12])
```
    (결과) [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]

토큰 2가 잘 인코딩 된 것으로 보인다. val_seq도 원-핫 인코딩으로 바꾸자.
```python
val_oh = keras.utils.to_categorical(val_seq)
```

이제 앞서 만든 모델의 구조를 출력해보자.
```python
model.summary()
```
    (결과) Model: "sequential"
           _________________________________________________________________
           Layer (type)                 Output Shape              Param #   
           =================================================================
           simple_rnn (SimpleRNN)       (None, 8)                 4072      
           _________________________________________________________________
           dense (Dense)                (None, 1)                 9         
           =================================================================
           Total params: 4,081
           Trainable params: 4,081
           Non-trainable params: 0
           _________________________________________________________________

입력 토큰은 500차원의 원-핫 인코딩 배열이다. 이 배열이 순환층의 뉴런 8개와 완전 연결되므로 총 500 x 8 = 4,000개의 가중치가 있다. 여기에 순환층의 은닉 상태는 다시 다음 타임스텝에 사용되므로 8(은닉 상태 크기) x 8(뉴런 개수) = 64개의 가중치가 필요하다. 각 뉴런마다 1개의 절편이 있으므로 총 4,000 + 64 + 8 = 4,072개의 모델 파라미터가 필요하다.

## 8-3. 순환 신경망 만들기
이 예에서는 기본 RMSprop 객체를 만들어 학습률을 0.0001로 지정할 것이다. 에포크 횟수는 100, 배치 크기는 64로 설정한다.
```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(train_oh, train_target, epochs=100, batch_size=64, validation_data=(val_oh, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
```
    (결과) Epoch 1/100
           313/313 [==============================] - 11s 34ms/step - loss: 0.6964 - accuracy: 0.5185 - val_loss: 0.6903 - val_accuracy: 0.5406
           Epoch 2/100
           313/313 [==============================] - 10s 32ms/step - loss: 0.6765 - accuracy: 0.5766 - val_loss: 0.6629 - val_accuracy: 0.6174
           ...
           Epoch 46/100
           313/313 [==============================] - 10s 33ms/step - loss: 0.4010 - accuracy: 0.8247 - val_loss: 0.4593 - val_accuracy: 0.7896
           Epoch 47/100
           313/313 [==============================] - 10s 32ms/step - loss: 0.3994 - accuracy: 0.8256 - val_loss: 0.4586 - val_accuracy: 0.7896


훈련 손실과 검증 손실을 그래프로 그려서 훈련 과정을 살펴보자.
```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```
![그림 8-2. 코드 결과](/assets/images/deeplearningtens/8-2.png)
{: .align-center}
그림 8-2. 코드 결과

훈련 손실은 꾸준히 감소한 반면, 검증 손실은 약 스무 번째 에포크에서 감소가 둔해지고 있다. 적절한 에포크에서 훈련을 멈춘 것으로 보인다.

우리가 입력 데이터를 원-핫 인코딩으로 변환했지만, 원-핫 인코딩은 사실 입력 데이터가 엄청 커지게 만드는 단점을 가진다.
```python
print(train_seq.nbytes, train_oh.nbytes)
```
    (결과) 8000000 4000000000

토큰 1개를 500차원으로 늘렸으므로 약 500배가 커졌다! 더 좋은 단어 표현 방법은 없을까?

## 8-4. 단어 임베딩을 사용하기
순환 신경망에서 텍스트를 처리할 때 즐겨 사용하는 방법이 바로 **단어 임베딩(Word embedding)** 이다. 단어 임베딩은 각 단어를 고정된 크기의 실수 벡터로 바꾸어 준다. 이런 단어 임베딩으로 만든 벡터는 원-핫 인코딩된 벡터보다 훨씬 의미 있는 값으로 채워져 자연어 처리에서 더 좋은 성능을 내는 경우가 많다. 케라스에서는 `keras.layers` 패키지 아래 `Embedding` 클래스로 임베딩 기능을 제공한다. 이 클래스를 다른 층처럼 모델에 추가하면 처음에는 모든 벡터가 랜덤하게 초기화되지만 훈련을 통해 데이터에서 좋은 단어 임베딩을 학습한다.

단어 임베딩의 장점은 입력으로 정수 데이터를 받는다는 것이다. 이 때문에 메모리를 훨씬 효율적으로 사용할 수 있다. 원-핫 인코딩이 샘플 하나를 500 차원으로 늘렸다면, 임베딩도 샘플 하나를 훨씬 적은 차원으로 늘려 2차원 배열로 늘린다. 원-핫 인코딩과는 달리 훨씬 작은 크기로도 단어를 표현할 수 있다.

```python
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

model2.summary()
```
    (결과) Model: "sequential_1"
           _________________________________________________________________
           Layer (type)                 Output Shape              Param #   
           =================================================================
           embedding (Embedding)        (None, 100, 16)           8000      
           _________________________________________________________________
           simple_rnn_1 (SimpleRNN)     (None, 8)                 200       
           _________________________________________________________________
           dense_1 (Dense)              (None, 1)                 9         
           =================================================================
           Total params: 8,209
           Trainable params: 8,209
           Non-trainable params: 0
           _________________________________________________________________

Embedding 클래스의 첫 번째 매개변수 500은 어휘 사전의 크기이다. 두 번째 매개변수 16은 임베딩 벡터의 크기이다. 세 번째 `input_length` 매개변수는 입력 시퀀스의 길이이다. `summary()` 메소드의 출력을 보면 이 모델은 (100, ) 크기의 입력을 받아 (100, 16) 크기의 출력을 만든다. `Embedding` 클래스는 500개의 각 토큰을 크기가 16인 벡터로 변경하므로 500 x 16 = 8,000개의 모델 파라미터를 갖는다. SimpleRNN 클래스는 (16 x 8) + (8 x 8) + 8 = 200개 이다. 원-핫 인코딩보다 SimpleRNN에 주입되는 입력의 크기는 크게 줄었다. 그러나 임베딩 벡터는 단어를 잘 표현하는 능력이 있으므로 훈련 결과는 이전 못지 않다.
```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model2.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data=(val_seq, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
```
    (결과) Epoch 1/100
           313/313 [==============================] - 8s 27ms/step - loss: 0.6856 - accuracy: 0.5534 - val_loss: 0.6678 - val_accuracy: 0.6112
           Epoch 2/100
           313/313 [==============================] - 8s 27ms/step - loss: 0.6376 - accuracy: 0.6716 - val_loss: 0.6094 - val_accuracy: 0.6992
           ...
           Epoch 24/100
           313/313 [==============================] - 8s 27ms/step - loss: 0.4246 - accuracy: 0.8121 - val_loss: 0.4688 - val_accuracy: 0.7826
           Epoch 25/100
           313/313 [==============================] - 8s 26ms/step - loss: 0.4222 - accuracy: 0.8140 - val_loss: 0.4626 - val_accuracy: 0.7898

출력 결과를 보면 원-핫 인코딩을 사용한 모델과 비슷한 성능을 냈다. 반면 순환층의 가중치 개수는 훨씬 작고 훈련 세트 크기도 훨씬 줄었다.
```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```
![그림 8-3. 코드 결과](/assets/images/deeplearningtens/8-3.png)
{: .align-center}
그림 8-3. 코드 결과