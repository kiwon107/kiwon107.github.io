---
layout: single
title: "[Deeplearning(pytorch)] 12. Seq2Seq 기계 번역"
folder: "deeplearningpyt"
categories:
    - deeplearningpyt
tag:
    - [deep learning, pytorch, rnn]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "펭귄브로의 3분 딥러닝, 파이토치맛" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 12-1. Seq2Seq 개요
언어를 다른 언어로 해석해주는 **뉴럴 기계 번역(Neural machine translation)** 모델이 있다. RNN 기반의 번역 모델인 Sequence to Sequence(=Seq2Seq) 모델은 기계 번역의 새로운 패러다임을 열었다. Seq2Seq 모델은 시퀀스를 입력받아 또 다른 시퀀스를 출력한다. 즉, 문장을 다른 문장으로 번역해주는 모델인 것이다.

Seq2Seq는 각자 다른 역할을 하는 두 개의 RNN을 이어붙인 모델이다. 외국어를 한국어로 번역할 때 다음과 같은 프로세스를 거친다.<br/>
1) 외국어 문장을 읽고 의미를 이해한다.<br/>
2) 외국어 문장의 의미를 생각하며 한국어 단어를 한 자 한 자 문맥에 맞게 적어나간다.<br/>

이처럼 번역은 원문을 이해하고 번역문을 작성하는 두 가지 동작으로 구성된다. 이 두 역할을 **인코더(Encoder)**와 **디코더(Decoder)**에 부여하여 번역을 수행한다.

## 12-2. 인코더 
인코더는 원문의 내용을 학습하는 RNN이다. 원문 속 모든 단어를 입력받아 문장의 뜻을 내포하는 하나의 고정 크기 텐서를 만들어낸다. 이렇게 압축된 텐서는 원문 뜻과 내용을 압축하고 있어 **문맥 벡터(Context vector)**라고 한다.

## 12-3. 디코더
인코더로부터 원문 문맥 벡터를 이어 받아 번역문 속의 토큰을 차례대로 예상한다. 번역할 때 '원문이 말하는 바가 무엇인가'를 항상 생각하고 있어야 한다. 이는 디코더가 번역문의 단어나 토큰 출력시 인코더로부터 정보를 전달받아야 한다는 뜻이기도 하다.

## 12-4. Seq2Seq 모델 구현하기
한 언어로 된 문장을 다른 언어로 번역시, 보통 단어를 문장의 초소 단위로 여겨 단어 단위의 임베딩을 한다. 그러나 이번 예제에서는 간단한 영단어를 스페인어로 번역하는 작업을 할 것이므로 글자 단위의 캐릭터 임베딩을 할 것이다.

```python
# 관련 모듈 임포트
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

# 사전에 담을 수 있는 토큰 수 = 총 아스키코드 개수
vocab_size = 256

# 아스키 코드로 변환
x_ = list(map(ord, "hello")) # ord(c)는 문자의 유니코드 값을 돌려주는 함수
y_ = list(map(ord, "hola"))
print("hello -> ", x_)
print("hola -> ", y_)
x = torch.LongTensor(x_) # Long 타입의 텐서로 변환
y = torch.LongTensor(y_)
```
    (결과) hello ->  [104, 101, 108, 108, 111]
           hola ->  [104, 111, 108, 97]

이제 모델을 설계할 차례다. 다음 그림처럼 디코더가 예측한 토큰을 다음 반복에서 입력될 토큰으로 갱신해주는 것이 정석이다. 그러나 학습이 아직 되지 않은 상태의 모델은 잘못된 예측 토큰을 입력으로 사용할 확률이 높다. 반복해서 잘못된 입력 토큰이 사용되면 학습은 더욱 더뎌지게된다. 이를 방지하는 방법 중 **티처 포싱(Teacher forcing)** 이라는 방법이 있다. 디코더 학습 시 실제 번역문의 토큰을 디코더의 전 출력값 대신 입력으로 사용해 학습을 가속하는 방법이다.

![그림 12-1. 원문의 문맥 벡터를 이어받아 번역문을 작성하는 디코더](/assets/images/deeplearningpyt/12-1.png)
{: .align-center}
그림 12-1. 원문의 문맥 벡터를 이어받아 번역문을 작성하는 디코더

```python
# Seq2Seq 모델 클래스 정의
# Seq2Seq 모델 클래스 정의
class Seq2Seq(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super(Seq2Seq, self).__init__()
    self.n_layers = 1
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, hidden_size) # 임베딩 차원 따로 정의하지 않고 hidden_size로 임베딩 토큰 차원값 정의!
    self.encoder = nn.GRU(hidden_size, hidden_size)
    self.decoder = nn.GRU(hidden_size, hidden_size)
    self.project = nn.Linear(hidden_size, vocab_size) # 디코더 토큰을 예상해내는 작은 신경망

  def forward(self, inputs, targets):
    initial_state = self._init_state() # 초기 은닉 벡터 정의. (1, 1, 16).
    embedding = self.embedding(inputs).unsqueeze(1) # 인코더에 입력되는 원문을 구성하는 모든 문자 임베딩. (5, 16)을 (5, 1, 16)으로 shape 변환.

    encoder_output, encoder_state = self.encoder(embedding, initial_state) # encoder_state: 문맥 벡터, encoder_output은 (5, 1, 16), encoder_state는 (1, 1, 16).

    decoder_state = encoder_state # decoder_state: 디코더의 첫 번째 은닉 벡터
    decoder_input = torch.LongTensor([0]) # 디코더에 문장의 시작을 알리기 위함. 이 토큰으로 h 토큰 예측!

    outputs = []
    for i in range(targets.size()[0]):
      decoder_input = self.embedding(decoder_input).unsqueeze(1) # (1, 1, 16)
      decoder_output, decoder_state = self.decoder(decoder_input, decoder_state) # 디코더 결과값(decoder_state)은 다시 디코더 모델에 입력됨!, decoder_output은 (1, 1, 16), decoder_state는 (1, 1, 16).
      projection = self.project(decoder_output) # 예상 글자 출력, (1, 1, 256).
      outputs.append(projection) # 예상 결과 저장하여 오차 계산에 사용

      decoder_input = torch.LongTensor([targets[i]]) # 타겟값 차례대로 디코더에 입력. 이것이 티처포싱!
    outputs = torch.stack(outputs).squeeze() # 번역문의 모든 토큰에 대한 결과값 반환, (4, 1, 1, 256) -> (4, 256)

    return outputs

  def _init_state(self, batch_size=1):
    weight = next(self.parameters()).data
    return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
```

이제 모델을 훈련시켜 결과를 확인해보자.
```python
# 모델, 오차, 옵티마이저 객체 생성
seq2seq = Seq2Seq(vocab_size, 16)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)

# 1000번의 에폭으로 학습
log = []
for i in range(1000):
  prediction = seq2seq(x, y)
  loss = criterion(prediction, y) # (4, 256) 과 (4)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  loss_val = loss.data
  log.append(loss_val)
  if i % 100 == 0:
    print("\n 반복: %d 오차: %s" % (i, loss_val.item()))
    _, top1 = prediction.data.topk(1, 1) # 예측값 저장. topk는 주어진 텐서에서 두번째 인자(1) 차원에 따라 가장 큰 값 첫번째 인자 값(1) 개수 리턴. top1 shape은 (4, 1)!
    print([chr(c) for c in top1.squeeze().numpy().tolist()]) # 예측값 한글자씩 가져옴

plt.plot(log)
plt.ylabel('cross entropy loss')
plt.show()
```
    (결과) 반복: 0 오차: 5.637078285217285
           ['J', "'", '\x7f', '7']

           반복: 100 오차: 2.161990165710449
           ['h', 'o', 'l', 'a']

           반복: 200 오차: 0.4579141139984131
           ['h', 'o', 'l', 'a']

           반복: 300 오차: 0.1976625919342041
           ['h', 'o', 'l', 'a']

           반복: 400 오차: 0.1192457526922226
           ['h', 'o', 'l', 'a']

           반복: 500 오차: 0.08297178149223328
           ['h', 'o', 'l', 'a']

           반복: 600 오차: 0.06233248859643936
           ['h', 'o', 'l', 'a']

           반복: 700 오차: 0.04910457134246826
           ['h', 'o', 'l', 'a']

           반복: 800 오차: 0.03995484113693237
           ['h', 'o', 'l', 'a']

           반복: 900 오차: 0.033283431082963943
           ['h', 'o', 'l', 'a']

![그림 12-2. 코드 결과](/assets/images/deeplearningpyt/12-2.png)
{: .align-center}
그림 12-2. 코드 결과