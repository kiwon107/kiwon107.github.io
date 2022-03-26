---
layout: single
title: "[Deeplearning(pytorch)] 11. RNN 개요와 영화 리뷰 감정 분석"
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

## 11-1. RNN 개요
지금까지 배운 신경망은 연달아 있는 데이터의 순서와 상호작용을 인식하여 전체 상황을 이해하는 능력을 가지고 있지 않았다. 즉, 시간에 대한 개념이 없는 데이터와 그에 따른 신경망을 다룬 것이다. 앞서 이미지와 같은 정적인 데이터를 인지하는 모델들을 다뤘지만, 현실에서 접하는 거의 모든 경험은 순차적이다. 이번에는 **순처적 데이터(Sequential data)** 혹은 **시계열 데이터(Time series data)** 정보를 받아 전체 내용을 학습하는 **RNN(Recurrent Neural Network)** 를 다뤄본다.

RNN은 정해지지 않은 길이의 배열을 읽고 설명하는 신경망이다. RNN의 출력은 순차적 데이터의 흐름을 모두 내포한다. 시계열 데이터의 정보를 하나씩 입력받을 때마다 지금까지 입력된 벡터들을 종합해 **은닉 벡터(Hidden vector)**를 만든다. 첫 번째 학습 데이터 벡터 입력1을 입력받은 RNN은 은닉벡터1을 생성한다. 입력1에 이어 입력2까지 RNN에 입력되면 입력1과 입력2를 압축한 은닉벡터2가 만들어진다. 이런식으로 마지막 입력K 까지 입력받은 RNN은 이들을 모두 압축하여 은닉벡터K를 만들어낸다. 마지막 은닉벡터K는 배열 속 모든 벡터들의 내용을 압축한 벡터라고 할 수 있다.

RNN 계열 신경망들은 텍스트나 자연어를 처리하고 학습하는데 주로 사용된다. 현재는 **LSTM(Long short term memory)**, **GRU(Gated recurrent unit)** 등 응용 RNN이 개발되어 언어 모델링, 텍스트 감정 분석, 기계 번역 등 분야에 활발하게 이용되고 있다.

RNN을 이용한 신경망 형태는 크게 4가지가 있다. **일대일(one to one)**은 일반적으로 그동안 보았던 신경망이나 CNN과 같다. **일대다(one to many)**는 이미지를 보고 이미지 안의 상황을 글로 설명하는 등의 신경망이다.  **다대일(many to one)**은 감정 분석 같이 순차적인 데이터를 보고 값 하나를 내는 경우이다. **다대다(many to many)**는 2가지 유형이 있는데, 챗봇과 기계 번역 같이 순차적인 데이터를 보고 순차적인 데이터를 출력하는 유형과 비디오 분류 같이 매 프레임을 레이블링할 때 사용되는 유형이 있다.

## 11-2. 토크나이징과 워드 임베딩
이번 시간에는 자연어 텍스트 데이터를 다뤄볼 것이다. 이러한 데이터를 인공 신경망에 입력시키려면 전처리 과정을 거쳐 데이터를 숫자로 나타내야 한다. 전처리를 위해 가장 먼저 해야할 작업은 텍스트 문장을 언어의 최소 단위인 **토큰(Token)**으로 나누는 것이다. `split()` 함수를 사용하여 단어 단위로 **토크나이징(Tokenizing)** 해도 큰 문제 없으며, `SpaCy` 같은 오픈소스 라이브러리를 사용하는 것도 좋다.<br/>
그 다음 문장 속 모든 토큰을 각각 벡터로 나타내주어야 한다. 벡터로 나타내기 위해서 데이터셋의 모든 단어 수만큼 벡터를 담는 **사전(Dictionary)**을 정의해야 한다. 만약 'Quick brown fox jumps over the lazy dog.' 이라는 문장이 있다면, 다음과 같이 사전에 단어 벡터가 총 8개 담기게 된다. `['quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']`. 'brown' 단어를 벡터로 변환하려면 이 사전에서 'brown'에 해당되는 벡터를 가져오면 된다. 이처럼 언어의 최소 단위인 토큰을 벡터 형태로 변환하는 작업을 **워드 임베딩(Word embedding)** 이라고 한다.

## 11-3. 영화 리뷰 감정 분석
데이터의 순서 정보를 학습한다는 점에서 RNN은 동영상, 자연어, 주가 등 동적인 데이터를 이용할 때 성능이 극대화된다. 이번에는 IMDB 데이터셋을 활용할 것이다. 이 데이터셋은 영화 리뷰 5만 건으로 이뤄졌다. 각 리뷰는 다수의 영어 문장으로 구성되었으며, 긍정 영화 리뷰는 2, 부정 영화 리뷰는 1로 레이블링 되어있다. 해당 데이터셋을 이용하여 영화 리뷰가 긍정적인지 부정적인지 판단해주는 간단한 분류 모델을 만들어 볼 것이다.<br/>

먼저 관련 모듈 임포트, 모델 파라미터 설정, 데이터 생성 및 전처리 단계이다
```python
# 관련 모듈 임포트
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data, datasets

# 모델 파라미터 설정
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 40
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# 데이터를 텐서로 변환시 필요한 설정 객체 생성
TEXT = data.Field(sequential=True, batch_first=True, lower=True) # sequential: 데이터셋이 순차적인가?, batch_first: 신경망에 입력되는 텐서의 첫 번째 차원값이 batch_size 되도록 설정, lower: 텍스트 데이터 속 모든 영문 알파벳이 소문자가 되도록 설정
LABEL = data.Field(sequential=False, batch_first=True)

# 모델에 입력되는 데이터 셋 생성
trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

# 워드 임베딩에 필요한 단어 사전(Word vocabulary) 생성
TEXT.build_vocab(trainset, min_freq=5) # min_freq: 최소 5번 이상 등장한 단어만을 사전에 담도록 설정, 5번 미만 등장한 단어는 unknown인 'unk'라는 토큰으로 대체
LABEL.build_vocab(trainset)

# 훈련, 검증, 테스트 데이터셋 생성
trainset, valset = trainset.split(split_ratio=0.8) # 80% 는 훈련셋, 나머지는 검증셋으로 활용
train_iter, val_iter, test_iter = data.BucketIterator.splits((trainset, valset, testset), batch_size=BATCH_SIZE, shuffle=True, repeat=False) # 배치 단위의 반복자(Iterator) 생성

# 사전 속 데이터들의 개수와 레이블 수 정해주는 변수 생성
vocab_size = len(TEXT.vocab)
n_classes = 2

print("[학습셋]: %d [검증셋]: %d [테스트셋]: %d [단어수]: %d [클래스] %d" % (len(trainset), len(valset), len(testset), vocab_size, n_classes))
```
    (결과) [학습셋]: 20000 [검증셋]: 5000 [테스트셋]: 25000 [단어수]: 46159 [클래스] 2

그 다음은 모델 설계 단이다.<br/>
여기서는 GRU 라는 모델을 사용한다. 왜 RNN은 안쓰고 GRU를 사용하려는 걸까?<br/>
RNN은 입력이 길어지면 학습 도중 기울기가 너무 작아지거나 커져서 앞부분에 대한 정보를 정확히 담지 못할 수 있다. 기울기가 학습 도중 폭발적으로 커지는 현상을 **기울기 폭발(Gradient explosion)** 이라 하며, 기울기가 너무 작아지는 현상을 **기울기 소실(Vanishing gradient)**라고 한다. 이를 보완하기 위해 만들어진 대표 모델이 **LSTM(Long-short term memory)**과 **GRU(Gated recurrent unit)**이다. GRU는 시계열 데이터 속 벡터 사이의 정보 전달량을 조절하여 기울기를 적정하게 유지하고 문장 앞부분의 정보가 끝까지 도달할 수 있도록 도와준다.<br/>
좀 더 설명을 하자면, GRU에는 시계열 데이터 내 정보 전달량을 조절하는 **업데이트 게이트(Update gate)**와 **리셋 게이트(Reset gate)**가 있다. 업데이트 게이트는 이전 은닉 벡터가 지닌 정보를 새로운 은닉 벡터가 얼마나 유지할지 정해준다. 리셋 게이트는 새로운 입력이 이전 은닉 벡터와 어떻게 조합하는지를 결정한다.
```python
# 모델 설계
class BasicGRU(nn.Module):
  def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
    super(BasicGRU, self).__init__()
    print("Bulding Basic GRU model...")
    self.n_layers = n_layers # 아주 복잡한 모델이 아닌 이상 2 이하로 정의
    self.embed = nn.Embedding(n_vocab, embed_dim) # n_vocab: 전체 데이터셋의 모든 단어를 사전 형태로 나타낼 때 그 사전에 등재된 단어 수, embed_dim: 임베딩된 단어 텐서가 지나가는 차원
    self.hidden_dim = hidden_dim # 은닉벡터 차원
    self.dropout = nn.Dropout(dropout_p)
    self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True)
    self.out = nn.Linear(self.hidden_dim, n_classes)
  
  def forward(self, x):
    x = self.embed(x)
    h_0 = self._init_state(batch_size=x.size(0)) # 첫 번째 은닉 벡터 H0 정의
    x, _ = self.gru(x, h_0) # (batch_size, 입력 x의 길이, hidden_dim) 모양을 지닌 3d 텐서
    h_t = x[:, -1, :] # (batch_size, 1, hidden_dim) 모양의 텐서 추출
    self.dropout(h_t)
    logit = self.out(h_t)
    return logit

  def _init_state(self, batch_size=1):
    weight = next(self.parameters()).data # 가중치 정보들을 반복자 형태로 반환
    return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_() # 모델의 가중치과 같은 모양인 (n_layers, batch_size, hidden_dim) 모양을 갖춘 텐서로 변환 후 zero_() 함수로 텐서 내 모든 값을 0으로 초기화하여 첫 번째 은닉 벡터의 모든 특성값이 0인 벡터로 설정
```

이제 학습 함수와 평가 함수를 구현한다.
```python
def train(model, optimizer, train_iter):
  model.train()
  for b, batch in enumerate(train_iter):
    x, y = batch.text.to(DEVICE), batch.label.to(DEVICE) # batch.label은 1이나 2의 값 가짐
    y.data.sub_(1) # batch.label이 1 or 2가 아닌 1씩 빼서 0 or 1 값을 갖도록 수정

    optimizer.zero_grad()
    logit = model(x)
    loss = F.cross_entropy(logit, y)
    loss.backward()
    optimizer.step()

def evaluate(model, val_iter):
  model.eval()
  corrects, total_loss = 0, 0

  for batch in val_iter:
    x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
    y.data.sub_(1)
    logit = model(x)
    loss = F.cross_entropy(logit, y, reduction='sum') # 오차의 합을 구함
    total_loss += loss.item()  # 오차의 합을 total_loss에 더해줌
    corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
  
  size = len(val_iter.dataset)
  avg_loss = total_loss / size
  avg_accuracy = 100.0 * corrects / size
  return avg_loss, avg_accuracy
```

```python
model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
best_val_loss = None

for e in range(1, EPOCHS+1):
  train(model, optimizer, train_iter)
  val_loss, val_accuracy = evaluate(model, val_iter)
  print("[이폭: %d] 검증 오차:%5.2f | 검증 정확도: %5.2f" % (e, val_loss, val_accuracy))

  if not best_val_loss or val_loss < best_val_loss:
    if not os.path.isdir("snapshot"):
      os.makedirs("snapshot")
    torch.save(model.state_dict(), './snapshot/txtclassification.pt')
    best_val_loss = val_loss

model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))
```
    (결과) Bulding Basic GRU model...
           [이폭: 1] 검증 오차: 0.70 | 검증 정확도: 49.46
           [이폭: 2] 검증 오차: 0.71 | 검증 정확도: 49.78
           [이폭: 3] 검증 오차: 0.71 | 검증 정확도: 52.72
           ...
           [이폭: 37] 검증 오차: 0.86 | 검증 정확도: 86.28
           [이폭: 38] 검증 오차: 0.88 | 검증 정확도: 86.30
           [이폭: 39] 검증 오차: 0.90 | 검증 정확도: 86.32
           [이폭: 40] 검증 오차: 0.92 | 검증 정확도: 86.28
           테스트 오차:  0.35 | 테스트 정확도: 85.62