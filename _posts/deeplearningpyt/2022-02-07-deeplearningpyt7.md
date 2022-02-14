---
layout: single
title: "[Deeplearning(pytorch)] 7. CNN 기초와 모델 구현"
folder: "deeplearningpyt"
categories:
    - deeplearningpyt
tag:
    - [deep learning, pytorch, CNN]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "펭귄브로의 3분 딥러닝, 파이토치맛" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 7-1. 컴퓨터가 보는 이미지
컴퓨터에서 모든 이미지는 픽셀값들을 가로, 세로로 늘어놓은 행렬로 표현된다. 보통 인공 신경망은 다양한 형태의 입력에 대한 확정성이 떨어진다. 같은 신발 이미지라고 해도, 신발이 옆으로 조금만 치우쳐지면 예측률이 급격히 떨어진다. 특징을 추출하는 가중치가 가운데만 집중하도록 학습되었기 때문이다. 토치비전의 `transforms` 도구를 쓴다해도, 고화질 이미지의 크기를 고려하면 각 배치당 처리해야하는 데이터 수는 엄청 늘어난다. 신발의 가운데, 왼쪽 등 다양하게 배치된 이미지를 따로 학습하는 것도 이상적이지 않다. 인공지능이 하나를 배우면서 다른 하나도 자연스럽게 유추하도록 만들순 없을까?

## 7-2. 컨볼루션
컨볼루션의 목적은 계측정으로 인식할 수 있도록 단계마다 이미지의 특징을 추출하는 것이다. 각 단계에서는 이미지에 대한 다양한 필터를 적용하여 윤곽선, 질감, 털 등 각종 특징을 추출한다. 윤곽선을 검출하는 필터를 사용하면, 밑그림을 그린것 같은 이미지가 출력된다. 이러한 필터를 적용할 때, 이미지 왼쪽 위에서 오른쪽 밑까지 밀어가며 곱하고 더하는데, 이 작업을 **컨볼루션(Convolution)**이라고 한다. CNN은 이미지를 추출하는 필터를 학습한다. 필터가 하나의 작은 신경망인 것이다.

## 7-3. CNN 모델
CNN 모델은 컨볼루션 계층, 풀링 계층, 일반적인 인공 신경망 계층으로 구성된다. 
- 컨볼루션 계층: 이미지의 특징을 추출하는 역할. 
- 풀링 계층: 필터를 거친 여러 특징 중 가장 중요한 특징 하나를 골라냄. 덜 중요한 특징을 버려 이미지의 차원 감소함.

컨벌루션 연산은 이미지를 겹치는 매우 작은 조각으로 쪼개어 필터 기능을 하는 작은 신경망에 적용한다. 이 신경망은 모든 조각에 동일하게 적용되며, 특징을 추출하기 때문에 **컨볼루션 필터** 또는 **커널**이라고 부른다. 보통 3 x 3 또는 5 x 5 크기의 커널이 쓰이는데, 컨볼루션 계층 하나에 여러 개 존재할 수 있다. 학습이 시작되면, 필터 행렬의 값은 특징을 잘 뽑을 수 있도록 최적화 된다. 컨볼루션은 오른쪽 아래로 움직이며 다음 이미지를 만드는데, 한 칸 또는 여러 칸을 건너뛸 수 있다. 이 움직임을 조절하는 값을 **스트라이드**라고 한다. 스트라이드를 크게 주어 여러 칸을 건너뛰면 텐서의 크기는 작아지게 된다. 컨볼루션을 거쳐 만들어진 새로운 이미지는 **특징 맵** 이라고 부른다. 컨볼루션 계층 마다 여러 특징 맵이 만들어지고, 다음 단계인 풀링 계층으로 넘어가게 된다. 특징 맵 크기가 크면 과적합의 위험이 증가하므로, 풀링 계층에서 특징 맵의 크기를 줄여주기 위해 특징을 값 하나로 추려내어 특징을 강조하도록 한다. 보통 필터가 지나갈 때마다 픽셀을 묶어서 평균이나 최대값을 가져오는 간단한 연산이 이루어 진다. 특징 맵을 관찰하면, CNN 모델이 이미지를 계층적으로 인식한다는 것을 확인할 수 있다. 특징은 선 같은 저수준에서 눈, 코, 입 거쳐 얼굴 같은 고수준 특징이 추출되게 된다. 결과적으로, CNN은 사물이 조금만 치우쳐져도 인식하지 못하던 인공 신경망의 문제를 이미지 전체에 필터를 적용하여 특징을 추출하는 방식으로 해결해준다.

## 7-4. CNN 모델 구현하기
이제 코드로 구현해보자. 데이터셋을 만드는거 까지 쭉 가보자.
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

EPOCHS = 40
BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
  datasets.FashionMNIST('./data',
                  train=True,
                  download=True,
                  transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                  ])),
  batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  datasets.FashionMNIST('./.data',
                  train=False,
                  transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                  ])),
  batch_size=BATCH_SIZE, shuffle=True)
```

이제 CNN 모델을 설계할 차례다. `nn.Conv2d` 함수를 활용할 것이다. 이 함수의 첫 두 파라미터는 입력 채널 수 `in_channels`와 출력 채널 수 `out_channels` 이다. Fashion MNIST 데이터셋은 흑백이미지라, 색상 채널이 1개이다. 첫 컨볼루션 계층에서는 10개 특징 맵을 생성하고, 두 번째 컨볼루션 계층은 10개의 특징 맵을 받아 20개의 특징 맵을 만들도록 해보자. 커널 사이즈는 5x5로 만들 것이다. 커널 사이즈에 숫자 하나만 지정하면 정사각형으로 간주한다. (3, 5)로 입력하면 3 x 5 크기의 직사각형 커널을 만들수 도 있다.
```python
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.conv2d(10, 20, kernel_size=5)
    self.drop = nn.Dropout2d() # 이번에는 functional 대신 nn.Dropout2D 모듈 활용해봄!
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)  # 계층의 출력 크기는 특별한 이유 없이 계층이 진행될수록 작아지도록 임의로 정함.
  
  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))  # F.max_pool2d() 함수의 두 번째 입력은 커널 크기! 학습 파라미터 따로 없음.
    x = F.relu(F.max_pool2d(self.conv2(x), 2))  # F.max_pool2d() 외, nn.MaxPool2d 같은 일반 모듈도 사용 가능.
    x = x.view(-1, 320)   # 320은 x가 가진 원소 개수 의미
    x = F.relu(self.fc1(x))
    x = self.drop(x)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)
```

이제 나머지 코드를 작성하자!
```python
model = CNN().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(model, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 200 == 0:  # batch_idx는 데이터를 64개 씩 몇 번째 가져오고 있는지를 의미 
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:[:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

def evaluate(model, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(DEVICE), target.to(DEVICE)
      output = model(data)
      test_loss += F.cross_entropy(output, target, reduction='sum').item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  test_accuracy = 100. * correct / len(test_loader.dataset)

  return test_loss, test_accuracy

for epoch in range(1, EPOCHS + 1):
  train(model, train_loader, optimizer, epoch)
  test_loss, test_accuracy = evaluate(model, test_loader)

  print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))
```
    (결과) Train Epoch: 1 [0/60000 (0%)]	Loss:2.300701
           Train Epoch: 1 [12800/60000 (21%)]	Loss:1.112408
           Train Epoch: 1 [25600/60000 (43%)]	Loss:0.926304
           Train Epoch: 1 [38400/60000 (64%)]	Loss:0.991300
           Train Epoch: 1 [51200/60000 (85%)]	Loss:0.779528
           [1] Test Loss: 0.6457, Accuracy: 75.35%
           Train Epoch: 2 [0/60000 (0%)]	Loss:0.681881
           Train Epoch: 2 [12800/60000 (21%)]	Loss:1.025114
           Train Epoch: 2 [25600/60000 (43%)]	Loss:0.503789
           Train Epoch: 2 [38400/60000 (64%)]	Loss:0.849935
           Train Epoch: 2 [51200/60000 (85%)]	Loss:0.603705
           [2] Test Loss: 0.5394, Accuracy: 79.39%
           ...
           ...
           ...
           Train Epoch: 40 [0/60000 (0%)]	Loss:0.231346
           Train Epoch: 40 [12800/60000 (21%)]	Loss:0.226066
           Train Epoch: 40 [25600/60000 (43%)]	Loss:0.247825
           Train Epoch: 40 [38400/60000 (64%)]	Loss:0.190665
           Train Epoch: 40 [51200/60000 (85%)]	Loss:0.155920
           [40] Test Loss: 0.2999, Accuracy: 89.65%
