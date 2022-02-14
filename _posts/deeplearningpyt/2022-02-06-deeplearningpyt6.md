---
layout: single
title: "[Deeplearning(pytorch)] 6. 과대적합과 드롭아웃"
folder: "deeplearningpyt"
categories:
    - deeplearningpyt
tag:
    - [deep learning, pytorch, DNN]

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true

sidebar:
  nav: "docs"
---

본 포스팅은 "펭귄브로의 3분 딥러닝, 파이토치맛" 책 내용을 기반으로 작성되었습니다.
잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.

## 6-1. 과대적합, 과소적합, 조기종료
머신러닝 모델을 만들면 학습 성능은 잘 나오지만, 테스트셋이나 실제 상황에서는 성능이 나오지 않을 때가 있다. 이것을 **과대적합(Overfitting)** 이라고 한다. 즉, 너무 학습 데이터에만 치중되어 새로운 데이터에 대해서는 성능이 잘 나오지 않는 상황을 말한다. 반대로, 학습을 제대로 진행하지 않은 상황을 **과소적합(Underfitting)** 이라고 한다. 이 경우는 학습 데이터도 제대로 학습하지 않는 경우이다.<br/>
가장 베스트인 상태는, 과소적합과 과대적합의 중간이다. 학습 데이터와 학습하지 않은 실제 데이터에서 동시에 높은 성능을 내는 상태가 바로 **일반화(Generalization)** 라고 한다.<br/>

5장에서, 머신러닝에서는 보통 데이터셋을 학습, 검증, 테스트셋으로 나눈다고 하였다. 이는 과대적합, 과소적합을 탐지하려는 노력의 일환이다. 학습 데이터셋으로 계속 학습하면 오차는 무한정 내려가게 된다. 그러다 보면, 학습 성능은 계속 좋아지지만, 검증/테스트 성능은 오히려 떨이지는 것을 확인할 수 있다. 따라서! 학습 중간중간 검증용 데이터셋으로 모델이 학습 데이터에만 과대적합되지 않았는지 확인이 필요하다!<br/>
검증 데이터셋에 대한 성능이 나빠지기 시작하면, 직전이 가장 적합한 모델이라고 할 수 있다. 이 타이밍에 모델을 저장해서 이용하는 것을 조기 종료(Early stopping) 이라고 한다. 즉, 검증 오차가 올라가는 순간을 포착해서 학습을 종료하는 것이다.

과대적합을 막는 방법은 **학습 데이터를 늘리는 방법**과 **드롭아웃을 적용하는 방법**이 있다.

## 6-2. 데이터 늘리기
궁극적으로, 세상의 모든 데이터를 모으는 것이 가장 효과가 좋다(모으기가 힘들뿐...). 새로운 데이터를 얻기가 어렵다면, 이미 가진 데이터를 최대한 늘리는 방법(Data Augmentaion)을 찾아야 한다. 이미지 데이터라면 이미지 일부를 자르거나, 돌리거나, 노이즈를 더하거나, 색상을 변경하는 등 여러가지 방법을 사용할 수 있다. 예제에서는 가로 대칭이동(왼쪽, 오른쪽 뒤집기) 전략을 써먹어 보자. 토치비전의 `transform` 패키지를 사용하면 간단하다.
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 50
BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('./.data',
                  train=True,
                  download=True,
                  transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(), # 이미지를 무작위로 수평 뒤집기 수행! 학습 데이터셋에만 적용!
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                  ])),
  batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('./.data',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                  ])),
  batch_size=BATCH_SIZE, shuffle=True)
```
이미지를 뒤집는 것만으로도 학습 데이터 수가 2배 늘었다!

## 6-3. 드롭아웃
드롭아웃은 학습 진행 과정에서 신경망의 일부를 사용하지 않는 방법이다. 만약 50% 드롭아웃을 한다면, 학습 단계마다 절반의 뉴런만 사용하도록 한다. 그리고 검증과 테스트 단계에서는 모든 뉴런을 사용한다. 학습에서 배재된 뉴런 외에 다른 뉴런들에 가중치를 분산시키고, 개별 뉴런이 특징에 고정되는 현상을 방지하는 기능을 한다.
```python
class Net(nn.Module):
  def __init__(self, dropout_p=0.2):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(784, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 10)
    self.dropout_p = dropout_p  # 드롭아웃 확률 기본값 0.2로 설정! 학습시 20% 뉴런을 사용하지 않겠다는 의미

  def forward(self, x):
    x = x.view(-1, 784)
    x = F.relu(self.fc1(x))

    # 가중치가 없으므로, torch.nn.functional 패키지에서 가져와서 사용! nn.Dropout 클래스 사용도 가능!
    # self.training은 몇 가지 내부 변수를 자동으로 적용해주는 모듈이다.
    # model.train()과 model.eval()로 학습모드냐, 평가모드냐에 따라 모델 내부의 self.training 변수값이 True 또는 False로 바뀐다!
    # fc1 지나면서 1번 수행
    x = F.dropout(x, training=self.training, p=self.dropout_p) 
    
    x = F.relu(self.fc2(x))
    # fc2 지나면서 또 다시 1번 수행
    x = F.dropout(x, training=self.training, p=self.dropout_p)

    x = self.fc3(x)

    return x

model = Net(dropout_p=0.2).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

이제 나머지 코드도 작성해서 결과를 확인해보자!
```python
def train(model, train_loader, optimizer):
  model.train()

  for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(DEVICE), target.to(DEVICE)
      optimizer.zero_grad()
      output = model(data)
      loss = F.cross_entropy(output, target)
      loss.backward()
      optimizer.step()

def evaluate(model, train_loader):
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
  train(model, train_loader, optimizer)
  test_loss, test_accuracy = evaluate(model, test_loader)
  print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))
```
    (결과) [1] Test Loss: 0.5419, Accuracy: 83.04%
           [2] Test Loss: 0.4139, Accuracy: 87.15%
           [3] Test Loss: 0.3397, Accuracy: 89.55%
           [4] Test Loss: 0.2877, Accuracy: 91.15%
           [5] Test Loss: 0.2485, Accuracy: 92.31%
           [6] Test Loss: 0.2207, Accuracy: 93.18%
           [7] Test Loss: 0.2026, Accuracy: 93.71%
           [8] Test Loss: 0.1851, Accuracy: 94.25%
           [9] Test Loss: 0.1743, Accuracy: 94.55%
           [10] Test Loss: 0.1636, Accuracy: 94.83%
           [11] Test Loss: 0.1539, Accuracy: 95.21%
           [12] Test Loss: 0.1481, Accuracy: 95.39%
           [13] Test Loss: 0.1409, Accuracy: 95.57%
           [14] Test Loss: 0.1377, Accuracy: 95.70%
           [15] Test Loss: 0.1325, Accuracy: 95.87%
           [16] Test Loss: 0.1317, Accuracy: 95.87%
           [17] Test Loss: 0.1267, Accuracy: 96.01%
           [18] Test Loss: 0.1245, Accuracy: 95.95%
           [19] Test Loss: 0.1207, Accuracy: 96.07%
           [20] Test Loss: 0.1190, Accuracy: 96.16%
           [21] Test Loss: 0.1147, Accuracy: 96.37%
           [22] Test Loss: 0.1126, Accuracy: 96.32%
           [23] Test Loss: 0.1101, Accuracy: 96.44%
           [24] Test Loss: 0.1097, Accuracy: 96.44%
           [25] Test Loss: 0.1070, Accuracy: 96.41%
           [26] Test Loss: 0.1037, Accuracy: 96.63%
           [27] Test Loss: 0.1002, Accuracy: 96.73%
           [28] Test Loss: 0.1031, Accuracy: 96.72%
           [29] Test Loss: 0.1020, Accuracy: 96.69%
           [30] Test Loss: 0.0989, Accuracy: 96.85%
           [31] Test Loss: 0.0996, Accuracy: 96.81%
           [32] Test Loss: 0.0976, Accuracy: 96.88%
           [33] Test Loss: 0.0953, Accuracy: 96.92%
           [34] Test Loss: 0.0960, Accuracy: 96.96%
           [35] Test Loss: 0.0931, Accuracy: 97.04%
           [36] Test Loss: 0.0923, Accuracy: 97.08%
           [37] Test Loss: 0.0952, Accuracy: 97.08%
           [38] Test Loss: 0.0937, Accuracy: 97.14%
           [39] Test Loss: 0.0914, Accuracy: 97.20%
           [40] Test Loss: 0.0903, Accuracy: 97.15%
           [41] Test Loss: 0.0904, Accuracy: 97.22%
           [42] Test Loss: 0.0882, Accuracy: 97.15%
           [43] Test Loss: 0.0902, Accuracy: 97.01%
           [44] Test Loss: 0.0870, Accuracy: 97.16%
           [45] Test Loss: 0.0879, Accuracy: 97.16%
           [46] Test Loss: 0.0864, Accuracy: 97.26%
           [47] Test Loss: 0.0856, Accuracy: 97.36%
           [48] Test Loss: 0.0856, Accuracy: 97.26%
           [49] Test Loss: 0.0866, Accuracy: 97.23%
           [50] Test Loss: 0.0858, Accuracy: 97.29%    



