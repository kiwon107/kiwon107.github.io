---
layout: single
title: "[Deeplearning(pytorch)] 5. 인공 신경망으로 패션 아이템 분류하기"
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

## 5-1. 환경 설정하기
```python
import torch
import torch.nn as nn # 파이토치, 인공 신경망 모델의 재료들 담고 있는 모듈
import torch.optim as optim # 최적화
import torch.nn.functional as F
from torchvision import transforms, datasets
```

`torch.cuda.is_available()` 함수는 현재 컴퓨터에서 CUDA를 이용할 수 있는지 알아보는 함수이다. CUDA 용 Pytorch를 설치하고 CUDA도 제대로 설치했다면 True, CUDA를 설치하지 않았거나 오류가 있다면 False를 반환한다. 이 값을 기준으로 CUDA를 지원하면 'cuda'를, 아니면 'cpu'를 torch.device에 설정하자.
```python
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
```

`DEVICE` 변수는 나중에 텐서와 가중치에 대한 연산 수행시 CPU와 GPU 중 어디서 실행할지 결정할 때 쓰인다.

미니배치의 크기를 64개로 설정하고, 에폭은 30으로 설정하자. 그리고 미니 배치만큼 데이터를 쪼개자. 에폭이란 학습 데이터 전체를 총 몇번이나 볼 것인가에 대한 설정이다.
```python
EPOCHS = 30
BATCH_SIZE = 64

transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.FashionMNIST(
  root = './.data/',
  train = True,
  download = True,
  transform = transform
)
testset = datasets.FashionMNIST(
  root = './.data/',
  train = False,
  download = True,
  transform = transform
)

train_loader = data.DataLoader(
  dataset = trainset,
  batch_size = BATCH_SIZE,
  shuffle = True,
)

test_loader = data.DataLoader(
  dataset = testset,
  batch_size = BATCH_SIZE,
  shuffle = True,
)
``` 

참고로, 배치 크기와 모델 가중치가 2의 거듭제곱으로 설정된 경우가 많다. 그 이유는, CPU와 GPU 메모리 크기가 보통 2의 배수이기 때문에, 배치 크기가 2의 n승이면 메모리에서 데이터를 주고 받는 효율을 높일 수 있다고 한다.

## 5-2. 이미지 분류를 위한 인공 신경망 구현
레이어가 3개인 3층 인공 신경망을 구현하자.
먼저 생성자에 우리 모델의 가중치 변수들이 들어가는 연산을 선언하자.
```python
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(784, 256) # 픽셀값 784개 입력받아 가중치를 행렬곱 하고 편향 더해 값 256개 출력
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 10)  # 마지막에 값 10개를 출력한다. 출력값 10개 각각은 클래스를 나타내며, 가장 큰 값을 갖는 클래스가 이 모델의 예측값이 된다.

  def forward(self, x):
    x = x.view(-1, 784)  # (64 x 1 x 28 x 28) 형태를 (64 x 784) 형태로 데이터 변환
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```
`nn.Linear` 클래스는 선형결합을 수행하는 객체를 생성한다.<br/>
더 자세한 데이터 흐름은 `forward()` 함수에 정의된다. 이전에는 `ReLU()` 활성화 함수 사용시, `torch.nn.ReLU()` 함수를 이용했다. 가중치가 없는 연산은 `torch.nn.Functional`에 있는 함수를 직접 사용하여 `ReLU()` 함수를 사용하기도 한다. 취향에 따라 `torch.nn.ReLU()` 또는 `torch.nn.fucntional.relu()`를 사용하자. 단, `torch.nn.ReLU()` 함수는 가중치가 있는 연산이므로, 생성자에 선언해주는 것이 좋다.

모델의 설계가 끝났으니, 이제 모델을 선언하자.<br/>
선언과 동시에 `to()` 함수로 연산을 CPU 또는 GPU에서 수행할지 정할 수 있다. CPU를 사용하려면 해당 함수를 지정할 필요 없으며, GPU를 사용하려면 `to("CUDA")`로 지정하여 모델의 파라미터들을 GPU 메모리로 보내야 한다.
```python
model = Net().to(DEVICE)
```

최적화 알고리즘은 `optim.SGD`를 사용하자. SGD는 모델 최적화를 위한 확률적 경사하강법이다. 학습률은 0.01로 설정하고, 모델 내부의 정보를 넘겨주는 `model.parameters()` 함수를 입력하자.
```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

이제 학습에 들어가는 연산 함수 `train()`를 만들자.
```python
def train(model, train_loader, optimizer):
  model.train()  # 학습 모드! 학습/평가 모드에 따라 동작이 다른 파이토치 모듈(드롭아웃 등)이 있다
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(DEVICE), target.to(DEVICE)  # 가중치를 GPU로 보냈다면, 학습 데이터도 같은 장치로 보내야 연산 수행이 가능하다!
    optimizer.zero_grad()   # 반복 때 마다 기울기 계산 새로 해야함
    output = model(data)
    loss = F.cross_entropy(output, target)  # output과 레이블 오차 구함
    loss.backward()  # 오차에 대한 기울기 계산!
    optimizer.step()  # 기울기값 활용하여 가중치 수정!
```

각 코드에 대한 설명은 주석으로 달아놨다. 참고로 `nn.CrossEntropyLoss`를 써도 되지만, 가중치를 보관할 필요가 없으므로, `torch.nn.functional.corss_entropy()` 함수를 사용하였다. `loss`는 미니배치 64개의 오차 평균이다!

## 5-3. 성능 측정하기
우리는 학습 데이터 뿐만 아니라, 전체 데이터에서 높은 성능을 보이는 모델을 원한다. 모든 데이터에 최적화 하는 것을 일반화(Generalization)라고 한다. 그리고 학습 데이터를 기반으로 만든 모델이 학습하지 않은 데이터에 얼마나 적응하는지 수치로 나타낸 것을 일반화 오류(Generalization Error) 라고 한다. 일반화 오류는 학습과 실제 성능의 괴리를 의미하기 때문에, 값이 작을수록 좋다.<br/>

세상 모든 데이터를 다 가질순 없기 때문에, 보통 학습 데이터의 일부를 떼어내어 평가용 데이터셋으로 활용한다. 평가용 데이터셋을 테스트셋 이라고 한다. Fashion MNIST는 비교적 규모가 작은 데이터셋 이기 때문에, 학습과 테스트 두가지로 나뉜다. 그러나 일반적으로 머신러닝 데이터셋은 **학습**, **검증**, **테스트** 3단계로 나뉜다. 학습용 데이터셋은 가중치 조절, 검증용 데이터셋은 배치 크기와 모델 설계 같은 하이퍼파라미터를 조절하는데 사용한다. 테스트셋은 성능 보고에 사용한다.<br/>

모델을 평가하기 위한 함수 `evaluate()`를 만들자. 이 함수는 에폭이 끝날 때마다 테스트셋으로 모델의 성능을 측정하는 역할을 한다.
```python
def evaluate(model, test_loader):
  model.eval()  # 평가 모드!
  test_loss = 0 # 테스트 오차
  correct = 0   # 예측이 맞은 수
  with torch.no_grad():  # 기울기 계산 안함!
    for data, target in test_loader:
      data, target = data.to(DEVICE), target.to(DEVICE)  # 데이터를 DEVICE로 보냄!
      output = model(data)  
      test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 교차 엔트로피 거칠 때 미니배치의 평균 대신 합을 받아오도록 함!
      pred = output.max(1, keepdim=True)[1]  # output.max는 첫번째 인자 방향(차원)으로 가장 큰 값과 그 값이 있는 인덱스 값 출력함. [1]로 인덱스 값을 가져오자!
      correct += pred.eq(target.view_as(pred)).sum().item()  # eq() 함수는 일치하면 1 아니면 0 출력. view_as() 함수는 target 텐서를 pred 모양대로 정렬해줌.
    
    test_loss /= len(test_loader.dataset)   # 평균 오차 계산
    test_accuracy = 100. * correct / len(test_loader.dataset)  # 정확도 산출
    return test_loss, test_accuracy
```

이제 코드를 돌리자!
```python
for epoch in range(1, EPOCHS + 1):
  train(model, train_loader, optimizer)
  test_loss, test_accuracy = evaluate(model, train_loader)

  print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))
```
    (결과) [1] Test Loss: 0.4785, Accuracy: 82.83%
           [2] Test Loss: 0.4724, Accuracy: 83.02%
           [3] Test Loss: 0.4601, Accuracy: 83.69%
           [4] Test Loss: 0.4316, Accuracy: 84.76%
           [5] Test Loss: 0.4393, Accuracy: 84.26%
           [6] Test Loss: 0.4233, Accuracy: 84.88%
           [7] Test Loss: 0.4036, Accuracy: 85.82%
           [8] Test Loss: 0.3938, Accuracy: 86.24%
           [9] Test Loss: 0.3867, Accuracy: 86.51%
           [10] Test Loss: 0.3809, Accuracy: 86.67%
           [11] Test Loss: 0.3735, Accuracy: 86.91%
           [12] Test Loss: 0.3915, Accuracy: 85.98%
           [13] Test Loss: 0.3684, Accuracy: 87.03%
           [14] Test Loss: 0.3683, Accuracy: 86.94%
           [15] Test Loss: 0.3637, Accuracy: 87.11%
           [16] Test Loss: 0.3462, Accuracy: 87.81%
           [17] Test Loss: 0.3532, Accuracy: 87.56%
           [18] Test Loss: 0.3402, Accuracy: 88.03%
           [19] Test Loss: 0.3456, Accuracy: 87.65%
           [20] Test Loss: 0.3334, Accuracy: 88.22%
           [21] Test Loss: 0.3318, Accuracy: 88.28%
           [22] Test Loss: 0.3277, Accuracy: 88.40%
           [23] Test Loss: 0.3690, Accuracy: 86.96%
           [24] Test Loss: 0.3341, Accuracy: 88.08%
           [25] Test Loss: 0.3109, Accuracy: 89.05%
           [26] Test Loss: 0.3081, Accuracy: 89.08%
           [27] Test Loss: 0.3087, Accuracy: 88.97%
           [28] Test Loss: 0.3281, Accuracy: 88.05%
           [29] Test Loss: 0.2982, Accuracy: 89.45%
           [30] Test Loss: 0.3120, Accuracy: 88.63%