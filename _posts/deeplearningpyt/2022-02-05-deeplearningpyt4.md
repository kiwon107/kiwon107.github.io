---
layout: single
title: "[Deeplearning(pytorch)] 4. Fashion MNIST 데이터셋 알아보기"
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

## 4-1. Fashion MNIST
딥러닝에서는 모델보다 좋은 데이터 셋이 더 중요하다. 데이터셋은 우리가 풀고자 하는 문제를 정의하는 역할을 한다고 봐야한다. 문제 정의가 잘못되면 풀이를 아무리 열심히해도 말짱 도루묵이다.<br/>
Fashion MNIST는 28 x 28 픽셀을 가진 70,000 개의 흑백 이미지로 구성된다. 신발, 드레스, 가방 등 총 10가지 카테고리가 있다. 실제 서비스를 만들 때, 딥러닝 엔지니어는 대부분의 시간을 데이터를 가공하고 파이프라인을 만드는데 쓴다. 이처럼 데이터를 얻고 가공하는데 많은 시간이 소모되지만, 토치비전은 다행히 Fashion MNIST 데이터셋을 자동으로 내려받을 수 있게 해주고 심지어 학습 데이터를 나누는 일까지 해준다.<br/>

일단 이미지 데이터를 다루기 위한 파이토치와 토치비전 관련 몇가지 모듈을 확인해보자.<br/>
- `torch.utils.data`: 데이터셋 표준 정의, 데이터셋 불러오기, 자르기, 섞기 관련 도구들 제공. `torch.utils.data.Dataset`이 데이터셋의 표준 정의함. `torch.utils.data.DataLoader` 클래스는 `Dataset` 모듈을 상속하며, 인스턴스 입력으로 학습에 필요한 데이터를 로딩해줌.
- `torchvision.datasets`: `torch.utils.data.Dataset` 상속하는 이미지 데이터셋 모음. 패션 아이템 데이터셋은 여기에 들어있음.
- `torchvision.transforms`: 이미지 데이터셋에 쓸 여러가지 변환 필터 들어있음. 텐서로의 변환, 크기조절(resize), 크롭(crop)과 같은 이미지 수정 기능이 있으며, 밝기(brightness), 대비(contrast) 등 조절하는 기능도 있음.
- `torchvision.utils`: 이미지 데이터 저장 및 시각화 위한 도구 들어있음.

```python
from torchvision import datasets, transforms, utils
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
```

이제 이미지를 텐서로 바꿔주는 코드를 입력하자. 참고로 `torchvision.transforms` 안에 있는 주요 기능들은 다음과 같다.
- `ToTensor`: 이미지를 파이토치 텐서로 변환
- `Resize`: 이미지 크기 조정
- `Normalize`: 주어진 평균과 표준편차 이용하여 정규화
- `RandomHorizontalFlip`: 무작위로 이미지 오른쪽과 왼쪽 뒤집는 기능
- `RandomCrop`: 이미지를 무작위로 자르는 기능

```python
transform = transforms.Compose([transforms.ToTensor()])
```

위에서는 `ToTensor()` 함수만 썻지만, `transforms.Compose()` 함수안의 리스트에 여러가지 기능 넣어주면 순서대로 변환이 이루어진다!

이제 `torchvision.datasets` 패키지를 활용하여 데이터셋을 내려받고 `Compose()` 함수로 만든 이미지 변환 설정을 적용하자.
```python
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
```

참고로 train 매개변수를 True 또는 False로 설정하느냐에 따라 학습용 트레이닝셋과 성능 평가용 테스트셋으로 나눠지게 된다.<br/>

`torchvision.datasets`으로 생성된 객체는 파이토치 내부 클래스 `torch.utils.data.Dataset`을 상속한다. 이에따라, 파이토치의 `DataLoader` 클래스를 바로 사용할 수 있다. `DataLoader`는 데이터셋을 배치라는 작은 단위로 쪼개고 학습 시 반복문 안에서 데이터를 공급해주는 클래스이다. 여기서 배치란 한 번에 처리하는 데이터 개수를 의미한다. 즉, 반복마다 배치 크기 만큼의 개수를 갖는 데이터(여기서는 이미지)를 읽어 훈련하게 된다. 컴퓨터 메모리 공간에 여유가 있으면 크게 해도 되고, 여유가 없다면 작게 해도 된다. 여기서는 배치크기를 16으로 하겠다.
```python
batch_size = 16

train_loader = data.DataLoader(
  dataset = trainset,
  batch_size = batch_size
)

test_loader = data.DataLoader(
  dataset = testset,
  batch_size = batch_size
)
```

데이터로더가 준비되었으니 편리하게 데이터를 뽑아 쓸 수 있다. `iter()` 함수를 사용하여 데이터로더를 iterator 객체로 만들고 `next()` 함수로 데이터를 가져와보자.
```python
dataiter = iter(train_loader)
images, labels = next(dataiter)
```

images와 labes에는 설정한 배치 크기만큼의 이미지와 라벨이 들어있을 것이다. 토치비전의 `utils.make_grid()` 함수를 이용하여 여러 이미지를 모아 하나의 이미지로 만들어보자. 이때 이미지는 파이토치 텐서 자료형이므로, `numpy()` 함수를 사용하여 넘파이 행렬로 바꿔줘야 한다. 그래야 `matplotlib`과 호환이 되어 시각화를 할 수 있다. 또한, `matplotlib`이 인식할 수 있는 차원으로 만들기위해, `np.transpose()` 함수를 사용하여 첫 번째 차원을 맨 뒤로 보낼 것이다.
```python
img = utils.make_grid(images, padding=0)
npimg = img.numpy()
plt.figure(figsize=(10, 7))
plt.imshow(np.transpose(npimg, (1,2,0)))
plt.show()
```
![그림 4-1. 코드 결과](/assets/images/deeplearningpyt/4-1.png)
{: .align-center}
그림 4-1. 코드 결과

여러 개의 패션 아이템이 나열되어 나왔다. 각 변형 함수를 통해 데이터의 shape이 어떻게 바뀌었는지 확인해보자.
```python
nt = np.transpose(npimg, (1,2,0))

print(images.shape)  # data.DataLoader 함수로 배치 크기 만큼 불러온 이미지 원본
print(img.shape)  # 여러 이미지 모아 하나의 이미지로 만든 형태
print(npimg.shape)  # 넘파이 행렬 형태로 변환
print(nt.shape)  # 시각화 위해 첫번째 차원을 끝차원으로 이동
```
    (결과) torch.Size([16, 1, 28, 28])
           torch.Size([3, 56, 224])
           (3, 56, 224)
           (56, 224, 3)

라벨도 확인해보자.
```python
print(labels)
```
    (결과) tensor([9, 0, 0, 3, 0, 2, 7, 2, 5, 5, 0, 9, 5, 5, 7, 9])

각각의 클래스들은 티셔츠/윗옷, 바지, 스웨터, 드레스, 코트, 샌들, 셔츠, 운동화, 가방, 앵클부츠 이렇게 10가지이다. 데이터셋에는 이름 대신 숫자 번호로 레이블이 주어진다. 해석하기 편하도록 다음과 같이 딕셔너리 형태로 변환하자.
```python
CLASSES = {
  0: 'T-shirt/top',
  1: 'Trouser',
  2: 'Pullover',
  3: 'Dress',
  4: 'Coat',
  5: 'Sandal',
  6: 'Shirt',
  7: 'Sneaker',
  8: 'Bag',
  9: 'Ankle boot',
}

for label in labels:
  index = label.item()  # 텐서 형태의 값을 가져오려면 item() 함수 써야하나보다!
  print(CLASSES[index])
```
    (결과) Ankle boot
           T-shirt/top
           T-shirt/top
           Dress
           T-shirt/top
           Pullover
           Sneaker
           Pullover
           Sandal
           Sandal
           T-shirt/top
           Ankle boot
           Sandal
           Sandal
           Sneaker
           Ankle boot

마지막으로 개별 이미지를 시각화해보자. 개별 이미지는 3차원 데이터인데, 흑백이미지라 마지막 차원의 크기는 1이다. 칼라이미지 였다면 RGB 채널마다 값이 있어 마지막 차원의 크기는 3일 것이다. 2차원 형태로 시각화하기 위해, `squeeze()` 함수를 이용하여 차원의 크기가 1인 마지막 차원을 날리자.
```python
idx = 1
item_img = images[idx]
item_npimg = item_img.squeeze().numpy()  # 크기가 1인 마지막차원 날리고 넘파이 형태로 변환
plt.title(CLASSES[labels[idx].item()])
plt.imshow(item_npimg, cmap='gray')
plt.show()
```
![그림 4-2. 코드 결과](/assets/images/deeplearningpyt/4-2.png)
{: .align-center}
그림 4-2. 코드 결과