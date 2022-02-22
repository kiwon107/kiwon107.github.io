var store = [{
        "title": "[확통] 1. 확률",
        "excerpt":"본 포스팅은 “프로그래머를 위한 확률과 통계” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.        ","categories": ["prob"],
        "tags": ["probaility"],
        "url": "/prob/prob1/",
        "teaser": null
      },{
        "title": "[확통] 2. 여러 확률변수의 조합 I",
        "excerpt":"본 포스팅은 “프로그래머를 위한 확률과 통계” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다.        ","categories": ["prob"],
        "tags": ["probaility"],
        "url": "/prob/prob2/",
        "teaser": null
      },{
        "title": "Test",
        "excerpt":"01-1. Test      hi   오 이게 되네   print('hi')  'hi'   01-2. Test2      와우   ","categories": ["test"],
        "tags": ["Test","test"],
        "url": "/test/test/",
        "teaser": null
      },{
        "title": "[Python] 1. 레퍼런스 카운트와 가비지 컬렉션",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 1-1. 가비지 컬렉션 다음 코드를 보자. laugh = '푸헤헿' 우리는 이를 보고 “변수 laugh에 문자열 ‘푸헤헿’을 저장했다.”고 표현한다. 하지만 상기 코드는 다음과 같음을 알고 있어야 한다. “변수 laugh가 문자열 ‘푸헤헿’을 참조(Reference)한다.” 즉, 포스트잇...","categories": ["pythonmd"],
        "tags": ["python","reference count","garbage collection"],
        "url": "/pythonmd/pythonmd1/",
        "teaser": null
      },{
        "title": "[Python] 2. 수정 가능한 객체와 수정 불가능한 객체",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 2-1. Immutable &amp; Mutable Mutable 객체: 객체가 지닌 값의 수정이 가능한 객체 Ex) 리스트, 딕셔너리 today = ['공부하자', '운동하자'] id(today) (결과) 1896625928712 today += ['요리하자', '일찍자자'] id(today) (결과) 1896625928712 Immutable 객체: 객체가 지닌...","categories": ["pythonmd"],
        "tags": ["python","immutable","mutable"],
        "url": "/pythonmd/pythonmd2/",
        "teaser": null
      },{
        "title": "[Python] 3. 깊은 복사와 얕은 복사",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 3-1. 두 객체의 비교와 복사 객체를 비교할 때, 헷갈리는 두 가지 유형의 연산자가 있다. ‘v1 == v2’ vs ‘v1 is v2’ v1 = [3, 6, 9] v2 = [3, 6, 9] v1 ==...","categories": ["pythonmd"],
        "tags": ["python","deep copy","shallow copy"],
        "url": "/pythonmd/pythonmd3/",
        "teaser": null
      },{
        "title": "[Python] 4. 리스트 컴프리헨션",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 4-1. 리스트 생성 방법 다음과 같이 for문을 활용하여 3, 6, 9를 요소로 갖는 리스트를 만들 수 있다. v1 = [1, 2, 3] v2 = [] for i in v1: v2.append(i * 3) v2...","categories": ["pythonmd"],
        "tags": ["python","list comprehension"],
        "url": "/pythonmd/pythonmd4/",
        "teaser": null
      },{
        "title": "[Python] 5. Iterable 객체와 Iterator 객체",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 5-1. Iter 함수 다음과 같은 코드를 본적이 있는가? spiderman = ['톰 홀랜드', '토비 맥과이어', '앤드류 가필드'] ir_spiderman = iter(spiderman) next(ir) next(ir) next(ir) (결과) '톰 홀랜드' '토비 맥과이어' '앤드류 가필드' 이게 어떻게 가능할까? iter...","categories": ["pythonmd"],
        "tags": ["python","list Iterable object","Iterator object"],
        "url": "/pythonmd/pythonmd5/",
        "teaser": null
      },{
        "title": "[Python] 6. 객체처럼 다뤄지는 함수 그리고 람다",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 6-1. 파이썬에서는 함수도 객체 파이썬은 모든 것을 객체로 처리한다. 참고로 클래스가 메소드와 변수로 구성된 코드 껍데기라면, 객체는 클래스의 코드가 작동되도록 영혼을 불어넣은 것이라 할 수 있다. x = 3.0 type(x) x.is_integer() # 소수점...","categories": ["pythonmd"],
        "tags": ["python","Object","Function","Lambda"],
        "url": "/pythonmd/pythonmd6/",
        "teaser": null
      },{
        "title": "[Python] 7. Map & Filter",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 7-1. Map &amp; Filter 설명 없이 바로 코드부터 들어간다. def square(n): return n ** 2 num = [2, 3, 5] num_square = list(map(pow, num)) num_square (결과) [4, 9, 25] 이것만 봐도 map 함수가...","categories": ["pythonmd"],
        "tags": ["python","Map","Filter"],
        "url": "/pythonmd/pythonmd7/",
        "teaser": null
      },{
        "title": "[Python] 8. 제너레이터 함수",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 8-1. 제너레이터에 대한 이해와 제너레이터 함수 전 시간에 iterator 객체에 대해 공부하였다. 제너레이터는 iterator 객체의 한 종류이다. 그러므로, 제너레이터를 전달하고 next 함수를 호출하면 값을 하나씩 얻을 수 있다. 제너레이터를 만드는 방법은 크게 두가지가...","categories": ["pythonmd"],
        "tags": ["python","Generator"],
        "url": "/pythonmd/pythonmd8/",
        "teaser": null
      },{
        "title": "[Python] 9. 튜플 패킹 & 언패킹, 네임드 튜플",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 9-1. 패킹과 언패킹 튜플 패킹: 하나 이상의 값을 튜플로 묶는 행위 tpl_one = (12, 15) tpl_two = 12, 15 tpl_one tpl_two (결과) (12, 15) (12, 15) 투플 언패킹: 튜플에 묶여 있는 값들을 풀어내는...","categories": ["pythonmd"],
        "tags": ["python","tuple","packing","unpacking"],
        "url": "/pythonmd/pythonmd9/",
        "teaser": null
      },{
        "title": "[Python] 10. dict의 생성과 zip / dict의 루핑 기술과 컴프리헨션",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 10-1. dict 생성 방법 d1 = {'a': 1, 'b': 2, 'c':3} d2 = dict([('a', 1), ('b', 2), ('c', 3)]) d3 = dict(a=1, b=2, c=3) # 키가 문자열일 때! d4 = dict(zip(['a', 'b', 'c'],...","categories": ["pythonmd"],
        "tags": ["python","dict","view"],
        "url": "/pythonmd/pythonmd10/",
        "teaser": null
      },{
        "title": "[Python] 11. 함수 호출과 매개변수 선언시 *와 **의 사용 규칙",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 11-1. iterable 객체와 매개변수 function(*iterable): 리스트, 튜플, 문자열(iterable 객체)을 풀어서 전달 function(**iterable): 딕셔너리 값을 풀어서 전달 def out(a, b, c): print(a, b, c, sep=', ') a = [1, 2, 3] b = (1,...","categories": ["pythonmd"],
        "tags": ["python","dict","view"],
        "url": "/pythonmd/pythonmd11/",
        "teaser": null
      },{
        "title": "[Python] 12. dict & defaultdict & OrderedDict",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 12-1. 키가 존재할 때와 존재하지 않을 때 딕셔너리에 키 존재 시 대입 연산 → 값의 수정 딕셔너리에 해당 키 존재하지 않을 시 대입 연산 → 새로운 키와 값의 추가 키 저장되어 있는 상태에서...","categories": ["pythonmd"],
        "tags": ["python","defaultdict","OrderedDict"],
        "url": "/pythonmd/pythonmd12/",
        "teaser": null
      },{
        "title": "[논문] 1. A Comprehensive Survey on Graph Neural Networks",
        "excerpt":"01-1. 그래프 그래프 $G$ 는 다음과 같이 정의된다. $G = (V, E)$. $V$는 노드들의 집합이고, $E$는 엣지들의 집합이다. 노드 $v_{i}$는 $v_{i}\\in V$ 이며, $v_{j}$와 $v_{i}$를 잇는 엣지 $e_{ij}$는 $e_{ij}=(e_{i}, e_{j})\\in E$ 이다. 노드 $v$의 이웃은 $ N(v)=\\{ u \\in V|(v,u) \\in E \\} $ 이다. 인접행렬 $\\mathbf{A}$는 $n \\times n$...","categories": ["paper"],
        "tags": ["paper"],
        "url": "/paper/paper1/",
        "teaser": null
      },{
        "title": "[Python] 13. 지료형 분류와 set & frozenset",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 13-1. 자료형 분류 시퀀스 타입: 저장된 값의 순서 정보 존재 Ex) 리스트, 튜플, 레인지, 문자열 인덱싱 연산: 특정 값 하나 참조 Ex) a[0], a[1], … 슬라이싱 연산: 시작과 끝 정하여 참조 Ex) a[0:3]...","categories": ["pythonmd"],
        "tags": ["python","set","frozenset"],
        "url": "/pythonmd/pythonmd13/",
        "teaser": null
      },{
        "title": "[Python] 14. 정렬 기술, enumerate와 문자열 비교",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 14-1. 리스트의 sort 메소드 l = [3, 4, 1, 2] l.sort() # 오름차순 l l.sort(reverse = True) # 내림차순 l (결과) [1, 2, 3, 4] [4, 3, 2, 1] def name(t): return t[0]...","categories": ["pythonmd"],
        "tags": ["python","sort","sorted","enumerate"],
        "url": "/pythonmd/pythonmd14/",
        "teaser": null
      },{
        "title": "[Python] 15. 표현식 / 메소드 기반 문자열 조합",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 15-1. 문자열 조합 friend = ('Wonny', 33) print('My friend', friend[0], 'is', str(friend[1]), 'years old') print('My friend ' + friend[0] + ' is ' + str(friend[1]) + ' years old') 15-2. 표현식 기반 문자열...","categories": ["pythonmd"],
        "tags": ["python","string formatting expressions","string formatting method calls"],
        "url": "/pythonmd/pythonmd15/",
        "teaser": null
      },{
        "title": "[Python] 16. 클래스와 객체의 본질",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 16-1. 객체 안에 변수가 만들어지는 시점 클래스: 객체를 만들기 위한 일종의 설계도로써 클래스 내 들어갈 변수(데이터)와 메소드(기능)을 결정함 객체: 클래스를 기반으로 만들어진 실제 사물 class Simple: # def __init__(self): # self.i = 0...","categories": ["pythonmd"],
        "tags": ["python","class","object"],
        "url": "/pythonmd/pythonmd16/",
        "teaser": null
      },{
        "title": "[논문] 2. Deep One-Class Classification (Deep SVDD)",
        "excerpt":"I. Motivation 해당 논문에서는 Deep Support Vector Data Description(Deep SVDD)를 소개한다. Deep SVDD는 초구(Hypersphere)의 부피를 최소화하는 신경망을 훈련함. 이 초구의 부피는 그림 2-1과 같이 데이터를 둘러싸서 정상 데이터의 범위를 나타내는 역할을 함. 초구의 부피를 최소화 하도록 훈련된 이 신경망은 데이터 포인트들을 초구의 중앙에 가깝게 매핑시켜 정상 데이터 내 공통 요인을...","categories": ["paper"],
        "tags": ["paper"],
        "url": "/paper/paper2/",
        "teaser": null
      },{
        "title": "[Python] 17. 상속",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 17-1. 부모 클래스와 자식 클래스 다음 그림을 보자. 그림 17-1. 파이썬 상속의 예 (참조: https://techvidvan.com/tutorials/python-inheritance/) Vehicle 클래스: 부모클래스, 슈퍼클래스, 상위클래스 Car, Bus, Bike 클래스: 자식클래스, 서브클래스, 하위클래스 상속을 하면 부모클래스가 갖는 모든 메소드를...","categories": ["pythonmd"],
        "tags": ["python","inheritance"],
        "url": "/pythonmd/pythonmd17/",
        "teaser": null
      },{
        "title": "[Python] 18. isinstance 함수와 object 클래스",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 18-1. isinstance 함수 isinstance(object, classinfo) → 객체의 클래스 유형을 확인하는 함수 class Vehicle: pass class Car(Vehicle): pass class ElectricCar(Car): pass isinstance([1, 2], list) ec = ElectricCar() isinstance(ec, ElectricCar) isinstance(ec, Car) isinstance(ec, Vehicle) (결과)...","categories": ["pythonmd"],
        "tags": ["python","instance","object class"],
        "url": "/pythonmd/pythonmd18/",
        "teaser": null
      },{
        "title": "[Python] 19. 스페셜 메소드",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 19-1. 스페셜 메소드 스페셜 메소드: 이름을 명시하지 않고 다른 경로를 통해 or 상황에 따라 자동으로 호출되는 메소드 형태: __name__ t = (1, 2, 3) len(t) # == t.__len__() itr = iter(t) # ==...","categories": ["pythonmd"],
        "tags": ["python","special method","iterable","iterator"],
        "url": "/pythonmd/pythonmd19/",
        "teaser": null
      },{
        "title": "[Python] 20. 연산자 오버로딩",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 20-1. 연산자 오버로딩 간단히 이해하기 class Account: def __init__(self, aid, abl): self.aid = aid self.abl = abl def __add__(self, m): self.abl += m print('__add__') def __sub__(self, m): self.abl -= m print('__sub__') def __call__(self):...","categories": ["pythonmd"],
        "tags": ["python","operator overloading"],
        "url": "/pythonmd/pythonmd20/",
        "teaser": null
      },{
        "title": "[Python] 21. 정보은닉과 __dict__, __slots__의 효과",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 21-1. 속성 감추기 class Person: def __init__(self, n, a): self.__name = n self.__age = a def add_age(self, a): if(a &lt; 0): print('나이 정보 오류') else: self.__age += a def __str__(self): return '{0}: {1}'.format(self.__name,...","categories": ["pythonmd"],
        "tags": ["python","__","__dict__","__slots__"],
        "url": "/pythonmd/pythonmd21/",
        "teaser": null
      },{
        "title": "[Python] 22. 프로퍼티",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 22-1. 안전하게 접근하기 class Natural: def __init__(self, n): if(n&lt;1): self.__n = 1 else: self.__n = n def getn(self): return self.__n def setn(self, n): if(n&lt;1): self.__n = 1 else: self.__n = n def main():...","categories": ["pythonmd"],
        "tags": ["python","property","getter","setter"],
        "url": "/pythonmd/pythonmd22/",
        "teaser": null
      },{
        "title": "[Python] 23. 네스티드 함수와 클로저",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 23-1. 함수를 만들어서 반환하는 함수 네스티드 함수: 함수 안에 정의된 함수 def maker(m): def inner(n): return m * n return inner f1 = maker(2) f1(7) (결과) 14 23-2. 클로져 위 예제에서 m은 maker...","categories": ["pythonmd"],
        "tags": ["python","nested","closure"],
        "url": "/pythonmd/pythonmd23/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 1. 텐서와 Autograd",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 1-1. 텐서의 차원 자유자재로 다루기 파이토치 임포트 import torch 텐서: 파이토치에서 다양한 수식 계산하는데 사용하는 가장 기본적인 자료구조로써 숫자들을 특정한 모양으로 배열한 것 텐서는 ‘차원’ 또는 ‘랭크’ 라는 개념 가짐 랭크0 텐서: 스칼라,...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","tensor","autograd"],
        "url": "/deeplearningpyt/deeplearningpyt1/",
        "teaser": null
      },{
        "title": "[Python] 24. 데코레이터",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 24-1. 데코레이터에 대한 이해 def smile(): print(\"^_^\") def confused(): print(\"@_@\") def deco(func): def df(): print('emotion!') func() print('emotion!') return df smile = deco(smile) smile() confused = deco(confused) confused() (결과) emotion! ^_^ emotion! emotion! @_@...","categories": ["pythonmd"],
        "tags": ["python","decorator"],
        "url": "/pythonmd/pythonmd24/",
        "teaser": null
      },{
        "title": "[Python] 25. 클래스 메소드와 static 메소드",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 25-1. 클래스 변수에 대한 이해 class Simple: def __init__(self): self.iv = 10 # iv는 인스턴스 변수(첫 대입 연산에서 생성되는 변수로써 객체별로 존재) s = Simple() s.iv # 인스턴스 변수는 개게 통해 접근! (결과)...","categories": ["pythonmd"],
        "tags": ["python","class method","static method"],
        "url": "/pythonmd/pythonmd25/",
        "teaser": null
      },{
        "title": "[Python] 26. __name__ & __main__",
        "excerpt":"본 포스팅은 “윤성우의 열혈 파이썬 중급편” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 26-1. name # who_are_you.py def main(): print('file name: who_are_you.py') print('__name__: {0}'.format(__name__)) main() (결과) file name: who_are_you.py __name__: __main__ import who_are_you # who_are_you.py의 main 함수 실행 print('play importer') print('__name__: {0}'.format(__name__)) (결과) file name: who_are_you.py...","categories": ["pythonmd"],
        "tags": ["python","__name__"],
        "url": "/pythonmd/pythonmd26/",
        "teaser": null
      },{
        "title": "[Machine learning] 1. 마켓과 머신러닝",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 1-1. 생선 분류 문제 생선을 분류하는 문제이다. 도미를 분류하고 싶은데, 전문가는 생선 길이가 30cm 이상이면 도미라고 알려줬다. if fish_length &gt;= 30: print(\"도미\") 위 코드로 도미를 제대로 분류할 수 있을까? 다른 생선도 분명 30cm 이상의...","categories": ["machinelearning"],
        "tags": ["machine learning","feature","training","K-Nearest Neighbors","accuracy"],
        "url": "/machinelearning/machinelearning1/",
        "teaser": null
      },{
        "title": "[Machine learning] 2. 훈련 세트와 테스트 세트",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 2-1. 지도학습과 비지도학습 지도학습 알고리즘은 훈련하기 위한 데이터와 정답이 필요하다. 1장에서 훈련시킨 kn 모델이 지도학습 모델의 일종이다. 지도학습에서 데이터와 정답을 각각 입력과 타깃 이라 한다. 그리고 이 둘을 합쳐 훈련데이터 라고 한다. 1장에서 입력으로...","categories": ["machinelearning"],
        "tags": ["machine learning","supervised learning","unsupervised learning","train set","test set"],
        "url": "/machinelearning/machinelearning2/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 2. 경사하강법으로 이미지 복원하기",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 2-1. 오염된 이미지 문제와 복원 방법 오염된 이미지와 이미지 처리 함수 weird_function()을 이용하여 원본 이미지를 복원하는 문제이다. 다음과 같은 사고 과정을 거쳐보자! 오염된 이미지와 같은 크기의 랜덤 텐서 생성 랜덤 텐서를 weird_function() 함수에...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","gradient descent","example","practice"],
        "url": "/deeplearningpyt/deeplearningpyt2/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 3. 신경망 모델 구현하기",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 3-1. 인공 신겸망(ANN) 인공 신경망(Artificial Neural Network)는 인간의 뇌 혹은 신경계의 작동 방식에서 영감을 받았다. 입력층: 인공 신경망에서 자극을 입력받는 감각기관에 해당하는 부분 은닉층: 입력층을 거친 자극을 처리해 다음 은닉층(인접한 신경세포)로 전달하는 부분....","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","ANN"],
        "url": "/deeplearningpyt/deeplearningpyt3/",
        "teaser": null
      },{
        "title": "[Machine learning] 3. 데이터 전처리",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 3-1. 넘파이로 데이터 준비하기 2장에서 썼던 도미와 빙어 데이터를 그대로 활용해보자. 넘파이까지 임포트 할 거다. fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0,...","categories": ["machinelearning"],
        "tags": ["machine learning","data preprocessing","standard score","broadcasting"],
        "url": "/machinelearning/machinelearning3/",
        "teaser": null
      },{
        "title": "[Machine learning] 4. k-최근접 이웃 회귀",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 4-1. k-최근접 이웃 회귀 지도 학습 알고리즘은 크게 분류와 회귀로 나뉜다. 회귀는 임의의 어떤 숫자를 예측하는 문제이다. 예를 들면 배달이 도착할 시간 예측 같은 문제이다. 생선의 무게를 예측하는 것도 회귀 문제이다. k-최근접 이웃 알고리즘은...","categories": ["machinelearning"],
        "tags": ["machine learning","regression","k-neighbor regressor","coefficient of determination","overfitting","underfitting"],
        "url": "/machinelearning/machinelearning4/",
        "teaser": null
      },{
        "title": "[Machine learning] 5. 선형회귀",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 5-1. K-최근접 이웃의 한계 이전 문제에서 길이가 훨씬 더 긴 농어에 대한 무게를 구해보자. import numpy as np perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0, 21.0, 21.0, 21.3,...","categories": ["machinelearning"],
        "tags": ["machine learning","linear regression","coefficient","weight","model parameter","polynomial regression"],
        "url": "/machinelearning/machinelearning5/",
        "teaser": null
      },{
        "title": "[Machine learning] 6. 특성 공학과 규제",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 6-1. 다중 회귀 5장에서 하나의 특성(생선 길이)을 사용하여 선형 회귀 모델을 훈련시켰다. 여러 개의 특성을 사용한 선형 회귀를 다중 회귀라고 한다. 1개의 특성을 사용하면 직선을 학습한다. 2개의 특성을 사용하면 선형 회귀는 평면을 학습한다. 그...","categories": ["machinelearning"],
        "tags": ["machine learning","multiple regression","feature engineering"],
        "url": "/machinelearning/machinelearning6/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 4. Fashion MNIST 데이터셋 알아보기",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 4-1. Fashion MNIST 딥러닝에서는 모델보다 좋은 데이터 셋이 더 중요하다. 데이터셋은 우리가 풀고자 하는 문제를 정의하는 역할을 한다고 봐야한다. 문제 정의가 잘못되면 풀이를 아무리 열심히해도 말짱 도루묵이다. Fashion MNIST는 28 x 28 픽셀을...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","DNN"],
        "url": "/deeplearningpyt/deeplearningpyt4/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 5. 인공 신경망으로 패션 아이템 분류하기",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 5-1. 환경 설정하기 import torch import torch.nn as nn # 파이토치, 인공 신경망 모델의 재료들 담고 있는 모듈 import torch.optim as optim # 최적화 import torch.nn.functional as F from torchvision import transforms, datasets...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","DNN"],
        "url": "/deeplearningpyt/deeplearningpyt5/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 6. 과대적합과 드롭아웃",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 6-1. 과대적합, 과소적합, 조기종료 머신러닝 모델을 만들면 학습 성능은 잘 나오지만, 테스트셋이나 실제 상황에서는 성능이 나오지 않을 때가 있다. 이것을 과대적합(Overfitting) 이라고 한다. 즉, 너무 학습 데이터에만 치중되어 새로운 데이터에 대해서는 성능이 잘...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","DNN"],
        "url": "/deeplearningpyt/deeplearningpyt6/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 7. CNN 기초와 모델 구현",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 7-1. 컴퓨터가 보는 이미지 컴퓨터에서 모든 이미지는 픽셀값들을 가로, 세로로 늘어놓은 행렬로 표현된다. 보통 인공 신경망은 다양한 형태의 입력에 대한 확정성이 떨어진다. 같은 신발 이미지라고 해도, 신발이 옆으로 조금만 치우쳐지면 예측률이 급격히 떨어진다....","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","CNN"],
        "url": "/deeplearningpyt/deeplearningpyt7/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 8. ResNet으로 컬러 데이터셋에 적용하기",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 8-1. ResNet 소개 ResNet(Residual Network) 모델은 CNN을 응용한 모델이다. 이미지 천만 장을 학습하여 15만 장으로 인식률을 겨루는 이미지넷 대회에서 2015년도 우승한 모델. 신경망을 깊게 쌓으면 오히려 성능이 나빠지는 문제를 해결하는 방법으로 제시했다. 컨벌루션층의...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","CNN","ResNet"],
        "url": "/deeplearningpyt/deeplearningpyt8/",
        "teaser": null
      },{
        "title": "[자료구조와 알고리즘] 1. 알고리즘이란?",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 1-1. 세 정수의 최대값 구하기 print('세 정수의 최대값을 구하라.') a = int(input('정수 a 값 입력: ')) # input() 함수는 키보드로 문자열 입력받아 반환 b = int(input('정수 b 값 입력: ')) # int() 함수는...","categories": ["algopy"],
        "tags": ["Data structure","Algorithm"],
        "url": "/algopy/algopy1/",
        "teaser": null
      },{
        "title": "[Machine learning] 7. 로지스틱 회귀",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 7-1. 럭키백의 확률 럭키백에 들어간 생선의 크기, 무게 등이 주어졌을 때, 7개 생선에 대한 확률을 출력해야 한다고 하자. 길이, 높이, 두께, 대각선 길이, 무게를 특성으로 사용할 수 있다고 한다. 사이킷런의 K-최근접 이웃 분류기로 클래스...","categories": ["machinelearning"],
        "tags": ["machine learning","logistic regression","multi-class classification","sigmoid function","softmax function"],
        "url": "/machinelearning/machinelearning7/",
        "teaser": null
      },{
        "title": "[Machine learning] 8. 확률적 경사 하강법",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 8-1. 점진적인 학습 훈련데이터가 항상 한 번에 준비되면 너무 좋을 것이다. 그러나 현실은 녹록치 않다. 만약 데이터가 조금씩 꾸준히 전달되는 경우라면 어떻게 해야할까? 새로운 데이터를 추가할 때 마다 이전 데이터를 버려서 훈련 데이터 크기를...","categories": ["machinelearning"],
        "tags": ["machine learning","stochastic gradient descent","loss function","epoch"],
        "url": "/machinelearning/machinelearning8/",
        "teaser": null
      },{
        "title": "[Machine learning] 9. 결정 트리",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 9-1. 로지스틱 회귀로 와인 분류하기 와인 데이터를 한번 봐보자. import pandas as pd wine = pd.read_csv('https://bit.ly/wine_csv_data') wine.head() (결과) alcohol sugar pH class 0 9.4 1.9 3.51 0.0 1 9.8 2.6 3.20 0.0 2 9.8...","categories": ["machinelearning"],
        "tags": ["machine learning","decision tree","impurity","information gain","feature importance"],
        "url": "/machinelearning/machinelearning9/",
        "teaser": null
      },{
        "title": "[Machine learning] 10. 교차 검증과 그리드 서치",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 10-1. 검증 세트 테스트 세트를 사용하지 않으면 모델이 과대적합인지 과소적합인지 판단하기가 어렵다. 테스트 세트를 사용하지 않고 이를 측정하려면 어떻게 해야 할까? 바로 훈련 세트를 나누는 것이다! 훈련 세트로부터 분리된 데이터를 검증 세트(Validation set) 라고...","categories": ["machinelearning"],
        "tags": ["machine learning","validation set","cross validation","grid search","random search"],
        "url": "/machinelearning/machinelearning10/",
        "teaser": null
      },{
        "title": "[Machine learning] 11. 트리의 앙상블",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 11-1. 정형 데이터와 비정형 데이터 csv, DB, 혹은 엑셀 같이 특성별로 정리된 형태의 데이터를 정형 데이터(Structured data) 라고 한다. 정형 데이터와 반대되는 데이터를 비정형 데이터(Unstructured data) 라고 한다. 책의 글과 같은 텍스트 데이터나 사진,...","categories": ["machinelearning"],
        "tags": ["machine learning","ensemble learning","random forest extra tree","gradient boosting"],
        "url": "/machinelearning/machinelearning11/",
        "teaser": null
      },{
        "title": "[Machine learning] 12. 군집 알고리즘",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 12-1. 타깃을 모르는 비지도 학습 비지도 학습(Unsupervised learning)은 타깃이 없을 때 사용하는 머신러닝 알고리즘이다. 사람이 가르쳐 주지 않아도 데이터에 있는 무언가를 학습한다. 12-2. 과일 사진 데이터 준비하기 !wget https://bit.ly/fruits_300_data -O fruits_300.npy 를 코랩에 쳐서...","categories": ["machinelearning"],
        "tags": ["machine learning","unsupervised learning","histogram","cluster"],
        "url": "/machinelearning/machinelearning12/",
        "teaser": null
      }]
