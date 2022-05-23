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
        "excerpt":"본 포스팅은 “자료구조와 함께 배우는 알고리즘(파이썬)” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 1-1. 세 정수의 최대값 구하기 print('세 정수의 최대값을 구하라.') a = int(input('정수 a 값 입력: ')) # input() 함수는 키보드로 문자열 입력받아 반환 b = int(input('정수 b 값 입력: ')) # int() 함수는...","categories": ["algopy"],
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
      },{
        "title": "[Machine learning] 13. K-평균",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 13-1. K-평균 알고리즘 소개 k-평균 알고리즘 작동 방식은 다음과 같다. 1). 무작위로 k개의 클러스터 중심을 정한다. 2). 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정한다. 3). 클러스터에 속한 샘플의 평균값으로 클러스터...","categories": ["machinelearning"],
        "tags": ["machine learning","k-means","cluster center","elbow method"],
        "url": "/machinelearning/machinelearning13/",
        "teaser": null
      },{
        "title": "[Machine learning] 14. 주성분 분석",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 14-1. 차원과 차원 축소 13장에서 과일 사진은 10.000개의 픽셀이 있었다. 이는 10,000개의 특성이 있는것과 같다. 이런 특성을 머신러닝 에서는 차원(Dimension)이라고 한다. 이 차원을 줄일 수 있다면 저장 공간을 크게 절약할 수 있다. 참고로 다차원...","categories": ["machinelearning"],
        "tags": ["machine learning","dimensionality reduction","principal components analysis","explained variance"],
        "url": "/machinelearning/machinelearning14/",
        "teaser": null
      },{
        "title": "[Deeplearning(Tensorflow)] 1. 인공 신경망",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 1-1. 패션 MNIST 머신러닝에서 붓꽃 데이터셋이 유명하다면, 딥러닝에서는 MNIST 데이터셋이 유명하다. 이 데이터는 손으로 쓴 0~9까지의 숫자로 이루어져 있다. 텐서플로를 사용하여 이 데이터를 불러올 수 있다. 텐서플로의 케라스 패키지를 임포트하고 패션 MNIST 데이터를 다운로드하자....","categories": ["deeplearningtens"],
        "tags": ["deep learning","artificial neural network","tensorflow","dense layer","one-hot encoding"],
        "url": "/deeplearningtens/deeplearningtens1/",
        "teaser": null
      },{
        "title": "[Deeplearning(Tensorflow)] 2. 심층 신경망",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 2-1. 2개의 층 from tensorflow import keras (train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data() from sklearn.model_selection import train_test_split train_scaled = train_input / 255.0 train_scaled = train_scaled.reshape(-1, 28*28) train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2,...","categories": ["deeplearningtens"],
        "tags": ["deep learning","deep neural network","relu function","optimizer"],
        "url": "/deeplearningtens/deeplearningtens2/",
        "teaser": null
      },{
        "title": "[Deeplearning(Tensorflow)] 3. 신경망 모델 훈련",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 3-1. 손실 곡선 fit() 메소드로 모델을 훈련할 때 훈련 과정이 상세하게 출력되었다. 에포크, 횟수, 손실, 정확도 등이 있었다. 출력의 마지막에 다음과 같은 메세지도 있었다. &lt;tensorflow.python.keras.callbacks.History at 0x18340a10e10&gt; fit() 메소드는 History 클래스 객체를 반환한다. History...","categories": ["deeplearningtens"],
        "tags": ["deep learning","dropout","callback","early stopping"],
        "url": "/deeplearningtens/deeplearningtens3/",
        "teaser": null
      },{
        "title": "[Deeplearning(Tensorflow)] 4. 합성곱 신경망의 구성 요소",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 4-1. 합성곱 합성곱(Convolution)은 마치 입력 데이터에 마법의 도장을 찍어서 유용한 특성만 드러나게 하는 것이다. 인공 신경망은 처음에 가중치 $w_{1}$ ~ $w_{10}$ 과 절편 $b$를 랜덤하게 초기화한 다음 에포크를 반복하면서 경사 하강법 알고리즘을 사용하여 손실이...","categories": ["deeplearningtens"],
        "tags": ["deep learning","convolution","filter","feature map","padding","stride","pooling"],
        "url": "/deeplearningtens/deeplearningtens4/",
        "teaser": null
      },{
        "title": "[Deeplearning(Tensorflow)] 5. 합성곱 신경망을 사용한 이미지 분류",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 5-1. 패션 MNIST 데이터 불러오기 3장에서 완전 연결 신경망에 입력 이미지를 밀집층에 연결하기 위해 일렬로 펼쳤다. 합성곱 신경망은 2차원 이미지를 그대로 사용하므로 일렬로 펼칠 필요가 없다. 다만 입력 이미지는 항상 깊이 차원이 있어야 한다....","categories": ["deeplearningtens"],
        "tags": ["deep learning","Conv2D","MaxPooling2D","plot_model"],
        "url": "/deeplearningtens/deeplearningtens5/",
        "teaser": null
      },{
        "title": "[Deeplearning(Tensorflow)] 6. 합성곱 신경망의 시각화",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 6-1. 가중치 시각화 합성곱 층은 여러 개의 필터를 사용해 이미지에서 특징을 학습한다. 각 필터는 커널이라 부르는 가중치와 절편을 갖는다. 절편은 사실 시각적으로 의미가 있지는 않다. 가중치는 입력 이미지의 2차원 영역에 적용되어 어떤 특징을 크게...","categories": ["deeplearningtens"],
        "tags": ["deep learning","weights visualization","feature map visualization","functional API"],
        "url": "/deeplearningtens/deeplearningtens6/",
        "teaser": null
      },{
        "title": "[Machine learning] 2. 지도 학습",
        "excerpt":"본 포스팅은 “딥러닝 텐서플로 교과서” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 2-1. 지도학습 지도 학습은 정답(=레이블)을 컴퓨터에 미리 알려주고 데이터를 학습시키는 방법이다. 지도 학습에는 크게 분류와 회귀가 있다. 분류는 주어진 데이터를 정해진 범주에 따라 분류하고, 회귀는 데이터들의 특성을 기준으로 연속된 값을 그래프로 표현하여 패턴이나 트렌드를...","categories": ["tensortextbook"],
        "tags": ["machine learning","supervised learning","knn","svm","decision tree","logistic regression"],
        "url": "/tensortextbook/machinelearning16/",
        "teaser": null
      },{
        "title": "[Deeplearning(Tensorflow)] 7. 순차 데이터와 순환 신경망",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 7-1. 순차 데이터 순차 데이터(Sequential data)는 텍스트나 시계열 데이터(Time series data)와 같이 순서에 의미가 있는 데이터를 말한다. 텍스트 데이터는 단어의 순서가 중요한 순차 데이터이다. 이런 데이터는 순서를 유지하며 신경망에 주입해야 한다. 단어 순서를 마구...","categories": ["deeplearningtens"],
        "tags": ["deep learning","sequential data","recurrent neural network","cell","hidden state"],
        "url": "/deeplearningtens/deeplearningtens7/",
        "teaser": null
      },{
        "title": "[Deeplearning(Tensorflow)] 8. 순환 신경망으로 IMDB 리뷰 분류하기",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 8-1. IMDB 리뷰 데이터셋 IMDB 리뷰 데이터셋은 유명한 인터넷 영화 데이터베이스인 imdb.com 에서 수집한 리뷰를 감상평에 따라 긍정과 부정으로 분류해 놓은 데이터셋이다. 참고로 자연어 처리는 컴퓨터를 사용해 인간의 언어를 처리하는 분야인데, 음성 인식, 기계...","categories": ["deeplearningtens"],
        "tags": ["deep learning","corpus","token","one-hot encoding","word embedding"],
        "url": "/deeplearningtens/deeplearningtens8/",
        "teaser": null
      },{
        "title": "[Deeplearning(Tensorflow)] 9. LSTM과 GRU 셀",
        "excerpt":"본 포스팅은 “혼자 공부하는 머신러닝+딥러닝” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 9-1. LSTM 구조 LSTM(Long Shor-Term Memory)는 단기 기억을 오래 기억하기 위해 고안되었다. LSTM에는 입력과 가중치를 곱하고 절편을 더해 활성화 함수를 통과시키는 구조를 여러개 가지고 있다. 이런 계산 결과는 다음 타임스텝에 재사용 된다. 은닉상태를 먼저...","categories": ["deeplearningtens"],
        "tags": ["deep learning","LSTM","cell state","GRU"],
        "url": "/deeplearningtens/deeplearningtens9/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 9. 오토인코더 기초와 오토인코더 기반 이미지 특징 추출",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 9-1. 오토인코더 기초 데이터 형태와 관계없이 사람이 레이블링 하여 정답을 직접 알려주면 머신러닝 모델은 효율적으로 학습할 수 있다. 하지만 데이터셋에 정답이 포함되지 않으면 이야기가 달라진다. 정답 없이 주어진 데이터만으로 패턴을 찾는 것을 비지도학습이라...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","autoencoder"],
        "url": "/deeplearningpyt/deeplearningpyt9/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 10. 오토인코더로 망가진 이미지 복원하기",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 10-1. 잡음 제거 오토인코더 구현 앞서 설명한 것 처럼 오토인코더는 일종의 ‘압축’을 한다. 압축은 데이터의 특성에 우선순위를 매기고 낮은 순위의 데이터를 버린다는 뜻이다. 잡음 제거 오토인코더의 아이디어는 중요한 특징을 추출하는 오토인코더 특성을 이용하여...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","autoencoder"],
        "url": "/deeplearningpyt/deeplearningpyt10/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 11. RNN 개요와 영화 리뷰 감정 분석",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 11-1. RNN 개요 지금까지 배운 신경망은 연달아 있는 데이터의 순서와 상호작용을 인식하여 전체 상황을 이해하는 능력을 가지고 있지 않았다. 즉, 시간에 대한 개념이 없는 데이터와 그에 따른 신경망을 다룬 것이다. 앞서 이미지와 같은...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","rnn"],
        "url": "/deeplearningpyt/deeplearningpyt11/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 12. Seq2Seq 기계 번역",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 12-1. Seq2Seq 개요 언어를 다른 언어로 해석해주는 뉴럴 기계 번역(Neural machine translation) 모델이 있다. RNN 기반의 번역 모델인 Sequence to Sequence(=Seq2Seq) 모델은 기계 번역의 새로운 패러다임을 열었다. Seq2Seq 모델은 시퀀스를 입력받아 또 다른...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","rnn","seq2seq"],
        "url": "/deeplearningpyt/deeplearningpyt12/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 13. 딥러닝을 해킹하는 적대적 공격",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 13-1. 적대적 공격이란? 머신러닝 모델의 착시를 유도하는 입력을 적대적 예제(Adversarial example) 라고 한다. 적대적 예제를 생성해서 여러 가지 머신러닝 기반 시스템의 성능을 의도적으로 떨어뜨려 보안 문제를 일으키는 것을 적대적 공격(Adversarial attack) 이라고 한다....","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","adversarial attack"],
        "url": "/deeplearningpyt/deeplearningpyt13/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 14. 경쟁하며 학습하는 GAN",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 14-1. GAN 기초 GAN(Generative adversarial network)는 직역하면 적대적 생성 신경망이다. 단어 의미 하나하나 살펴보자. 먼저 GAN은 생성(Generative)을 하는 모델이다. CNN과 RNN은 새로운 이미지나 음성을 만들어내지 못한다. 그러나 GAN은 새로운 이미지나 음성을 창작하도록 고안되었다....","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","gan"],
        "url": "/deeplearningpyt/deeplearningpyt14/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 15. cGAN으로 생성 제어하기",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 15-1. cGAN으로 원하는 이미지 생성하기 14장에서의 GAN 모델은 ‘여러 종류의 패션 아이템 중 무엇을 생성하라!’ 라고 지시하는 로직이 없다. 즉, 사용자가 원하는 패션 아이템을 생성하는 능력은 없고 무작위 벡터를 입력받아 무작위로 패션 아이템을...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","cgan"],
        "url": "/deeplearningpyt/deeplearningpyt15/",
        "teaser": null
      },{
        "title": "[Deeplearning(pytorch)] 16. 주어진 환경과 상호작용하여 학습하는 DQN",
        "excerpt":"본 포스팅은 “펭귄브로의 3분 딥러닝, 파이토치맛” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 16-1. 강화학습과 DQN 기초 강화학습(Reinforcement learning)은 주어진 환경과 상호작용하여 좋은 점수를 얻는 방향으로 성장하는 머신러닝 분야이다. 그동안 배운 학습법들은 원하는 데이터셋을 외우는 주입식 학습법이었다. 강화학습은 자기주도적 학습법이라 할 수 있다. 강화학습 모델은 주어진...","categories": ["deeplearningpyt"],
        "tags": ["deep learning","pytorch","dqn"],
        "url": "/deeplearningpyt/deeplearningpyt16/",
        "teaser": null
      },{
        "title": "[Python] 1. 시작하기",
        "excerpt":"본 포스팅은 “파이썬 동시성 프로그래밍” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 1-1. 스레드 스레드란 운영체제에서 작동되는 스케줄링될 수 있는 인스트럭션(Instruction)의 순차적인 흐름이다. 일반적으로 스레드는 프로세스에 속해 있고, 프로그램 카운터, 스택, 레지스터를 비롯해 식별자로 구성된다. 스레드는 프로세서가 시간을 할당할 수 있는 최소 단위의 실행이라고도 할 수...","categories": ["pyconcur"],
        "tags": ["Concurrency in python"],
        "url": "/pyconcur/pyconcur1/",
        "teaser": null
      },{
        "title": "[Python] 2. 병렬화",
        "excerpt":"본 포스팅은 “파이썬 동시성 프로그래밍” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 2-1. 동시성에 대한 이해 동시성은 한 사람이 여러 작업을 수행하고, 그 작업을 빠르게 바꾸며 진행하는 모습이라 할 수 있다. 동시성 시스템에서 다음과 같은 특징을 볼 수 있다. 다양한 구성: 여러 프로세서와 스레드가 각자의 작업에...","categories": ["pyconcur"],
        "tags": ["Parallization in python"],
        "url": "/pyconcur/pyconcur2/",
        "teaser": null
      },{
        "title": "[Python] 3. 멀티프로세싱 1",
        "excerpt":"본 포스팅은 “파이썬 동시성 프로그래밍” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 3-1. GIL 작업 전역 인터프리터 락(GIL, Global Interpreter Lock)은 CPU 기반 작업에서 성능을 저해하는 메커니즘이다. 멀티프로세싱을 이용하면 이러한 한계를 극복할 수 있다. 파이썬에서는 CPU의 독립적인 코어에서 실행가능한 여러 프로세스를 실행할 수 있다. 간단한 출력문을...","categories": ["pyconcur"],
        "tags": ["Parallization in python","multiprocessing"],
        "url": "/pyconcur/pyconcur3/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 1. 네트워크의 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 1-1. 컴퓨터 네트워크란? 그림 1-1 처럼 컴퓨터 간의 네트워크를 연결한 것이 컴퓨터 네트워크이다. 두 대 이상의 컴퓨터가 연결되어 있으면 컴퓨터 네트워크라 할 수 있고, 네트워크를 통해 컴퓨터 간 필요한 데이터(정보)를 주고 받을 수 있다. 그림...","categories": ["comm"],
        "tags": ["Network","패킷"],
        "url": "/comm/comm1/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 2. 정보의 양을 나타내는 단위",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 2-1. 비트와 바이트 모든 컴퓨터는 숫자 0과 1만을 다룬다. 0과 1의 집합을 디지털 데이터라고 한다. 굉장히 많은 0과 1이 있으면 이미지 데이터도 나타낼 수 있다. 이 0과 1의 정보를 나타내는 최소 단위를 비트(Bit) 라고 한다....","categories": ["comm"],
        "tags": ["Network","Bit","Byte","ASCII 코드"],
        "url": "/comm/comm2/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 3. 랜과 왠",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 3-1. 랜과 왠의 차이 건물 안이나 특정 지역을 범위로 하는 네트워크를 랜(LAN) 이라고 한다. 가정이나 빌딩 안에 있는 사무실 같이 지리적으로 제한된 곳에서 컴퓨터와 프린터를 연결할 수 있는 네트워크이다. 반면 지리적으로 넓은 범위에 구축된 네트워크를...","categories": ["comm"],
        "tags": ["Network","LAN","WAN"],
        "url": "/comm/comm3/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 4. 가정에서 하는 랜 구성",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 4-1. 가정에서의 네트워크 구성 네트워크는 크게 랜과 왠으로 나뉜다고 앞에서 설명했다. 집에서 구성하는 네트워크는 랜이다. 인터넷을 사용하려면 먼저 인터넷 서비스 제공자(ISP)와 인터넷 회선이 필요하다. 그림 4-1. 가정에서의 랜 구성 인터넷 개통할 때 인터넷 서비스 제공자와...","categories": ["comm"],
        "tags": ["Network","가정에서의 네트워크"],
        "url": "/comm/comm4/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 5. 회사에서 하는 랜 구성",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 5-1. 소규모 회사에서의 네트워크 구성 가정에서의 랜 구성과 다른 점은 DMZ(DeMilitarized Zone)라는 네트워크 영역이 있다는 것이다. DMZ는 외부에 서버를 공개하기 위한 네트워크이다. 웹 사이트를 불특정 다수의 외부 사용자에게 공개하려면 웹 서버, 외부 사용자와 메일을 주고...","categories": ["comm"],
        "tags": ["Network","회사에서의 네트워크"],
        "url": "/comm/comm5/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 6. 네트워크의 규칙",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 6-1. 프로토콜이란? 네트워크에서 문제없이 통신하려면 규칙(약속)을 지켜야 한다. 노드 간 통신으로 데이터를 주고 받을 때 어떤 식으로 데이터를 주고 받자! 라는 규칙이 바로 프로토콜(Protocol) 이다. 편지가 배송되는 대략적인 과정은 다음과 같다. 1) 편지를 쓴다. 2)...","categories": ["comm"],
        "tags": ["Network","LAN","WAN"],
        "url": "/comm/comm6/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 7. OSI 모델과 TCP/IP 모델",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 7-1. OSI 모델 옛날에는 같은 회사의 컴퓨터끼리만 통신이 가능했었다. 케이블을 연결하는 커넥터도 회사별로 다르면 더 곤란했다. 이런 일들 때문에, 공통으로 사용할 수 있는 표준 규격을 정해야만 했다. 표준 규격을 정하는 단체는 여러 곳이 있는데, 대표적인...","categories": ["comm"],
        "tags": ["Network","OSI 모델","TCP/IP 모델"],
        "url": "/comm/comm7/",
        "teaser": null
      },{
        "title": "[Python] 4. 멀티프로세싱 2",
        "excerpt":"본 포스팅은 “파이썬 동시성 프로그래밍” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 4-1. 멀티프로세싱 풀 파이썬 애플리케이션이 멀티프로세스로 동작하면, 멀티프로세싱 모듈 내 다양한 기능을 가진 Pool 클래스를 활용할 수 있다. Pool 클래스는 프로그램 내 여러 자식 프로세스를 쉽게 실행하고, 풀에서 작업자를 선택할 수 있다. 1. concurrent.futures.ProcessPoolExecutor와...","categories": ["pyconcur"],
        "tags": ["Parallization in python","multiprocessing"],
        "url": "/pyconcur/pyconcur4/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 8. 캡슐화와 역캡슐화",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 8-1. 캡슐화와 역캡슐화 데이터를 보내려면 데이터 앞 부분에 전송하는데 필요한 정보를 붙여서 다음 계층으로 보내야한다. 즉 ‘필요한 정보-송신 데이터’ 형태로 다음 계층에 보내야한다는 것이다. 이 ‘필요한 정보’를 헤더라고 하며, 헤더는 데이터를 전달받을 상대방에 대한 정보도...","categories": ["comm"],
        "tags": ["Network","헤더","트레일러","캡슐화","역캡슐화","VPN"],
        "url": "/comm/comm8/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 9. 물리 계층의 역할과 랜 카드의 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 9-1. 전기 신호 네트워크를 통해 데이터를 주고받을 때는 0과 1의 비트열을 전기 신호로 변환해야 한다. 0과 1로 이루어진 비트열을 전기 신호로 변한하려면 OSI 모델의 맨 아래 계층인 물리 계층의 기술이 필요하다. 전기 신호 종류에는 아날로그...","categories": ["comm"],
        "tags": ["Network","아날로그 신호","디지털 신호","물리 계층","랜 카드"],
        "url": "/comm/comm9/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 10. 케이블의 종류와 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 10-1. 트위스트 페어 케이블 네트워크 전송 매체는 데이터가 흐르는 물리적인 경로이다. 크게 유선과 무선으로 나뉜다. 유선은 트위스트 페어 케이블, 광케이블 등이 있다. 무선은 라디오파, 마이크로파, 적외선 등이 있다. 가장 많이 사용되는 유선 전송 매체는 트위스트...","categories": ["comm"],
        "tags": ["Network","트위스트 페어 케이블","다이렉트 케이블","크로스 케이블"],
        "url": "/comm/comm10/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 11. 리피터와 허브의 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 11-1. 리피터 리피터(Repeater)는 일그러진 전기 신호를 복원(정형)하고 증폭하는 기능을 가진 네트워크 중계 장비이다. 통신하는 상대방이 멀리 있을 때 리피터를 사이에 넣는다! 그러나 요즘은 다른 네트워크 장비가 리피터 기능을 지원하므로 리피터를 쓸 필요가 없다. 11-2. 허브...","categories": ["comm"],
        "tags": ["Network","리피터","허브"],
        "url": "/comm/comm11/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 12. 데이터 링크 계층의 역할과 이더넷",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 12-1. 이더넷 랜에서 데이터를 주고 받으려면 데이터 링크 계층의 기술이 필요하다. 데이터 링크 계층은 네트워크 장비 간에 신호를 주고받는 규칙이다. 즉, 랜에서 데이터를 정상적으로 주고받기 위해 필요한 계층이다. 가장 많이 사용되는 규칙은 바로 이더넷(Ethernet)이다. 이더넷은...","categories": ["comm"],
        "tags": ["Network","이더넷","CSMA/CD"],
        "url": "/comm/comm12/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 13. MAC 주소의 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 13-1 MAC 주소 랜 카드는 비트열을 전기 신호로 변환한다고 했다. 랜 카드에는 MAC 주소(Media Access Address, 물리 주소) 라는 번호가 정해져 있다. 제조할 때 새겨지므로 물리 주소라고도 부른다. 전 세계에서 유일한 번호로 할당이 된다. MAC...","categories": ["comm"],
        "tags": ["Network","MAC 주소","프레임"],
        "url": "/comm/comm13/",
        "teaser": null
      },{
        "title": "[전기 기초] 3. 직류(DC)와 교류(AC)의 차이점",
        "excerpt":"본 포스팅은 “김기사의 e-쉬운 전기” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 3-1. 직류의 성질 직선으로 쭉 그어지며 전압과 전류가 흐르기에 직류라고 한다. (+)극과 (-)극으로 구분되고 전력이 일정한 크기로 한 방향으로 진행한다는 점이 특징이다. 직류 12V를 DC 12V, 12VDC로 표기한다. 직류는 시간에 따른 전력의 변화가 없고...","categories": ["elecbasic"],
        "tags": ["probaility"],
        "url": "/elecbasic/elecbasic3/",
        "teaser": null
      },{
        "title": "[전기 기초] 6. 전자제품에서 꼭 필요한 것",
        "excerpt":"본 포스팅은 “김기사의 e-쉬운 전기” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 6-1. 저항기, 인덕터, 커패시터의 역할 저항기는 전류의 흐름을 방해하면서 전자제품의 조절을 도와주는 장치이다. TV의 경우 화면밝기나 소리크기를 조절할 때 저항기가 자신의 임무를 수행한다. 전자제품을 사용하다 보면 제품 한 구석이 따뜻하게 느껴지는데, 전자제품 내부의 저항기가...","categories": ["elecbasic"],
        "tags": ["probaility"],
        "url": "/elecbasic/elecbasic6/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 14. 스위치의 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 14-1. MAC 주소 테이블 스위치는 데이터 링크 계층에서 동작하고 레이어 2 스위치 또는 스위칭 허브라고도 불린다. 외형은 허브와 비슷하다. 그러나 기능은 완전히 다르다. 스위치 내부에는 MAC 주소 테이블(MAC address table)이라는 것이 있다. MAC 주소 테이블은...","categories": ["comm"],
        "tags": ["Network","스위치","MAC 주소 필터링"],
        "url": "/comm/comm14/",
        "teaser": null
      },{
        "title": "[전기 기초] 9. 용량이 큰 냉난방기가 3상 전력을 사용하는 이유",
        "excerpt":"본 포스팅은 “김기사의 e-쉬운 전기” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 9-1. 단상 전력과 3상 전력의 차이 1. 단상 전력과 3상 전력 가정을 비롯해 많은 곳에서 교류 220V 전압을 이용한 전력을 사용한다. 이를 단상(Single-phase) 교류 220V라고 한다. 그림 9-1 (a)는 3개의 단상 교류를 사용하고 있다....","categories": ["elecbasic"],
        "tags": ["probaility"],
        "url": "/elecbasic/elecbasic9/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 15. 데이터가 케이블에서 충돌하지 않는 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 15-1. 전이중 통신과 반이중 통신 전이중 통신 방식 데이터의 송수신 동시에 처리 데이터 동시에 전송해도 충돌 발생X 반이중 통신 방식 회선 하나로 송신과 수신 번갈아가면서 통신 데이터 동시에 전송시 충돌 발생 만약 그림 15-1 처럼...","categories": ["comm"],
        "tags": ["Network","전이중 통신","반이중 통신","충돌 도메인","ARP"],
        "url": "/comm/comm15/",
        "teaser": null
      },{
        "title": "[자료구조와 알고리즘] 2. 반복하는 알고리즘",
        "excerpt":"본 포스팅은 “자료구조와 함께 배우는 알고리즘(파이썬)” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 2-1. range() 함수로 이터러블 객체 생성하기 range(n): 0 이상 n 미만인 수를 차례로 나열하는 수열 range(a, b): a 이상 b 미만인 수를 차례로 나열하는 수열 range(a, b, step): a 이상 b 미만인 수를...","categories": ["algopy"],
        "tags": ["Data structure","Algorithm"],
        "url": "/algopy/algopy2/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 16. 이더넷의 종류와 특징",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 16-1. 이더넷 규격 이더넷은 케이블 종류나 통신 속도에 따라 아래 그림과 같이 다양한 규격으로 분류된다. 그림 16-1. 주요 이더넷 규격 그림 16-1의 10BASE-T 를 보자. 10은 Mbps 단위인 통신 속도이다. BASE는 BASEBAND 라는 전송 방식을...","categories": ["comm"],
        "tags": ["Network","이더넷 규격"],
        "url": "/comm/comm16/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 17. 네트워크 계층의 역할",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 17-1. 네트워크 간의 연결 구조 그림 17-1. OSI 모델의 네트워크 계층 앞에서 데이터 링크 계층은 이더넷 규칙으로 데이터 전송을 담당한다고 했다. 이 규칙을 이용하면 네트워크에 있는 컴퓨터로는 데이터를 전송할 수 있다. 그러나 인터넷 또는 다른...","categories": ["comm"],
        "tags": ["Network","이더넷 규격"],
        "url": "/comm/comm17/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 18. IP 주소의 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 18-1. IP 주소 IP 주소는 우편물 주소와 같은 개념이다. 데이터를 다른 네트워크의 목적지로 보낼 때 IP 주소가 필요하다. IP 주소는 인터넷 서비스 제공자(ISP)에게 받을 수 있다. IP 버전에는 IPv4와 IPv6가 있다. IPv4는 32비트로 되어 있어서...","categories": ["comm"],
        "tags": ["Network","이더넷 규격"],
        "url": "/comm/comm18/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 19. IP 주소의 클래스 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 19-1. IP 주소 클래스란? IPv4의 IP 주소는 32 비트이다. 비트로 구분되지만 네트워크 ID를 크게 만들거나, 호스트 ID를 작게 만들어 네트워크 크기를 조정할 수 있다. 네트워크 크기는 클래스라는 개념으로 구분한다. 그림 19-1. 클래스 종류 일반 네트워크에서는...","categories": ["comm"],
        "tags": ["Network","이더넷 규격"],
        "url": "/comm/comm19/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 20. 네트워크 주소와 브로드캐스트 주소의 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 20-1. 네트워크 주소와 브로드캐스트 주소란? IP 주소에는 네트워크 주소와 브로드캐스트 주소가 있다. 이 두 주소는 특별한 주소로 컴퓨터나 라우터가 자신의 IP로 사용하면 안되는 주소이다. 네트워크 주소는 호스트 ID가 10진수로 0, 2진수로는 00000000인 주소이다. 브로드캐스트 주소는...","categories": ["comm"],
        "tags": ["Network","이더넷 규격"],
        "url": "/comm/comm20/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 21. 서브넷의 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 21-1. 서브넷 앞에서 A 클래스는 네트워크 ID가 8비트에 호스트 ID가 24비트, B 클래스는 네트워크 ID가 16비트에 호스트 ID가 16비트, C 클래스는 네트워크 ID가 24비트에 호스트 ID가 8비트라고 했다. A 클래스는 호스트 ID가 24 비트라 IP...","categories": ["comm"],
        "tags": ["Network","이더넷 규격"],
        "url": "/comm/comm21/",
        "teaser": null
      },{
        "title": "[네트워크 초급] 22. 라우터의 구조",
        "excerpt":"본 포스팅은 “모두의 네트워크” 책 내용을 기반으로 작성되었습니다. 잘못된 내용이 있을 경우 지적해 주시면 감사드리겠습니다. 22-1. 라우터 서로 다른 네트워크와 통신하려면 라우터가 필요하다. 그림 22-1. 네트워크를 분리하는 라우터 그림 22-1 처럼 네트워크를 분리할 수 있다. 스위치만 있는 네트워크에서는 모든 컴퓨터와 스위치가 그림 22-2 처럼 동일한 네트워크에 속하게 된다. 그림 22-2....","categories": ["comm"],
        "tags": ["Network","이더넷 규격"],
        "url": "/comm/comm22/",
        "teaser": null
      }]
