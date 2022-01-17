var store = [{
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
        "tags": ["python"],
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
        "tags": ["python"],
        "url": "/pythonmd/pythonmd17/",
        "teaser": null
      }]
