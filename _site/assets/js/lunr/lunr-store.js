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
      }]
