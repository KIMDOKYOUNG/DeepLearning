## 밑바닥부터 시작하는 딥러닝 1 뽀개기

#### CHAPTER 5 오차역전파법 

#### 일자 : 2020-08-16



- 수치미분은 구현이 쉽지만 계산 시간이 오래걸린다. 

- 가중치 매개변수의 기울기를 효율적으로 계산하는 오차역전파법!



---



## 덧셈, 곱셈의 역전파 => 계산 그래프를 사용해서 계산 과정을 보면 쉽다. 



- 덧셈의 역전파에서는 상류의 값을 그대로 흘려보내서 순방향 입력 신호의 값이 필요하지 않다.
- 곱셈의 역전파는 상류의 값에 순방향 입력 신호를 서로 바꿔서 곱한 값을 흘려보낸다. 

## 계산 그래프 

- 가격이 100원인 사과를 2개 사고 소비세가 10% 부과될 때 지불 금액은?

![사진1.jpg](https://github.com/KIMDOKYOUNG/DeepLearning/blob/master/chapter5/img/사진1.jpg)

- 사과 2개, 귤 3개를 구매, 사과는 1개에 100원, 귤을 1개에 150원이고 소비세 10%가 붙을 때 지불 금액을 계산 그래프로 나타내면!

![사진2.jpg](https://github.com/KIMDOKYOUNG/DeepLearning/blob/master/chapter5/img/사진2.jpg)

## 국소적 계산

- 계산 그래프의 특징은 자신과 직접 관계된 작은 범위(국소적)의 계산을 전파함으로써 최종 결과를 얻는다. 



## 계산 그래프를 사용하는 이유??

- 역전파를 통해 '미분'(기울기)를 효율적으로 계산할 수 있다! 

- 역전파는 국소적인 미분을 순방향과는 반대인 오른쪽에서 왼쪽으로 전달한다. 

- 연쇄법칙(Chain rule)을 따른다



## 계산 그래프의 역전파 

![사진3.jpg](https://github.com/KIMDOKYOUNG/DeepLearning/blob/master/chapter5/img/사진3.jpg)

- 노드로 들어온 입력 신호에 그 노드의 국소적 미분(편미분)을 곱한 다음 다음 노드로 전달한다. 

- 위의 사진에서 볼 수 있듯이 연쇄법칙을 통해서 역전파를 나타낼 수 있다.

## 덧셈 노드의 역전파 & 곱셈 노드의 역전파

![사진4-1.jpg](https://github.com/KIMDOKYOUNG/DeepLearning/blob/master/chapter5/img/사진4-1.jpg)

![사진4-2.jpg](https://github.com/KIMDOKYOUNG/DeepLearning/blob/master/chapter5/img/사진4-2.jpg)

- 덧셈 노드의 역전파
  - 덧셈 노드의 역전파는 z = x+y의 편미분을 했을 때 둘다 1이 나와서 상류의 값에 1을 곱하는 것이므로 입력값(상류값)을 그대로 흘려보내는 것과 같다

- 곱셈 노드의 역전파
  - 곱셈 노드의 역전파는 z = xy의 편미분을 했을 때 x에 대한 z의 미분은 y가 나오고, y에 대한 z의 미분은 x가 나오므로 서로 바꾼 값이 나온 것을 알 수 있다. 
  - 그래서 상류값에 입력 신호들을 서로 바꾼 값을 곱해서 하류로 흘려 보낸다. 

 

위에서 다뤘던 사과를 사는 문제와 사과와 귤을 사는 문제에 대한 역전파를 계산 해보면 아래와 같다! 

![사진5.jpg](https://github.com/KIMDOKYOUNG/DeepLearning/blob/master/chapter5/img/사진5.jpg)

- 소비세의 미분은 200, 사과 가격의 미분은 2.2이다. 

- 소비세와 사과 가격이 같은 양만큼 오르면 최종금액에는 소비세가 200의 크기로, 사과 가격이 2.2의 크기로 영향을 준다고 해석 할 수 있다. 

## 활성화 함수 계층 구현하기 



### 1. ReLU 계층 

- ReLU 순전파
  - 입력인 x가 0보다 크면 x가 그대로 전달되고, 0보다 작으면 전달 되지 않는다. (0이 전달 되는 것과 같다) 

- ReLU 역전파 
  - 순전파 때의 입력인 x가 0보다 크면 상류의 값을 그대로 하류로 흘린다. 
  - 순전파 때 x가 0 이하면 역전파 때는 하류로 신호를 보내지 않는다. 

![사진6.jpg](https://github.com/KIMDOKYOUNG/DeepLearning/blob/master/chapter5/img/사진6.jpg)

## Sigmoid 계층 



- 순전파와 역전파를 직접 계산 그래프를 그리면 더 이해하기 쉽다. 

- 총 4번의 과정을 거쳐서 나온다. 

![사진7.jpg](https://github.com/KIMDOKYOUNG/DeepLearning/blob/master/chapter5/img/사진7.jpg)

- sigmoid 계산을 할때 중간 과정을 하나씩 다 안 계산해도 한번에 순전파의 출력만을 가지고 계산할 수 있다. 

![사진8.jpg](https://github.com/KIMDOKYOUNG/DeepLearning/blob/master/chapter5/img/사진8.jpg)

## Affine/Softmax 계층 구현하기

### Affine 계층 



- 신경망의 순전파 때 수행하는 행렬의 내적을 affine transformation이라고 한다. 

![사진9.jpg](https://github.com/KIMDOKYOUNG/DeepLearning/blob/master/chapter5/img/사진9.jpg)

### 배치용 Affine 계층 

- 순전파의 편향 덧셈은 각각의 원소에 더해진다.

- 역전파의 경우 각 데이터의 역전파 값이 평향의 원소에 모여야된다. 열방향 (axis = 0)에 대해서 총합을 구한다.

![사진10.jpg](https://github.com/KIMDOKYOUNG/DeepLearning/blob/master/chapter5/img/사진10.jpg)