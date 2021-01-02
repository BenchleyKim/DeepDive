import torch
import numpy as np

# 2차원 형태의 리스트를 활용하여 텐서를 생성할 수 있다.
torch.tensor([[1,2],[3,4]])

# GPU에 텐서를 만들 수 있다.
torch.tensor([[1,2],[3,4.]], device="cuda:0")

# dtype을 이용하여 텐서의 데이터 형태를 지정할 수도 있다.
torch.tensor([[1,2],[3,4]],dtype = torch.float64)

#arange를 이용하여 1차원 텐서 생성
torch.arange(0,10)

# 모든 값이 0인 3x5 텐서를 작성하여 to 메소드를 사용하여 gpu로 전송
torch.zeros(3,5).to("cuda:0")

# 정규분포로 3x5 텐서를 작성
torch.randn(3,5)

# 텐서의 shape는 size 메서드로 확인
t = torch.randn(3,5)
t.size()

# numpy를 사용하여 ndarray로 변환
t = torch.tensor([[1,2],[3,4]])
x = t.numpy()

# GPU 상의 텐서는 to 메서드로  CPU 텐서로 변환 후 ndarray로 변환해야함
t = torch.tensor([[1,2],[3,4]], device= "cuda:0")
x = t.to("cpu").numpy()

x = torch.linspace(0,10,5)
y = torch.exp(x)
# plt.plot(x.numpy(),y.numpy())


##
## 텐서의 인덱싱
##

t = torch.tensor([[1,2,3],[4,5,6.]])

# 인덱스 접근
t[0, 2]
# : tensro(3.)

# 슬라이드로 접근
t[:, :2]
# : tensor([[1., 2.],[4.,5.]])

# 마스크 배열을 이용하여 True 값만 추출
t[t > 3]
# : temsor([4., 5., 6.])

# 슬라이스를 이용하여 일괄 대입
t[:,1] = 10

# 마스크 배열을 이용하여 일괄 대입
t[t > 5 ] = 10 

##
## 텐서의 연산
##

# 길이 3인 벡터
v = torch.tensor([1,2,3])
w = torch.tensor([0,10,20])

# 2 x 3의 행렬
m = torch.tensor([[0,1,2],[100,200,300.]])

# 벡터와 스칼라의 덧셈
v2 = v+10

# 제곱
v2 = v**2

# 동일 길이의 벡터간 덧셈 연산
z = v - w

# 여러가지 조합
u = 2 * v - w /10 + 6.0

# 행렬과 스칼라 곱
m2 = m * 2.0








