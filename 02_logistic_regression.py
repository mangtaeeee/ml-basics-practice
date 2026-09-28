
import numpy as np
from sklearn.linear_model import LogisticRegression

# 공부시간
# 이 값들이 모델이 판단에 사용하는 Feature(입력값)
x = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# 결과
# 0 = 불합격
# 1 = 합격
y = np.array([0, 0, 0, 1, 1, 1])

# Logistic Regression 모델 객체 생성
# 이 모델은 숫자를 직접 예측하는 게 아니라
# 0 / 1 같은 클래스를 분류하는 데 사용
model = LogisticRegression()

# 모델 학습
# x(공부시간) 와 y(합격/불합격)의 관계를 학습함
model.fit( x, y )

# 새로운 공부시간을 넣어서
# 합격(1)인지 불합격(0)인지 예측
prediction = model.predict([[2.5], [3.5], [5]])

print("예측 결과 : ", prediction)

# predict_proba()는 각 클래스일 확률을 보여줌
# 예를 들어 [[0.6, 0.4]]가 나오면
# 0(불합격)일 확률 60%
# 1(합격)일 확률 40%
probability = model.predict_proba([[3.5]])

# 확률 출력
print("확률 :", probability)