# Random Forest 분류 모델 가져오기
from sklearn.ensemble import RandomForestClassifier

# 숫자 배열 처리를 위한 Numpy
import numpy as np

# Feature(입력값)
# 공부시간
x = np.array([1,2,3,4,5,6]).reshape(-1,1)

# Label(정답)
# 0 = 불합격
# 1 = 합격
y = np.array([0,0,0,1,1,1])

# Random Forest 모델 생성
# n_estimators=100 -> Decision Tree를 100개 만든다는 뜻
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 모델 학습
model.fit(x,y)

#새로운 공부시간으로 합격 / 불합격 예측
prediction = model.predict([
    [2.5],
    [3.5],
    [5]
])

print("예측 결과 : ", prediction)


# 각각 불합격 / 합격일 확률 확인
probability = model.predict_proba([
    [2.5],
    [3.5],
    [5]
])

print("에측 확률 : ")
print(probability)