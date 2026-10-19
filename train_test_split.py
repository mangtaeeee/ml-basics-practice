from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


import numpy as np


# Feature
# 공부시간
x = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

# Label
# 0 = 불합격
# 1 = 합격
y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1
])


# 전체 데이터를 학습용 / 테스트용으로 나눔
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# Random Forest 모델 생성
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 학습용 데이터로만 학습
model.fit(x_train, y_train)


# 모델이 한 번도 보지 못한 테스트 데이터로 예측
prediction = model.predict(x_test)

accuracy = accuracy_score(y_test, prediction)

print("정확도 :", accuracy)


print("학습용 Feature :", x_train)
print("테스트용 Feature :", x_test)

print("실제 정답 :", y_test)
print("모델 예측 :", prediction)