from sklearn.linear_model import LinearRegression
import numpy as np

# 선형 회귀 — 직선으로 숫자 예측하기

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

model = LinearRegression()
model.fit(x, y)

prediction = model.predict([[6], [7], [8]])

print (prediction)
print ("기울기 : ", model.coef_)
print ("절편 : ", model.intercept_)

