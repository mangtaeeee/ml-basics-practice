
# 숫자 배열 처리를 위한 Numpy
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


# Feature(입력값)
# 공부시간
x = np.array([1,2,3,4,5,6]).reshape(-1,1)

# Label(정답)
# 0 = 불합격
# 1 = 합격
y = np.array([0,0,0,1,1,1])

# Decision Tree 분류 모델
model = DecisionTreeClassifier()

# Feature와 Label 을 이용해 학습
model.fit(x,y)

prediction = model.predict([
    [2.5],
    [3.5],
    [5]
])

print("예측 결과 :", prediction)

tree_rule = export_text(
    model,
    feature_names=["공부시간"]
)

print(tree_rule)