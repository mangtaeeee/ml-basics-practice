# Isolation Forest 모델 가져오기
from sklearn.ensemble import IsolationForest

import numpy as np

x = np.array([
    [10],
    [11],
    [9],
    [12],
    [10],
    [11],
    [100]
])

# 모델 생성
# 전체 데이터 중 약 15%는 이상치일 거라고 가정
model = IsolationForest(
    contamination=0.15,
    random_state=42
)

# 학습
model.fit(x)

# 각 데이터 가 정상인지 이상치인지 예측
prediction = model.predict(x)

print("예측 결과 :", prediction)
