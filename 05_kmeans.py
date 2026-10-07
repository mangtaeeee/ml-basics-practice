# K-Means 군집화 모델 가져오기
from sklearn.cluster import KMeans

# 숫자 배열 처리를 위한 Numpy
import numpy as np


# Feature 데이터
# 각 사람의 [키, 몸무게]
x = np.array([
    [160, 50],
    [165, 55],
    [170, 65],
    [175, 70],
    [180, 80],
    [185, 85]
])

# K-Means 모델 생성
# n_clusters = 2 -> 데이터를 2개의 그룹으로 나누겠다는 뜻
model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init= 10
)

# 학습
# K-Means 는 Label(y)이 없음
# 입력 데이터 x만 가지고 비슷한 것끼리 그룹을 찾음
model.fit(x)

# 각 데이터가 몇번 그룹에 속하는지 확인
labels = model.labels_
print("그룹 번호 : ", labels)

# 각 그룹의 중심점 확인
centers = model.cluster_centers_

print("그룹 중심점")
print(centers)