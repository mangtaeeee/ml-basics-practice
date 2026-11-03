import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. CSV 파일 읽기
# --------------------------------------------------
# pandas = 판다스 / 데이터 처리 라이브러리
# read_csv() = CSV 파일을 읽는 함수
# DataFrame = 엑셀 표처럼 행/열로 데이터를 다루는 구조

df = pd.read_csv("login_logs.csv")


# --------------------------------------------------
# 2. 데이터 일부 확인
# --------------------------------------------------
# head() = 위에서부터 기본 5개 행 확인

print("===== 데이터 상위 5개 =====")
print(df.head())
print()


# --------------------------------------------------
# 3. 컬럼 확인
# --------------------------------------------------

print("===== 컬럼 목록 =====")
print(df.columns)
print()


# --------------------------------------------------
# 4. 데이터 크기 확인
# --------------------------------------------------
# shape = (행 개수, 열 개수)

print("===== 데이터 크기 =====")
print(df.shape)
print()


# --------------------------------------------------
# 5. Label = 정답값 분포 확인
# --------------------------------------------------
# 0 = 정상 로그인
# 1 = 이상 로그인
# value_counts() = 각 값이 몇 개인지 세기

print("===== Label(정답값) 분포 =====")
print(df["label"].value_counts())
print()


# --------------------------------------------------
# 6. Feature = 특성 / 입력값 선택
# --------------------------------------------------
# 모델이 판단할 때 사용할 정보
#
# login_hour       = 로그인 시간
# failed_count     = 로그인 실패 횟수
# country_changed  = 평소와 다른 국가 여부
# new_device       = 새로운 기기 여부
# ip_change_count  = 최근 IP 변경 횟수
# request_count    = 짧은 시간 동안 요청 횟수

feature_columns = [
    "login_hour",
    "failed_count",
    "country_changed",
    "new_device",
    "ip_change_count",
    "request_count"
]

X = df[feature_columns]


# --------------------------------------------------
# 7. Label = 정답값 선택
# --------------------------------------------------

y = df["label"]


# --------------------------------------------------
# 8. Feature / Label 확인
# --------------------------------------------------

print("===== Feature(특성) 상위 5개 =====")
print(X.head())
print()

print("===== Label(정답값) 상위 5개 =====")
print(y.head())
print()


# --------------------------------------------------
# 9. Train / Test Split
# 학습용 / 테스트용 데이터 분리
# --------------------------------------------------
# test_size=0.3 = 전체의 30%를 테스트용으로 사용
# random_state=42 = 랜덤 결과 고정
# stratify=y = 정상/이상 비율이 너무 다르게 나뉘지 않게 함

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("===== Train / Test 크기 =====")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print()


# --------------------------------------------------
# 10. Random Forest = 랜덤 포레스트 모델 생성
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# 11. 모델 학습
# --------------------------------------------------
# Train 데이터만 사용해서 학습

model.fit(X_train, y_train)


# --------------------------------------------------
# 12. 예측
# --------------------------------------------------
# 모델이 학습할 때 보지 못한 X_test를 예측
# 0 = 정상
# 1 = 이상

prediction = model.predict(X_test)


# --------------------------------------------------
# 13. Confusion Matrix = 혼동 행렬
# --------------------------------------------------
# sklearn 기준
#
# [[TN FP]
#  [FN TP]]
#
# TN = 실제 정상 / 예측 정상
# FP = 실제 정상 / 예측 이상
# FN = 실제 이상 / 예측 정상
# TP = 실제 이상 / 예측 이상

matrix = confusion_matrix(y_test, prediction)

print("===== Confusion Matrix(혼동 행렬) =====")
print(matrix)
print()


# --------------------------------------------------
# 14. 평가 지표 계산
# --------------------------------------------------
# Accuracy = 정확도
# 전체 중 몇 개를 맞혔는가?

accuracy = accuracy_score(y_test, prediction)

# Precision = 정밀도
# 모델이 이상(1)이라고 한 것 중 실제 이상인 비율
precision = precision_score(y_test, prediction)

# Recall = 재현율
# 실제 이상 중 모델이 이상이라고 찾아낸 비율
recall = recall_score(y_test, prediction)

# F1 Score = F1 점수
# Precision(정밀도)과 Recall(재현율)의 균형 점수
f1 = f1_score(y_test, prediction)


print("===== 평가 결과 =====")
print("Accuracy(정확도) :", round(accuracy, 4))
print("Precision(정밀도) :", round(precision, 4))
print("Recall(재현율) :", round(recall, 4))
print("F1 Score(F1 점수) :", round(f1, 4))
print()


# --------------------------------------------------
# 15. 실제값과 예측값 비교
# --------------------------------------------------

result = X_test.copy()
result["actual"] = y_test
result["prediction"] = prediction

print("===== 실제값 / 예측값 비교 =====") 
print(result)
print()


# --------------------------------------------------
# 16. 틀린 데이터만 확인
# --------------------------------------------------
# actual != prediction인 행만 추출

wrong = result[result["actual"] != result["prediction"]]

print("===== 틀린 데이터 =====")

if len(wrong) == 0:
    print("틀린 데이터가 없습니다.")
else:
    print(wrong)
