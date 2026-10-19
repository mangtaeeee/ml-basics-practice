# ML Basics Practice

4년차 Spring Backend 개발자가 Machine Learning의 기초 개념을 직접 실습하며 정리하는 저장소입니다.

단순히 코드를 실행하는 데서 끝내지 않고, 각 개념과 알고리즘이 어떤 의미인지 이해하고 나중에 다시 봐도 복습할 수 있도록 정리하는 것을 목표로 합니다.

---

# 1. Machine Learning 기본 개념

## Machine Learning이란?

Machine Learning은 사람이 모든 규칙을 직접 작성하는 대신, 데이터를 이용해 컴퓨터가 규칙이나 패턴을 학습하도록 하는 방법입니다.

일반적인 프로그램은 사람이 규칙을 직접 작성합니다.

```text
입력
 ↓
사람이 만든 규칙
 ↓
결과
```

Machine Learning은 데이터를 통해 모델이 규칙을 찾습니다.

```text
학습 데이터
 ↓
모델 학습
 ↓
패턴 / 규칙 발견
 ↓
새로운 데이터 예측
```

핵심은 다음과 같습니다.

> 사람이 모든 조건문을 직접 만드는 것이 아니라, 데이터에서 규칙을 찾게 한다.

---

# 2. Feature / Label / Prediction

## Feature

Feature는 모델이 판단할 때 사용하는 입력값 또는 속성입니다.

예:

```text
공부시간 → 시험점수
```

여기서 공부시간이 Feature입니다.

Feature가 여러 개일 수도 있습니다.

```text
공부시간
수면시간
출석률
과제점수
   ↓
시험점수
```

이 경우 공부시간, 수면시간, 출석률, 과제점수가 모두 Feature입니다.

## Label

Label은 모델이 학습할 때 사용하는 실제 정답값입니다.

```text
Feature = 공부시간
Label   = 시험점수
```

분류 문제에서는 Label이 다음처럼 표현될 수도 있습니다.

```text
0 = 불합격
1 = 합격
```

## Prediction

Prediction은 학습이 끝난 모델이 새로운 Feature를 보고 예측한 결과입니다.

```text
Label      = 실제 정답
Prediction = 모델이 예상한 값
```

---

# 3. 기대값(Expected Value)

기대값은 여러 결과가 발생할 수 있을 때 각각의 확률을 고려해 계산한 평균적인 예상값입니다.

예:

```text
50% 확률로 0원
50% 확률로 100원

기대값 = 0 × 0.5 + 100 × 0.5 = 50
```

Label과 기대값은 다른 개념입니다.

---

# 4. 선형과 비선형

## 선형(Linear)

입력값이 변할 때 결과값이 일정한 비율로 변하는 관계입니다.

```text
x = 1 → y = 10
x = 2 → y = 20
x = 3 → y = 30
x = 4 → y = 40
```

대표적인 식:

```text
y = ax + b
```

```text
x = 입력값
y = 결과값
a = 기울기
b = 절편
```

### 기울기

x가 1만큼 변할 때 y가 얼마나 변하는지를 의미합니다.

### 절편

x가 0일 때의 y값입니다.

## 비선형(Non-linear)

입력값과 결과값의 관계를 직선 하나로 표현하기 어려운 관계입니다.

```text
x = 1 → y = 1
x = 2 → y = 4
x = 3 → y = 9
x = 4 → y = 16
x = 5 → y = 25
```

```text
y = x²
```

핵심:

```text
선형   = 직선으로 설명 가능
비선형 = 직선 하나로 설명하기 어려움
```

---

# 5. Linear Regression

Linear Regression은 숫자를 예측하는 대표적인 선형 회귀 모델입니다.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([50, 60, 70, 80, 90])

model = LinearRegression()
model.fit(x, y)

prediction = model.predict([[6]])

print(prediction)
print("기울기 :", model.coef_)
print("절편 :", model.intercept_)
```

이 데이터에서는 대략:

```text
y = 10x + 40
```

이라는 관계를 찾습니다.

---

# 6. 비선형 데이터를 Linear Regression에 넣으면?

```python
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])
```

실제 관계는:

```text
y = x²
```

이지만 Linear Regression은 직선밖에 만들 수 없습니다.

실습 결과:

```text
기울기 = 6
절편 = -7
```

즉 모델은:

```text
y = 6x - 7
```

이라는 직선을 찾았습니다.

핵심:

> 데이터의 관계에 맞는 모델을 선택하는 것이 중요하다.

---

# 7. Regression과 Classification

## Regression

숫자를 예측합니다.

```text
집값 예측
매출 예측
시험점수 예측
온도 예측
```

질문으로 표현하면:

```text
얼마인가?
```

## Classification

종류 또는 클래스를 예측합니다.

```text
합격 / 불합격
정상 / 비정상
스팸 / 정상메일
고양이 / 강아지
```

질문으로 표현하면:

```text
어느 쪽인가?
```

---

# 8. Logistic Regression

이름에는 Regression이 들어가지만 주로 Classification에 사용하는 모델입니다.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(x, y)

prediction = model.predict([[2.5], [3.5], [5]])
probability = model.predict_proba([[3.5]])

print("예측 결과 :", prediction)
print("확률 :", probability)
```

실습 결과:

```text
예측 결과 : [0 1 1]

확률 :
[[0.49996763 0.50003237]]
```

핵심:

```text
Linear Regression
→ 숫자를 예측

Logistic Regression
→ 확률을 계산하고 클래스를 분류
```

---

# 9. fit / predict / predict_proba

```python
model.fit(x, y)
```

Feature와 Label의 관계를 학습합니다.

```python
model.predict(...)
```

학습된 모델에 새로운 데이터를 넣어 결과를 예측합니다.

```python
model.predict_proba(...)
```

분류 모델에서 각 클래스에 속할 확률을 확인합니다.

---

# 10. reshape(-1, 1)

scikit-learn의 모델은 Feature 데이터를 보통 다음 형태로 받습니다.

```text
[샘플 개수, Feature 개수]
```

```python
np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
```

은:

```text
[
 [1],
 [2],
 [3],
 [4],
 [5]
]
```

형태로 바뀝니다.

---

# 11. Decision Tree

Decision Tree는 의사결정나무입니다.

질문을 하나씩 던지고 조건에 따라 갈라져서 최종 결과를 판단합니다.

```text
공부시간이 3.5시간 이하인가?
        ├─ YES → 불합격
        └─ NO  → 합격
```

Java의 if문과 비슷하게 이해할 수 있습니다.

```java
if (studyHours <= 3.5) {
    return 0;
} else {
    return 1;
}
```

차이는 사람이 조건을 직접 만드는 것이 아니라 모델이 데이터를 보고 조건을 찾는다는 점입니다.

```python
from sklearn.tree import DecisionTreeClassifier
```

---

# 12. Random Forest

Random Forest는 여러 개의 Decision Tree를 만들고 그 결과를 종합하는 모델입니다.

```text
Tree 1 → 합격
Tree 2 → 합격
Tree 3 → 불합격
Tree 4 → 합격
Tree 5 → 합격

최종 → 합격
```

```python
from sklearn.ensemble import RandomForestClassifier
```

예:

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

`n_estimators=100`은 Decision Tree를 100개 만든다는 뜻입니다.

`random_state=42`는 랜덤 과정을 고정해 같은 실험 결과를 재현하기 위한 값입니다.

실습 결과:

```text
예측 결과 :
[0 0 1]

예측 확률 :
[[1.00 0.00]
 [0.77 0.23]
 [0.02 0.98]]
```

핵심:

```text
Decision Tree
→ Tree 1개

Random Forest
→ Tree 여러 개
→ 여러 Tree의 결과를 종합
```

---

# 13. K-Means

K-Means는 비슷한 데이터끼리 자동으로 그룹으로 묶는 알고리즘입니다.

중요한 점은 Label이 필요하지 않다는 것입니다.

```python
from sklearn.cluster import KMeans
```

예:

```python
model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)
```

핵심:

```text
Label 없음
↓
데이터끼리 얼마나 비슷한지 확인
↓
비슷한 것끼리 그룹화
```

즉 비지도학습입니다.

---

# 14. Isolation Forest

Isolation Forest는 다른 데이터와 동떨어진 이상치를 찾는 알고리즘입니다.

예:

```text
9
10
10
11
11
12
100
```

대부분 9~12 사이에 있지만 100만 멀리 떨어져 있습니다.

```python
from sklearn.ensemble import IsolationForest
```

예:

```python
model = IsolationForest(
    contamination=0.15,
    random_state=42
)
```

`contamination=0.15`는 전체 데이터 중 이상치가 약 15% 정도 있을 것이라고 가정한다는 의미입니다.

모델이 각 데이터의 이상도를 계산하고 contamination 값으로 정상/이상의 기준을 잡습니다.

```text
1  = 정상
-1 = 이상치
```

핵심:

```text
K-Means
→ 비슷한 것끼리 묶기

Isolation Forest
→ 다른 것들과 동떨어진 값 찾기
```

---

# 15. 지도학습과 비지도학습

## 지도학습(Supervised Learning)

정답(Label)이 있는 데이터를 이용해 학습합니다.

```text
Linear Regression
Logistic Regression
Decision Tree
Random Forest
```

## 비지도학습(Unsupervised Learning)

정답(Label)이 없는 데이터에서 패턴을 찾습니다.

```text
K-Means
Isolation Forest
```

---

# 16. Train / Test Split

실제 Machine Learning에서는 학습용 데이터와 테스트용 데이터를 분리합니다.

이유:

> 이미 본 문제로 시험을 보면 모델의 실제 실력을 알기 어렵기 때문입니다.

```text
Train = 공부할 문제집
Test  = 처음 보는 시험 문제
```

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

`test_size=0.2`는 전체 데이터의 20%를 테스트용으로 사용한다는 뜻입니다.

실습에서는:

```text
실제 정답 : [1 0]
모델 예측 : [1 0]
```

으로 테스트 데이터 2개를 모두 맞혔습니다.

핵심:

```text
fit()
→ Train 데이터로 공부

predict()
→ Test 데이터로 시험

y_test
→ 실제 정답

prediction
→ 모델이 낸 답
```

---

# 17. Accuracy

Accuracy는 전체 문제 중 몇 개를 맞혔는지를 나타내는 정확도입니다.

```text
Accuracy = 맞힌 개수 / 전체 개수
```

예:

```text
2문제 중 2문제 정답

2 / 2 = 1.0
```

즉 100%입니다.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, prediction)
print("정확도 :", accuracy)
```

단, 테스트 데이터가 너무 적으면 Accuracy가 100%라도 좋은 모델이라고 바로 판단할 수 없습니다.

---

# 18. Confusion Matrix

Confusion Matrix는 모델이 어디서 맞았고 어디서 틀렸는지를 4가지 경우로 나눠서 보는 표입니다.

예:

```text
1 = 스팸
0 = 정상메일
```

```text
실제 스팸인데 스팸이라고 맞춤
→ TP (True Positive)

실제 스팸인데 정상이라고 틀림
→ FN (False Negative)

실제 정상인데 스팸이라고 틀림
→ FP (False Positive)

실제 정상인데 정상이라고 맞춤
→ TN (True Negative)
```

쉽게 기억하면:

```text
True  = 맞음
False = 틀림

Positive = 1
Negative = 0
```

따라서:

```text
TP = 1을 1이라고 맞춤
TN = 0을 0이라고 맞춤
FP = 0인데 1이라고 잘못 판단
FN = 1인데 0이라고 잘못 판단
```

코드:

```python
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, prediction)
print(matrix)
```

예:

```text
[[55  5]
 [10 30]]
```

scikit-learn에서는:

```text
[[TN FP]
 [FN TP]]
```

순서로 표시됩니다.

Confusion Matrix를 이해하면 이후 배우는 Precision, Recall, F1 Score를 이해하기 쉬워집니다.

---

# 19. scikit-learn 구조

`sklearn`은 Machine Learning에서 많이 사용하는 Python 라이브러리입니다.

```text
sklearn
├─ linear_model
├─ tree
├─ ensemble
├─ svm
├─ cluster
├─ neighbors
├─ preprocessing
├─ model_selection
├─ metrics
└─ ...
```

지금까지 사용한 모듈:

```text
sklearn.linear_model
→ LinearRegression
→ LogisticRegression

sklearn.tree
→ DecisionTreeClassifier

sklearn.ensemble
→ RandomForestClassifier
→ IsolationForest

sklearn.cluster
→ KMeans

sklearn.model_selection
→ train_test_split

sklearn.metrics
→ accuracy_score
→ confusion_matrix
```

---

# 20. 지금까지 배운 모델 정리

| 모델 | 문제 유형 | Label 필요 | 핵심 |
|---|---|---:|---|
| Linear Regression | 회귀 | 필요 | 직선으로 숫자 예측 |
| Logistic Regression | 분류 | 필요 | 확률 기반 분류 |
| Decision Tree | 분류/회귀 | 필요 | 조건을 따라 판단 |
| Random Forest | 분류/회귀 | 필요 | 여러 Tree의 결과 종합 |
| K-Means | 군집화 | 불필요 | 비슷한 것끼리 그룹화 |
| Isolation Forest | 이상탐지 | 불필요 | 동떨어진 데이터 탐지 |

---

# 21. 기본 Machine Learning 흐름

```text
데이터 준비
   ↓
Feature / Label 확인
   ↓
문제 유형 판단
   ↓
Train / Test 분리
   ↓
모델 선택
   ↓
model.fit()
   ↓
학습
   ↓
model.predict()
   ↓
예측
   ↓
실제 Label과 비교
   ↓
Accuracy / Confusion Matrix 등으로 평가
```

---

# 22. 현재 진행 상황

- [x] AI / ML / DL 기본 관계
- [x] Feature
- [x] Label
- [x] Prediction
- [x] 기대값 기본 개념
- [x] 선형 / 비선형
- [x] 1차 함수 기본 개념
- [x] 기울기
- [x] 절편
- [x] Linear Regression
- [x] 비선형 데이터에 Linear Regression 적용
- [x] Regression / Classification 차이
- [x] Logistic Regression
- [x] `fit()`
- [x] `predict()`
- [x] `predict_proba()`
- [x] Decision Tree
- [x] Random Forest
- [x] K-Means
- [x] Isolation Forest
- [x] Train / Test Split
- [x] Accuracy 기본 개념
- [x] Confusion Matrix 기본 개념
- [ ] Precision
- [ ] Recall
- [ ] F1 Score
- [ ] Overfitting / Underfitting
- [ ] Feature Scaling
- [ ] StandardScaler
- [ ] pandas 기반 데이터 처리
- [ ] 실제 Dataset 기반 실습
- [ ] XGBoost
- [ ] LightGBM
- [ ] SHAP

---

# 23. Tech Stack

현재:

```text
Python
NumPy
scikit-learn
PyCharm
```

추후:

```text
pandas
matplotlib
XGBoost
LightGBM
SHAP
FastAPI
LLM
RAG
AI Agent
```

---

# 24. Environment

가상환경 생성:

```powershell
py -m venv .venv
```

PowerShell 활성화:

```powershell
.\.venv\Scripts\Activate.ps1
```

라이브러리 설치:

```powershell
python -m pip install numpy pandas scikit-learn
```

---

# 25. Project Goal

이 저장소의 목표는 Machine Learning 알고리즘을 단순히 사용하는 것이 아니라 다음 질문에 답할 수 있는 수준까지 이해하는 것입니다.

```text
이 문제는 회귀인가?
분류인가?
군집화인가?
이상탐지인가?

Label이 있는가?

어떤 Feature를 사용할 것인가?

Train / Test는 어떻게 나눌 것인가?

모델이 얼마나 잘 맞혔는가?

어디에서 틀렸는가?

왜 이 모델을 선택했는가?

다른 모델보다 어떤 장단점이 있는가?
```

장기적으로는 기존 Backend 개발 경험과 Machine Learning / LLM 기술을 연결해 실제 시스템에 AI를 적용할 수 있는 개발 역량을 만드는 것을 목표로 합니다.
