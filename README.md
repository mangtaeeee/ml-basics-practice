# ML Basics Practice

단순히 코드를 실행하는 데서 끝내지 않고, 각 개념과 알고리즘이 어떤 의미인지 이해하고 나중에 다시 봐도 복습할 수 있도록 정리하는 것을 목표로 합니다.

---

# 1. Machine Learning 기본 개념

## Machine Learning이란?

Machine Learning은 사람이 모든 규칙을 직접 작성하는 대신, 데이터를 이용해 컴퓨터가 규칙이나 패턴을 학습하도록 하는 방법입니다.

일반적인 프로그램은 사람이 규칙을 작성합니다.

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

---

# 2. Feature / Label / Prediction

Machine Learning을 이해할 때 가장 먼저 구분해야 하는 개념입니다.

## Feature

Feature는 모델이 판단할 때 사용하는 입력값 또는 속성입니다.

예를 들어 공부시간을 이용해 시험점수를 예측한다면:

```text
공부시간 → 시험점수
```

여기서 공부시간이 Feature입니다.

```python
x = np.array([1, 2, 3, 4, 5])
```

Feature가 여러 개일 수도 있습니다.

예:

```text
공부시간
수면시간
출석률
과제점수
   ↓
시험점수
```

이 경우 공부시간, 수면시간, 출석률, 과제점수가 모두 Feature입니다.

---

## Label

Label은 모델이 학습할 때 사용하는 실제 정답값입니다.

예:

```text
공부시간 1시간 → 시험점수 50점
공부시간 2시간 → 시험점수 60점
```

여기서:

```text
Feature = 공부시간
Label   = 시험점수
```

입니다.

분류 문제에서는 Label이 다음처럼 표현될 수도 있습니다.

```text
0 = 불합격
1 = 합격
```

---

## Prediction

Prediction은 학습이 끝난 모델이 새로운 Feature를 보고 예측한 결과입니다.

예:

```text
학습 데이터

1시간 → 50점
2시간 → 60점
3시간 → 70점
4시간 → 80점
5시간 → 90점
```

새로운 입력:

```text
6시간
```

모델의 예측:

```text
100점
```

여기서 100점이 Prediction입니다.

---

## Label과 Prediction 차이

```text
Label      = 실제 정답
Prediction = 모델이 예상한 값
```

예:

```text
실제 점수(Label)      = 85점
모델 예측(Prediction) = 82점
```

두 값의 차이가 작을수록 모델이 잘 예측했다고 볼 수 있습니다.

---

# 3. 기대값(Expected Value)

기대값은 Machine Learning의 Label과는 다른 개념입니다.

기대값은 여러 결과가 발생할 수 있을 때 각각의 확률을 고려해 계산한 평균적인 예상값입니다.

예를 들어:

```text
50% 확률로 0원
50% 확률로 100원
```

이라면 기대값은:

```text
0 × 0.5 + 100 × 0.5 = 50
```

즉 장기적으로 평균 50원 정도를 기대할 수 있다는 의미입니다.

Machine Learning을 공부하면서 확률, Logistic Regression, 분류 모델 등을 이해할 때 기대값이나 확률 개념이 등장할 수 있습니다.

---

# 4. 선형과 비선형

## 선형(Linear)

입력값이 변할 때 결과값이 일정한 비율로 변하는 관계입니다.

예:

```text
x = 1 → y = 10
x = 2 → y = 20
x = 3 → y = 30
x = 4 → y = 40
```

x가 1 증가할 때마다 y가 10씩 증가합니다.

그래프로 그리면 직선이 됩니다.

대표적인 식:

```text
y = ax + b
```

여기서:

```text
x = 입력값
y = 결과값
a = 기울기
b = 절편
```

---

## 기울기

기울기는 x가 1만큼 변할 때 y가 얼마나 변하는지를 의미합니다.

예:

```text
y = 10x + 40
```

여기서 기울기는 10입니다.

즉:

```text
x가 1 증가
↓
y는 10 증가
```

합니다.

---

## 절편

절편은 x가 0일 때의 y값입니다.

예:

```text
y = 10x + 40
```

x에 0을 넣으면:

```text
y = 10 × 0 + 40
y = 40
```

따라서 절편은 40입니다.

그래프에서는 직선이 y축과 만나는 위치입니다.

---

## 비선형(Non-linear)

입력값과 결과값의 관계를 직선 하나로 표현하기 어려운 관계입니다.

예:

```text
x = 1 → y = 1
x = 2 → y = 4
x = 3 → y = 9
x = 4 → y = 16
x = 5 → y = 25
```

이 데이터는:

```text
y = x²
```

관계입니다.

변화량을 보면:

```text
1 → 4  = +3
4 → 9  = +5
9 → 16 = +7
16 → 25 = +9
```

변화량이 일정하지 않습니다.

중요:

```text
비선형 = 값이 랜덤하게 오락가락한다
```

는 의미가 아닙니다.

비선형에도 명확한 규칙이 있을 수 있습니다.

핵심은:

```text
선형   = 직선으로 설명 가능
비선형 = 직선 하나로 설명하기 어려움
```

입니다.

---

# 5. Linear Regression

Linear Regression은 숫자를 예측하는 대표적인 선형 회귀 모델입니다.

쉽게 말하면 데이터에 가장 잘 맞는 직선을 찾는 모델입니다.

예:

```text
공부시간 → 시험점수
```

실습 코드:

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Feature
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# Label
y = np.array([50, 60, 70, 80, 90])

# 모델 생성
model = LinearRegression()

# 학습
model.fit(x, y)

# 새로운 값 예측
prediction = model.predict([[6]])

print(prediction)
print("기울기 :", model.coef_)
print("절편 :", model.intercept_)
```

이 데이터에서는 모델이 대략 다음 관계를 찾습니다.

```text
y = 10x + 40
```

따라서:

```text
x = 6
y = 10 × 6 + 40
y = 100
```

이 되어 100을 예측합니다.

---

# 6. 비선형 데이터를 Linear Regression에 넣으면?

다음 데이터는 비선형 관계입니다.

```python
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])
```

실제 관계는:

```text
y = x²
```

입니다.

하지만 Linear Regression은 직선밖에 만들 수 없습니다.

그래서 실제 곡선을 완벽하게 표현하는 대신 전체 데이터를 최대한 잘 설명하는 직선을 찾습니다.

실습에서는 다음과 비슷한 결과가 나왔습니다.

```text
기울기 = 6
절편 = -7
```

즉 모델이 찾은 관계:

```text
y = 6x - 7
```

예:

```text
x = 6 → 29
x = 7 → 35
x = 8 → 41
```

하지만 실제 x² 관계라면:

```text
6² = 36
7² = 49
8² = 64
```

입니다.

이 실습을 통해 알 수 있는 점:

> 데이터의 관계에 맞는 모델을 선택하는 것이 중요하다.

데이터를 많이 넣는다고 해서 모든 문제가 해결되는 것은 아닙니다.

---

# 7. Regression과 Classification

Machine Learning에서는 해결하려는 문제에 따라 모델의 종류가 달라집니다.

## Regression

숫자를 예측합니다.

예:

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

대표 모델:

```text
Linear Regression
Random Forest Regressor
```

---

## Classification

종류 또는 클래스를 예측합니다.

예:

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

대표 모델:

```text
Logistic Regression
Decision Tree
Random Forest
```

---

# 8. Logistic Regression

이름에는 Regression이 들어가지만 주로 Classification에 사용하는 모델입니다.

예:

```text
공부시간 → 합격 / 불합격
```

실습 코드:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Feature: 공부시간
x = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# Label
# 0 = 불합격
# 1 = 합격
y = np.array([0, 0, 0, 1, 1, 1])

# 모델 생성
model = LogisticRegression()

# 학습
model.fit(x, y)

# 분류
prediction = model.predict([[2.5], [3.5], [5]])

print("예측 결과 :", prediction)

# 각 클래스에 속할 확률
probability = model.predict_proba([[3.5]])

print("확률 :", probability)
```

실행 결과 예:

```text
예측 결과 : [0 1 1]

확률 :
[[0.49996763 0.50003237]]
```

의미:

```text
3.5시간 공부

불합격(0) 확률 ≈ 49.99%
합격(1) 확률   ≈ 50.00%
```

합격 확률이 기준값 0.5를 조금 넘었기 때문에 1로 분류됩니다.

핵심:

```text
Linear Regression
→ 숫자를 예측

Logistic Regression
→ 확률을 계산하고 클래스를 분류
```

---

# 9. fit / predict / predict_proba

## fit()

모델을 학습시키는 함수입니다.

```python
model.fit(x, y)
```

의미:

```text
Feature x와 Label y의 관계를 학습해라
```

---

## predict()

학습된 모델에 새로운 데이터를 넣어 결과를 예측합니다.

```python
model.predict([[6]])
```

---

## predict_proba()

분류 모델에서 각 클래스에 속할 확률을 보여줍니다.

```python
model.predict_proba([[3.5]])
```

예:

```text
[0.4, 0.6]
```

이라면:

```text
0일 확률 = 40%
1일 확률 = 60%
```

입니다.

---

# 10. reshape(-1, 1)

scikit-learn의 모델은 Feature 데이터를 보통 다음 형태로 받습니다.

```text
[샘플 개수, Feature 개수]
```

예:

```text
1시간
2시간
3시간
4시간
5시간
```

은 샘플 5개, Feature 1개입니다.

그래서:

```python
np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
```

을 사용해 다음과 같은 구조로 바꿉니다.

```text
[
 [1],
 [2],
 [3],
 [4],
 [5]
]
```

`-1`은 NumPy가 행의 개수를 자동으로 계산하라는 의미입니다.

---

# 11. scikit-learn 구조

`sklearn`은 Machine Learning에서 많이 사용하는 Python 라이브러리입니다.

전체 도구상자라고 생각하면 이해하기 쉽습니다.

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

예:

```python
from sklearn.linear_model import LinearRegression
```

의미:

```text
sklearn
 └─ linear_model
      └─ LinearRegression
```

---

# 12. 대표적인 모델 학습 순서

현재 목표:

1. Linear Regression
   - 선형 회귀
   - 직선 관계를 이용한 숫자 예측

2. Logistic Regression
   - 로지스틱 회귀
   - 확률을 이용한 분류

3. Decision Tree
   - 의사결정나무
   - 조건을 따라가며 판단

4. Random Forest
   - 랜덤 포레스트
   - 여러 Decision Tree의 판단을 결합

5. K-Means
   - K-평균 군집화
   - 비슷한 데이터끼리 그룹화

6. Isolation Forest
   - 이상치 탐지
   - 다른 데이터와 동떨어진 값 탐지

---

# 13. Machine Learning 문제 유형

## 지도학습

정답(Label)이 존재하는 데이터를 이용해 학습합니다.

대표 문제:

```text
Regression
Classification
```

---

## 비지도학습

정답(Label)이 없는 데이터에서 패턴을 찾습니다.

대표 문제:

```text
Clustering
Anomaly Detection
```

---

# 14. 기본 Machine Learning 흐름

```text
데이터 준비
   ↓
Feature / Label 확인
   ↓
문제 유형 판단
   ↓
모델 선택
   ↓
model.fit()
   ↓
학습
   ↓
새로운 Feature 입력
   ↓
model.predict()
   ↓
예측 결과 확인
   ↓
모델 평가
```

Machine Learning 공부에서 중요한 것은 단순히 코드를 외우는 것이 아닙니다.

최종적으로는:

> 어떤 문제인지 파악하고, 그 문제에 어떤 모델을 사용하는 것이 적절한지 판단할 수 있어야 합니다.

---

# 15. 현재 진행 상황

- [x] AI / ML / DL 기본 관계
- [x] Feature
- [x] Label
- [x] Prediction
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
- [ ] Decision Tree
- [ ] Random Forest
- [ ] K-Means
- [ ] Isolation Forest
- [ ] Train / Test Split
- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] F1 Score
- [ ] Overfitting / Underfitting
- [ ] 실제 Dataset 기반 실습

---

# 16. Tech Stack

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

# 17. Environment

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

# 18. Project Goal

이 저장소의 목표는 Machine Learning 알고리즘을 단순히 사용하는 것이 아니라 다음 질문에 답할 수 있는 수준까지 이해하는 것입니다.

```text
이 문제는 회귀인가?
분류인가?
군집화인가?
이상탐지인가?

Label이 있는가?

어떤 Feature를 사용할 것인가?

왜 이 모델을 선택했는가?

다른 모델보다 어떤 장단점이 있는가?
```

장기적으로는 기존 Backend 개발 경험과 Machine Learning / LLM 기술을 연결해 실제 시스템에 AI를 적용할 수 있는 개발 역량을 만드는 것을 목표로 합니다.
