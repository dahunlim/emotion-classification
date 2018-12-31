# emotion-classification
NLP with LSTM


# Data
사용된 데이터로는 보배드림, 82cook, 루리웹, 와이고수의 게시글을 스크래핑하여 사용하였으며
직접 데이터의 감정을 레이블링 하였다.
(0 : 행복, 1 : 슬픔, 2 : 분노, 3 : 혐오, 4 : 두려움, 5 : 놀라움)

게시글 감정을 설정하는 과정에 있어 데이터의 유효성을 높이기 위하여 게시글에 감정을 직접적으로 나타내는 단어가 있는 게시글만 데이터로 활용하였다.


# Proprocessing
1. 형태소 분석
KoNLPy 패키지를 사용하였다. 그 중에서도 twitter에서 만든 한국어 처리기를 사용하여 명사만을 사용하였다.

2. 벡터화
scikit-learn에서 제공하는 CountVectorizer를 사용하여 빈도수를 체크한 뒤 TF-IDF를 이용하여 데이터를 벡터화한뒤 가공하였다.


# Model
LSTM을 사용하여 기존의 RNN이 가지고 있던 Vanishing Gradient 문제를 해결하고자 하였다.


# Conclusion
약 2800여개의 데이터를 가지고 80%는 train, 20%는 test 데이터로 활용하였으며 Cross-validation을 진행하였다.
그 결과 Acivation=relu, Dense=512, Dropout = 0.7, lstm 30 일 때 84%의 정확도를 확인할 수 있었다.
