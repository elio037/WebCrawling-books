# -*- coding: utf-8 -*-
"""알라딘 도서 리뷰 감성분석 LSTM 노트북 생성기 (네이버 영화리뷰 실습과 동일 구조)"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(src):   cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

md("""# 순환 신경망(RNN/LSTM) 기반 감성 분석 (알라딘 도서 리뷰)
1. 데이터 준비
2. 모델 구조 및 학습 설계 (구축)
3. 모델 학습
4. 모델 평가
5. 예측
6. 배포 (저장, 재사용)""")

# ===================== 1. 데이터 준비 =====================
md("""## 1. 데이터 준비
    1-1. 데이터 로딩
    1-2. 데이터 전처리
    1-3. 데이터 분리
    1-4. 학습용 데이터 준비
    1-5. 테스트용 데이터 준비""")

md("""### 1-1. 데이터 로딩
* 알라딘 리뷰 정제본 part01~04 (book_title, review_content, rating) 를 합쳐서 사용""")
code("""# 알라딘 리뷰 데이터 로딩 (part01~04 합치기)
import pandas as pd
import glob, os

DATA_DIR = r"C:\\\\Users\\\\user\\\\Desktop\\\\크롤링 내용"
files = sorted(glob.glob(os.path.join(DATA_DIR, "aladin_reviews_clean_part0[1-4].csv")))
review_df = pd.concat([pd.read_csv(f, encoding='utf-8-sig') for f in files], ignore_index=True)
review_df.head()""")

code("""# 평점(rating) 분포 확인
review_df.rating.value_counts().sort_index()""")

code("""review_df.info()""")

md("""### 1-2. 데이터 전처리
- 평점 기반 정답(label) 생성
- 결측치 제거
- 정제 (한글만 남기고 모두 삭제)
- 중복치 제거
- 클래스 불균형 보정 (균형 샘플링)
- 형태소 분석기로 토큰화""")

md("""#### 1-2-1. 정답(label) 생성
* 평점 1~2점 → 부정(0), 3점 → 중립(1), 4~5점 → 긍정(2)
* 3개 클래스(부정/중립/긍정) 다중 분류""")
code("""# 평점으로 3개 클래스 레이블 생성 (0=부정, 1=중립, 2=긍정)
def to_label(r):
    if r >= 4: return 2   # 긍정
    if r == 3: return 1   # 중립
    return 0              # 부정
review_df['label'] = review_df.rating.apply(to_label)
review_df.label.value_counts().sort_index()""")

md("""#### 1-2-2. 결측치 제거""")
code("""# 결측치 확인
review_df.isnull().sum()""")
code("""# review_content 결측치 제거
review_df.dropna(subset=['review_content'], inplace=True)
review_df.isnull().sum()""")

md("""#### 1-2-3. 정제
* 한글과 공백을 제외한 문자는 공백으로 치환하여 제거
* 한글이 없었던 리뷰는 공백만 남으므로, 결측치로 만들어 삭제 처리""")
code("""import re
# 한글과 공백 제외하고 모두 공백으로 치환
review_df['clean_review'] = review_df.review_content.apply(lambda x : re.sub('[^ 가-힣]+', ' ', str(x)))
# 문장 앞쪽 공백 제거
review_df.clean_review = review_df.clean_review.apply(lambda x : re.sub('^ +', '', x))
# 빈 문자열("")은 결측치로 변환
review_df.clean_review = review_df.clean_review.replace('', None)
review_df.head()""")
code("""# 정제 후 결측치 확인
review_df.clean_review.isnull().sum()""")
code("""# 결측치 제거
review_df.dropna(subset=['clean_review'], inplace=True)
review_df.clean_review.isnull().sum()""")

md("""#### 1-2-4. 중복치 제거""")
code("""# 중복치 확인
review_df.clean_review.duplicated().sum()""")
code("""# 중복치 제거
review_df.drop_duplicates(subset=['clean_review'], inplace=True)
review_df.clean_review.duplicated().sum()""")

md("""#### 1-2-5. 클래스 불균형 보정 (균형 샘플링)
* 알라딘 리뷰는 긍정(4~5점)이 90% 이상으로 매우 불균형함
* 가장 적은 클래스(부정) 수에 맞춰 세 클래스를 같은 수로 샘플링 → 균형 데이터 구성""")
code("""# 클래스별 개수 확인
print(review_df.label.value_counts())
# 적은 클래스 수에 맞춰 균형 샘플링
n = review_df.label.value_counts().min()
review_df = review_df.groupby('label', group_keys=False).sample(n=n, random_state=42).reset_index(drop=True)
review_df.label.value_counts()""")

md("""#### 1-2-6. 토큰화""")
code("""# 형태소 분석기 적용 (Okt)
from konlpy.tag import Okt
from tqdm import tqdm
tqdm.pandas()

review_df['tokens'] = review_df.clean_review.progress_apply(Okt().morphs)""")
code("""review_df['tokens_str'] = review_df.tokens.apply(lambda x : ' '.join(x))
review_df.head()""")
code("""# 전처리 결과 저장 (재사용용)
review_df.to_csv(os.path.join(DATA_DIR, 'aladin_review_ing.csv'))""")

md("""### 1-3. 데이터 분리
* 정답 데이터의 분포를 유지하며 학습/테스트로 분리 (stratify=label)""")
code("""# 입력 데이터와 정답 데이터 추출 (list)
review_list = list(review_df.tokens_str)
label_list = list(review_df.label)""")
code("""# label 분포 확인
review_df.label.value_counts()""")
code("""# 막대그래프로 그려보기 (부정/중립/긍정 3개 클래스)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
name_map = {0: '부정', 1: '중립', 2: '긍정'}
review_df.label.map(name_map).value_counts().reindex(['부정', '중립', '긍정']).plot(kind='bar', rot=0)""")
code("""# 학습 데이터와 테스트 데이터 분리
from sklearn.model_selection import train_test_split

review_train, review_test, label_train, label_test = train_test_split(
    review_list, label_list, test_size=0.1, stratify=label_list, random_state=42)
len(review_train), len(review_test), len(label_train), len(label_test)""")

md("""### 1-4. 학습 데이터 준비 for tensorflow
    1-4-1. Integer Encoding을 위한 tokenizer 생성
    1-4-2. 입력 데이터 Integer Encoding
    1-4-3. 입력 데이터 Padding
    1-4-4. 정답 데이터 원핫인코딩""")

md("""#### 1-4-1. Integer Encoding을 위한 tokenizer 생성
* num_words = 사용할 단어 수(vocab_size) + 1 (0은 OOV/패딩에 할당)""")
code("""# 단어 수 제한 없이 Tokenizer 생성하여 단어 수 확인
from tensorflow.keras.preprocessing.text import Tokenizer
test_tokenizer = Tokenizer()
test_tokenizer.fit_on_texts(review_train)
list(test_tokenizer.word_index.items())[:10]""")
code("""from lib.my_utils import word_status_below_threshold
# 등장 빈도수를 threshold로 설정하여 버릴 단어가 차지하는 비율 확인
threshold = 3
word_status_below_threshold(test_tokenizer, threshold)""")
code("""# 단어 수를 제한하여 tokenizer 생성
from tensorflow.keras.preprocessing.text import Tokenizer
vocab_size = 30000
num_words = vocab_size + 1
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(review_train)
len(tokenizer.word_index)""")

md("""#### 1-4-2. 입력 데이터 Integer Encoding
* 제한된 단어에만 index를 부여하므로, 희귀 단어로만 구성된 리뷰는 길이가 0이 됨 → 제거""")
code("""# 입력 데이터 Integer Encoding
encoded_review_train = tokenizer.texts_to_sequences(review_train)
print(encoded_review_train[:5])""")
code("""# 길이가 0인 리뷰의 index 추출
null_index = [index for index, review in enumerate(encoded_review_train) if len(review) < 1]
print(len(null_index))
null_index[:10]""")
code("""# 길이가 1 이상인 리뷰만으로 학습 데이터 재구성
new_review_train = [review for index, review in enumerate(encoded_review_train) if index not in null_index]
new_label_train = [label for index, label in enumerate(label_train) if index not in null_index]
len(new_review_train), len(new_label_train)""")

md("""#### 1-4-3. 입력 데이터 padding
* 입력 데이터의 길이(max_len)를 정하여 padding""")
code("""# 리뷰 길이 분포 확인 (히스토그램)
len_df = pd.DataFrame([len(review) for review in new_review_train])
len_df.hist()""")
code("""# 길이 통계 확인
len_df.describe()""")
code("""from lib.my_utils import text_len_status_below_maxlen
# 길이가 max_len 이하인 데이터의 비중 확인
max_len = 60
text_len_status_below_maxlen(new_review_train, max_len)""")
code("""# max_len 길이로 입력 데이터 padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_X = pad_sequences(new_review_train, maxlen=max_len)
len(train_X), train_X[:2]""")

md("""#### 1-4-4. 정답 데이터 one-hot encoding""")
code("""from tensorflow.keras.utils import to_categorical
train_y = to_categorical(new_label_train)
len(train_y), train_y[:2]""")

md("""### 1-5. 테스트 데이터 준비
    1-5-1. 입력 데이터 Integer Encoding (결측치 제거)
    1-5-2. 입력 데이터 padding
    1-5-3. 정답 데이터 one-hot encoding""")
code("""# 입력 데이터 Integer Encoding
encoded_review_test = tokenizer.texts_to_sequences(review_test)
print(encoded_review_test[:2])""")
code("""# 길이가 0인 리뷰 제거하여 테스트 데이터 재구성
null_index = [index for index, review in enumerate(encoded_review_test) if len(review) == 0]
new_review_test = [review for index, review in enumerate(encoded_review_test) if index not in null_index]
new_label_test = [label for index, label in enumerate(label_test) if index not in null_index]
len(new_review_test), len(new_label_test)""")
code("""# 입력 데이터 padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
test_X = pad_sequences(new_review_test, maxlen=max_len)
len(test_X), test_X[:2]""")
code("""# 정답 데이터 one-hot encoding
from tensorflow.keras.utils import to_categorical
test_y = to_categorical(new_label_test)
len(test_y), test_y[:5]""")

# ===================== 2. 모델 구축 =====================
md("""## 2. 모델 구축 및 컴파일""")
code("""# 신경망 구조 설계
from tensorflow.keras.layers import Embedding, LSTM, Dense

input_units = num_words      # 사용한 feature 수 = 단어 수
embedding_dim = 32
lstm_units = 64
dense_units = 16
output_units = 3             # 분류 수 3개 (부정/중립/긍정)

rnn_model = [
    Embedding(input_units, embedding_dim),
    LSTM(lstm_units),
    Dense(dense_units, activation='tanh'),
    Dense(output_units, activation='softmax')
]""")
code("""# 신경망 구조 생성
from tensorflow.keras.models import Sequential
model = Sequential(rnn_model)
model.build(input_shape=(None, max_len))
model.summary()""")
code("""# 모델 학습 설계
from tensorflow.keras.optimizers import RMSprop
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=RMSprop(learning_rate=0.001))""")
code("""# EarlyStopping, ModelCheckpoint 설정
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
checkpoint_file = './model/best_model_aladin.keras'
mc = ModelCheckpoint(checkpoint_file, monitor='val_loss', mode='min', save_best_only=True)""")

# ===================== 3. 모델 학습 =====================
md("""## 3. 모델 학습""")
code("""# 모델 학습
history = model.fit(train_X, train_y, epochs=20, batch_size=128,
                    validation_split=0.1, callbacks=[es, mc])""")

# ===================== 4. 모델 평가 =====================
md("""## 4. 모델 평가""")
code("""# 저장된 best 모델 가중치 로딩
model.load_weights(checkpoint_file)""")
code("""# 테스트 데이터로 평가
loss, acc = model.evaluate(test_X, test_y)
loss, acc""")
code("""# predict로 예측값 구하기
import numpy as np
preds = model.predict(test_X)
result = [np.argmax(pred) for pred in preds]
result[:20]""")
code("""# classification_report로 평가 (정확도 말고 macro-F1 확인)
from sklearn.metrics import classification_report
print(classification_report(new_label_test, result, target_names=['부정', '중립', '긍정']))""")

# ===================== 5. 예측 =====================
md("""## 5. 예측""")
code("""# 입력된 리뷰의 긍/부정 판단 함수
from konlpy.tag import Okt
def analyze_sentiment(text):
    # 전처리 -> 형태소 분석 -> Integer Encoding -> Padding
    tokens = Okt().morphs(text)
    encoded_text = tokenizer.texts_to_sequences([tokens])
    X = pad_sequences(encoded_text, maxlen=max_len)
    preds = model.predict(X, verbose=0)
    labels = ['부정', '중립', '긍정']
    result_index = np.argmax(preds[0])
    return labels[result_index], preds[0][result_index]""")
code("""# 함수 테스트 (도서 리뷰 예시)
reviews = [
    '정말 감동적이고 다시 읽고 싶은 책이에요',
    '내용도 부실하고 시간 낭비였습니다',
    '작가의 문장력이 뛰어나서 술술 읽힌다',
    '돈이 아깝다 추천하지 않아요',
    '올해 읽은 책 중 최고의 인생책',
    '번역이 너무 어색해서 읽기 힘들었어요',
    '그냥 무난하게 읽을 만한 책이에요'
]
for review in reviews:
    result, prob = analyze_sentiment(review)
    print(f'{review} --> {result}({prob*100:.2f}%)')""")

# ===================== 6. 배포 =====================
md("""## 6. 배포 (모델 저장 및 재사용)
    6-1. 모델 저장
    6-2. SentimentAnalyzer 클래스 구현
        * 저장된 모델 로딩 및 사용""")
md("""### 6-1. 모델 저장""")
code("""# keras 학습 모델 저장
model.save('./model/sa_model_aladin.keras')""")
code("""# Integer Encoding용 tokenizer 직렬화
import joblib
joblib.dump(tokenizer, './model/sa_tokenizer_aladin.pkl')""")

md("""### 6-2. SentimentAnalyzer 클래스 구현
- 객체 생성 시 예측 모델, Integer Encoder 로딩
- 한국어 형태소 분석기 정의
- 입력된 리뷰의 긍/부정 판단 함수""")
code("""from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_file, tokenizer_file, max_len=60):
        try:
            self.model = load_model(model_file)
        except FileExistsError:
            print(f'모델 파일 {model_file}이 없습니다.')
        try:
            self.tokenizer = joblib.load(tokenizer_file)
        except FileExistsError:
            print(f'토크나이저 파일 {tokenizer_file}이 없습니다.')
        self.morphs = Okt().morphs
        self.max_len = max_len

    def analyze_sentiment(self, text):
        # 전처리 -> 형태소 분석 -> Integer Encoding -> Padding
        tokens = self.morphs(text)
        encoded_text = self.tokenizer.texts_to_sequences([tokens])
        X = pad_sequences(encoded_text, maxlen=self.max_len)
        preds = self.model.predict(X, verbose=0)
        labels = ['부정', '중립', '긍정']
        result_index = np.argmax(preds[0])
        return labels[result_index], preds[0][result_index]""")
code("""# 클래스 사용 예시
analyzer = SentimentAnalyzer('./model/sa_model_aladin.keras', './model/sa_tokenizer_aladin.pkl', max_len=60)
analyzer.analyze_sentiment('이 책 정말 강력 추천합니다')""")

nb['cells'] = cells
nb.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
out = 'D:\\\\Lecture\\\\_AIService26\\\\인공지능서비스개발_10_순환신경망기반감성분석_알라딘도서리뷰_실습.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('생성 완료:', out, '| 셀 수:', len(cells))
