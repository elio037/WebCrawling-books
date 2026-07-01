# -*- coding: utf-8 -*-
"""
알라딘 리뷰 감성분석 학습 스크립트
- 입력 : aladin_reviews_clean_part01~04.csv  (book_title, review_content, rating)
- 모델 : TF-IDF(문자 n-gram) + LinearSVC  (CPU, 설치 추가 불필요)
실행 : python train_sentiment.py
"""
import os, glob, sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

try: sys.stdout.reconfigure(encoding="utf-8")
except: pass

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) part01~04 읽어서 합치기 ------------------------------------------------
files = sorted(glob.glob(os.path.join(DATA_DIR, "aladin_reviews_clean_part0[1-4].csv")))
df = pd.concat([pd.read_csv(f, encoding='utf-8-sig') for f in files], ignore_index=True)
print(f"불러온 데이터: {df.shape}")

# 2) 레이블 생성 : 이진(4~5 긍정 / 1~2 부정), 모호한 3점은 제외 -----------------
df = df[df['rating'] != 3].copy()
df['label'] = (df['rating'] >= 4).astype(int)   # 1=긍정, 0=부정
df = df.dropna(subset=['review_content'])

# 3) 클래스 불균형 보정 : 적은 쪽(부정)에 맞춰 균형 샘플링 ----------------------
n = df['label'].value_counts().min()
df = df.groupby('label', group_keys=False).sample(n=n, random_state=42)
print(f"균형 샘플링 후: {df.shape}  (클래스별 {n:,}건)")

# 4) 학습/평가 분할 --------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    df['review_content'], df['label'],
    test_size=0.2, stratify=df['label'], random_state=42)

# 5) TF-IDF (문자 2~4-gram → 형태소 분석기 없이 빠르고 한국어에 강함) -----------
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                        min_df=3, max_features=300_000, sublinear_tf=True)
Xtr = tfidf.fit_transform(X_tr)
Xte = tfidf.transform(X_te)
print("TF-IDF 행렬:", Xtr.shape)

# 6) 모델 학습 : LinearSVC (불균형 보정) ------------------------------------
clf = LinearSVC(class_weight='balanced', C=1.0)
clf.fit(Xtr, y_tr)

# 7) 평가 : accuracy 말고 macro-F1 / confusion matrix ----------------------
pred = clf.predict(Xte)
print("\n=== 분류 리포트 ===")
print(classification_report(y_te, pred, target_names=['부정(0)', '긍정(1)'], digits=4))
print("=== 혼동행렬 ===\n", confusion_matrix(y_te, pred))

# 8) 직접 테스트 ----------------------------------------------------------
samples = ["정말 감동적이고 다시 읽고 싶은 책이에요", "내용도 부실하고 시간 낭비였습니다", "그냥 그저 그런 책"]
for s, p in zip(samples, clf.predict(tfidf.transform(samples))):
    print(f"  [{'긍정' if p==1 else '부정'}] {s}")

# 9) (선택) 모델 저장 -----------------------------------------------------
import joblib
joblib.dump({'tfidf': tfidf, 'clf': clf}, os.path.join(DATA_DIR, "sentiment_model.joblib"))
print("\n모델 저장 완료 → sentiment_model.joblib")
