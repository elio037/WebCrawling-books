# -*- coding: utf-8 -*-
"""
알라딘 도서 리뷰 감성분석 전처리 파이프라인
- 입력 : aladin_clean_reviews_part01~28.csv  (혼합 인코딩: utf-8-sig / cp949)
- 출력 : aladin_reviews_for_sentiment.csv     (모델 학습용 정제 데이터)
실행 : python preprocess_sentiment.py
"""
import os, re, glob, sys
import numpy as np
import pandas as pd

# 윈도우 콘솔에서 한글 깨짐 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
USECOLS  = ['book_id','book_title','review_id','reviewer','review_content','rating','created_at']

# ----------------------------------------------------------------------
# 0. 로딩 : 파트 파일이 utf-8-sig / cp949 로 섞여 있어 파일별 인코딩 자동 판별
# ----------------------------------------------------------------------
def smart_read(path):
    for enc in ['utf-8-sig', 'cp949']:
        try:
            return pd.read_csv(path, encoding=enc,
                               usecols=lambda c: c in USECOLS, dtype=str)
        except UnicodeDecodeError:
            continue
    # 둘 다 실패하면 손상 바이트를 치환해서라도 읽음
    return pd.read_csv(path, encoding='cp949', encoding_errors='replace',
                       usecols=lambda c: c in USECOLS, dtype=str)

def load_all():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "aladin_clean_reviews_part*.csv")))
    df = pd.concat([smart_read(f) for f in files], ignore_index=True)
    return df

# ----------------------------------------------------------------------
# 1. 텍스트 정제 : HTML/URL/이모지/특수문자 제거 + 공백 정리
# ----------------------------------------------------------------------
RE_HTML  = re.compile(r'<[^>]+>')
RE_URL   = re.compile(r'(https?://|www\.)\S+')
RE_EMOJI = re.compile("[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
                          "\U0001F1E6-\U0001F1FF" "\U00002190-\U000021FF"
                          "\U0000FE00-\U0000FE0F" "]+", flags=re.UNICODE)
RE_HANGUL_ONLY = re.compile(r'[^가-힣\s]')   # 한글/공백만 남김
RE_MULTISPACE  = re.compile(r'\s+')

def clean_text(t: str) -> str:
    """원본 보존용 가벼운 정제 (HTML/URL/이모지 제거, 공백 정리)"""
    if not isinstance(t, str):
        return ""
    t = RE_HTML.sub(' ', t)
    t = RE_URL.sub(' ', t)
    t = RE_EMOJI.sub(' ', t)
    t = RE_MULTISPACE.sub(' ', t).strip()
    return t

def to_hangul(t: str) -> str:
    """NLP 토큰화 직전용 : 한글만 남김"""
    if not isinstance(t, str):
        return ""
    t = RE_HANGUL_ONLY.sub(' ', t)
    return RE_MULTISPACE.sub(' ', t).strip()

# ----------------------------------------------------------------------
# 2. 감성 레이블 생성
# ----------------------------------------------------------------------
def make_label_3class(r):                       # 1~2 부정 / 3 중립 / 4~5 긍정
    if r >= 4: return 'positive'
    if r == 3: return 'neutral'
    return 'negative'

# ----------------------------------------------------------------------
# 메인 전처리
# ----------------------------------------------------------------------
def preprocess(min_len=10, drop_neutral=False, dedup=True):
    df = load_all()
    n0 = len(df)
    print(f"[전처리 전] {n0:,} 행, {df.shape[1]} 열")

    # (a) 핵심 컬럼 결측 제거
    df = df.dropna(subset=['review_content', 'rating'])

    # (b) rating 숫자화 + 유효범위(1~5) 밖 이상치 제거  (15,34,47 등 크롤링 오류)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[df['rating'].between(1, 5)]
    df['rating'] = df['rating'].astype(int)

    # (c) 날짜 파싱
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    # (d) 텍스트 정제
    df['clean_text']  = df['review_content'].map(clean_text)   # 원문 정제본
    df['hangul_text'] = df['clean_text'].map(to_hangul)        # 한글만

    # (e) 너무 짧은 리뷰 제거 (한글 정제 후 기준)
    df = df[df['hangul_text'].str.len() >= min_len]

    # (f) 중복 제거 : 같은 사람이 같은 책에 남긴 동일 내용
    if dedup:
        df = df.drop_duplicates(subset=['book_id', 'reviewer', 'review_content'])
        df = df.drop_duplicates(subset=['review_content'])     # 완전 동일 문구(도배) 제거

    # (g) 레이블
    df['label'] = df['rating'].map(make_label_3class)
    if drop_neutral:                                           # 이진 분류용
        df = df[df['label'] != 'neutral']

    df = df.reset_index(drop=True)
    print(f"[전처리 후] {len(df):,} 행  (제거 {n0-len(df):,} / {(1-len(df)/n0)*100:.1f}%)")
    print("\n[레이블 분포]\n", df['label'].value_counts())
    return df

# ----------------------------------------------------------------------
# 3. (선택) KoNLPy Okt 토큰화 컬럼 추가
# ----------------------------------------------------------------------
STOPWORDS = set("의 가 이 은 들 는 좀 잘 걍 과 도 를 으로 자 에 와 한 하다 에서 그 등 및 또 더 "
                "너무 정말 진짜 책 읽다 보다 있다 없다 되다 이다 같다 이런 그런 저런".split())

def add_tokens(df, text_col='hangul_text', sample=None):
    from konlpy.tag import Okt
    okt = Okt()
    target = df.sample(sample, random_state=0) if sample else df
    def nouns(t):  return [w for w in okt.nouns(t) if len(w) > 1 and w not in STOPWORDS]
    def morphs(t): return [w for w, p in okt.pos(t, stem=True)
                           if p in ('Noun','Verb','Adjective') and len(w) > 1 and w not in STOPWORDS]
    out = target.copy()
    out['tokens_noun']  = out[text_col].map(lambda t: ' '.join(nouns(t)))
    out['tokens_morph'] = out[text_col].map(lambda t: ' '.join(morphs(t)))
    return out

# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = preprocess(min_len=10, drop_neutral=False)

    out_path = os.path.join(DATA_DIR, "aladin_reviews_for_sentiment.csv")
    keep = ['review_id','book_id','book_title','rating','label',
            'clean_text','hangul_text','created_at']
    df[keep].to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n저장 완료 → {out_path}")
    print(df[keep].head())
