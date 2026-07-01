# 01. 소스 코드 + 데이터 (Source Code + Data)

웹 크롤링부터 전처리·학습까지의 **소스 코드**와, 크롤링으로 수집한 **원본 CSV 데이터**입니다.

## 구성
```
01_source_code_and_data/
├── crawler/                # 알라딘 리뷰 크롤러 (100만 건 수집)
│   └── aladin_v2_crawler/      # 알라딘 댓글/마이리뷰 API 기반 대규모 수집기
│                               #   (출력: aladin_clean_reviews.csv → data/의 조각 파일)
├── preprocessing/          # 전처리 스크립트
│   ├── preprocess_reviews.py     # 무관 리뷰(배송·특전 등) 제거 → filtered
│   ├── preprocess_sentiment.py   # 평점→감성 라벨링/정제
│   └── train_sentiment.py        # (초기) 감성 모델 학습 스크립트
├── notebooks/              # 학습/실습 노트북
│   ├── 알라딘도서리뷰_실습.ipynb
│   └── 인공지능서비스개발_09_순환신경망기반감성분석_알라딘도서리뷰_실습.ipynb
└── data/                   # 크롤링 원본 CSV (전처리 입력)
    └── aladin_clean_reviews_part01~28.csv   # 크롤링 원본 리뷰 100만 (조각, 각 <25MB)
```

## 데이터 흐름 (파이프라인)
```
[크롤링]  crawler/  ─►  aladin_clean_reviews_part01~28.csv   (원본 8컬럼: book_id, title, review_id, reviewer, review, rating, date)
                          │
[병합/정리] ─────────────►  3컬럼(title, review, rating) 으로 통합
                          │
[전처리]  preprocess_reviews.py ─► aladin_reviews_filtered.csv  (무관 리뷰 제거, 약 100만)   → 02번 폴더 data/filtered/
                          │
[라벨/균형/토큰화] ───────►  aladin_review_ing.csv  (3클래스 균형 7.7만, 토큰화)              → 02번 폴더 data/ing/
                          │
[학습]  notebooks/ ──────►  sa_model_aladin.keras (LSTM)                                    → 02번 폴더 model/
```

## 원본 데이터 형식 (`aladin_clean_reviews_part*.csv`)
- 인코딩: **CP949(EUC-KR)** — 엑셀에서 열면 한글 정상, pandas는 `encoding='cp949'` 지정 필요
- 컬럼(8개): `book_id, book_title, review_id, reviewer, review_content, rating, created_at, (빈칸)`
- 이 원본을 3컬럼으로 정리하고 무관 리뷰를 제거한 결과가 02번 폴더의 전처리 데이터입니다.

> 용량이 큰 통합본(`aladin_reviews_clean_3col.csv` 597MB 등)은 GitHub 용량 제한으로 제외했고, 위 조각 파일로 동일하게 재구성할 수 있습니다.
