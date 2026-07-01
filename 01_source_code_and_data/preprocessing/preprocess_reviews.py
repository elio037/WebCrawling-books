# -*- coding: utf-8 -*-
"""
알라딘 리뷰 전처리: 책과 무관한 리뷰(배송·품절·특전·구매제한 등) 제거
- 입력 : aladin_reviews_clean_3col.csv (원본 100만)
- 출력 : aladin_reviews_filtered.csv   (정제 데이터 → 앱에서 사용)
"""
import sys, re
import pandas as pd
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = r"C:\Users\user\Desktop\크롤링 내용\aladin_reviews_clean_3col.csv"
OUT = r"D:\Lecture\_AIService26\data\aladin_reviews_filtered.csv"

OFF_TOPIC_KEYWORDS = [
    "배송", "포장", "택배", "배달", "발송", "품절", "재입고", "재고", "입고",
    "절판", "파본", "매진",
    "환불", "교환", "주문", "결제", "할인", "적립", "쿠폰",
    "굿즈", "사은품", "특전", "더특", "특장판", "특정판", "특별판", "한정판", "리커버",
    "되팔", "플미", "프리미엄", "재판매", "사재기", "수량", "제한", "판매량",
    "알림", "알람", "추첨", "응모", "미출간", "예판", "예약",
]
OFF_RE = re.compile("|".join(map(re.escape, OFF_TOPIC_KEYWORDS)))

print("원본 로딩 중...")
df = pd.read_csv(SRC, encoding="utf-8-sig",
                 usecols=lambda c: c in ["book_title", "review_content", "rating"])
n0 = len(df)

# 결측·평점 정제
df = df.dropna(subset=["book_title", "review_content", "rating"])
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df[df["rating"].between(1, 5)].copy()
df["rating"] = df["rating"].astype(int)
df["book_title"] = df["book_title"].astype(str).str.strip()
df["review_content"] = df["review_content"].astype(str)

# 책과 무관한 리뷰 제거
mask_off = df["review_content"].str.contains(OFF_RE, na=False)
removed = int(mask_off.sum())
df = df[~mask_off]

df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"전처리 전 : {n0:,}")
print(f"무관 제거  : {removed:,}")
print(f"정제 결과  : {len(df):,}  →  {OUT}")
