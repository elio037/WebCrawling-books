# 📚 알라딘 도서 리뷰 분석 (웹 크롤링 + LSTM 감성분석)

알라딘 온라인 서점의 도서 리뷰 **약 100만 건**을 직접 크롤링하여,
책 검색·순위·감성분석을 제공하는 **Streamlit 웹 서비스**입니다.
임의의 리뷰를 **LSTM 모델**이 읽고 긍정/중립/부정을 판단합니다.

> AI서비스개발 · 웹크롤링 프로젝트 · 박종인

---

## 📁 저장소 구성

| 폴더 | 내용 |
|------|------|
| [`01_source_code_and_data/`](01_source_code_and_data/) | **소스 코드 + 데이터** — 크롤러, 전처리 스크립트, 학습 노트북, 크롤링 원본 CSV |
| [`02_executable_and_model/`](02_executable_and_model/) | **실행 파일 + 데이터 + 모델** — Streamlit 앱(`app.py`), 전처리/학습 데이터, LSTM 모델, 실행법(README) |
| [`03_docs/`](03_docs/) | **문서** — 발표 자료(PPT), Technical Report(PDF, 서비스 설계·한계), 시연 영상(mp4) |

---

## 🚀 빠른 실행

```bash
cd 02_executable_and_model
pip install -r requirements.txt   # konlpy는 Java(JDK) 필요
streamlit run app.py
```
자세한 실행 방법: [`02_executable_and_model/README.md`](02_executable_and_model/README.md)

---

## 🧠 사용 모델 — LSTM 감성분석
- 3분류(부정·중립·긍정), `max_len=100`
- **테스트셋 정확도 약 62.6%**, macro-F1 0.618
- 클래스별 F1: 부정 0.71 · 긍정 0.68 · 중립 0.47
- 평가 상세: `02_executable_and_model/eval/aladin_eval.json`

---

## 🔧 데이터 합치기 (선택)
GitHub 용량 제한 때문에 큰 CSV는 25MB 단위로 분할되어 있습니다.
원본 단일 파일로 합치려면:

```python
import pandas as pd, glob
parts = sorted(glob.glob("02_executable_and_model/data/filtered/*.csv"))
df = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in parts], ignore_index=True)
df.to_csv("aladin_reviews_filtered.csv", index=False, encoding="utf-8-sig")
print(len(df), "행")
```
(앱은 분할 파일을 자동으로 병합 로딩하므로, 합치지 않아도 실행됩니다.)

---

## 📊 주요 기능
- 🔍 **책 검색**: 제목 일부 검색 → 평점·종합감성·리뷰총평·단어빈도
- 🏆 **책 순위**: 평점/리뷰수 기준 랭킹
- 🎲 **랜덤 리뷰 감성 판단**: LSTM이 리뷰 내용으로 긍/중/부 판단 → 별점과 비교

## ⚠️ 한계 (자세한 내용은 Technical Report)
긍정 편향(92%)·평점과 내용 불일치·중립 모호성으로 정확도에 한계가 있으며,
양방향 LSTM/KoBERT, 다중 서점 통합, 클라우드 배포는 향후 과제로 남겨두었습니다.
