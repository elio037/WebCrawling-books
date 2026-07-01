# 02. 실행 파일 + 데이터 + 모델 (Executable + Data + Model)

알라딘 도서 리뷰 분석 **Streamlit 웹앱**을 바로 실행할 수 있는 폴더입니다.
클론(다운로드)한 뒤 아래 순서대로 하면 됩니다. (모든 경로는 상대경로라 그대로 동작)

## 구성
```
02_executable_and_model/
├── app.py                  # 실행 파일 (Streamlit 앱)
├── requirements.txt        # 파이썬 패키지 목록
├── data/
│   ├── filtered/           # 전처리 완료 데이터 (앱이 사용, 25MB 단위 26조각)
│   └── ing/                # 학습용 데이터 (라벨·토큰화 완료, 9조각)
├── model/
│   ├── sa_model_aladin.keras    # 내가 학습시킨 LSTM 감성분석 모델
│   └── sa_tokenizer_aladin.pkl  # 토크나이저
└── eval/
    ├── _train_log.json     # 학습 곡선(epoch별 정확도/loss)
    └── aladin_eval.json    # 테스트셋 평가 결과(정확도/혼동행렬/리포트)
```

## 사전 준비 — Java(JDK) 설치
형태소 분석기 **KoNLPy(Okt)**가 Java를 필요로 합니다. 먼저 JDK를 설치하세요.
- Windows: [Temurin JDK 17](https://adoptium.net/) 등 설치 후 환경변수 `JAVA_HOME` 설정
- 설치 확인: `java -version`

## 실행 방법

### 1) 가상환경 생성 & 패키지 설치
```bash
# (권장) conda
conda create -n aladin python=3.10 -y
conda activate aladin
pip install -r requirements.txt

# 또는 venv
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2) 앱 실행
```bash
streamlit run app.py
```
브라우저에서 자동으로 `http://localhost:8501` 이 열립니다.

> **데이터 로딩**: 앱이 처음 켜질 때 `data/filtered/`의 26개 조각 csv를 자동으로 병합해 약 100만 건을 메모리에 올립니다(최초 1회 수십 초 소요, 이후 캐시).

## 기능
| 탭 | 설명 |
|----|------|
| 🔍 책 검색 | 제목 일부만 입력해도 검색 → 평점·종합감성·리뷰총평·단어빈도 |
| 🏆 책 순위 | 평점 높은순 / 낮은순 / 리뷰 많은순 랭킹 |
| 🎲 랜덤 리뷰 감성 판단 | 리뷰를 랜덤으로 뽑아 **LSTM 모델**이 긍/중/부 판단 → 별점 기준과 비교 |

## 모델 성능 (LSTM, 테스트셋 기준)
- **정확도 약 62.6%** / macro-F1 0.618 (3분류: 부정·중립·긍정)
- 클래스별 F1 — 부정 0.71 · 긍정 0.68 · **중립 0.47(약점)**
- 상세 수치는 `eval/aladin_eval.json` 참고

## 참고
- 검색/순위 탭은 가볍게 동작하고, **랜덤 감성 판단 탭에서만** TensorFlow·KoNLPy가 사용됩니다.
- 원본 단일 데이터(`aladin_reviews_filtered.csv`, 554MB)는 GitHub 용량 제한으로 25MB 단위 분할본으로 제공됩니다. 합치는 방법은 루트 README의 "데이터 합치기" 참고.
