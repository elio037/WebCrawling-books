# -*- coding: utf-8 -*-
##########################################################################
# 알라딘 도서 리뷰 분석 웹앱 (Streamlit)
#
# 기능
#  1) 책 검색 : 제목 일부만 입력해도 관련 책 검색
#  2) 책 상세 : 평점·종합 감성 + "리뷰 총평"(요약) + 단어 빈도수
#  3) 책 순위 : 평점 높은순 · 낮은순 · 리뷰 많은순
#  4) 랜덤 리뷰 감성 판단 : 리뷰를 랜덤으로 뽑아 LSTM 모델이 긍/중/부 판단
#
# 데이터 : 전처리 완료 데이터(data/filtered/*.csv, 무관 리뷰 제거됨, 약 100만 건)
# 모델   : LSTM 3분류 감성분석 (model/sa_model_aladin.keras)
# 감성   : 평점 기준 (1~2 부정 · 3 중립 · 4~5 긍정)
#
# ※ 모든 경로는 이 파일 위치 기준 상대경로 → 클론 후 바로 실행 가능
##########################################################################

import os
import glob
import random
from collections import Counter

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from konlpy.tag import Okt

# ── 경로 (이 파일 기준 상대경로) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "filtered")          # 분할된 전처리 csv 폴더
MODEL_PATH = os.path.join(BASE_DIR, "model", "sa_model_aladin.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "sa_tokenizer_aladin.pkl")
MAX_LEN = 100

# ── 한글 폰트 (Windows 기본 맑은 고딕, 없으면 무시) ──
FONT_PATH = "c:/Windows/Fonts/malgun.ttf"
try:
    if os.path.exists(FONT_PATH):
        rc("font", family=font_manager.FontProperties(fname=FONT_PATH).get_name())
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

SENTI_COLOR = {"긍정": "#2E7D32", "중립": "#C8902F", "부정": "#C0392B"}
SENTI_BG = {"긍정": "#EAF3EA", "중립": "#FBF3E2", "부정": "#FBEDEB"}


def _rating_to_sentiment(r: int) -> str:
    if r <= 2:
        return "부정"
    if r == 3:
        return "중립"
    return "긍정"


@st.cache_data(show_spinner="데이터를 불러오는 중입니다... (최초 1회, 분할 파일 병합)")
def load_data(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"데이터 조각을 찾을 수 없습니다: {data_dir}")
    frames = [
        pd.read_csv(f, encoding="utf-8-sig",
                    usecols=lambda c: c in ["book_title", "review_content", "rating"])
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["book_title", "review_content", "rating"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].between(1, 5)].copy()
    df["rating"] = df["rating"].astype(int)
    df["book_title"] = df["book_title"].astype(str).str.strip()
    df["review_content"] = df["review_content"].astype(str)
    df["sentiment"] = df["rating"].map(_rating_to_sentiment)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_book_stats(data_dir: str) -> pd.DataFrame:
    df = load_data(data_dir)
    g = (df.groupby("book_title")
         .agg(리뷰수=("rating", "size"), 평균평점=("rating", "mean"))
         .reset_index())
    g["평균평점"] = g["평균평점"].round(2)
    return g


@st.cache_resource(show_spinner=False)
def get_okt() -> Okt:
    return Okt()


@st.cache_resource(show_spinner="감성분석 모델 로딩 중...")
def load_sentiment_model():
    from tensorflow.keras.models import load_model
    import joblib
    return load_model(MODEL_PATH), joblib.load(TOKENIZER_PATH)


def judge_text(text: str):
    """LSTM 모델로 리뷰 텍스트의 감성을 판단 (긍/중/부, 확신도)."""
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    model, tokenizer = load_sentiment_model()
    okt = get_okt()
    seq = tokenizer.texts_to_sequences([okt.morphs(str(text))])
    X = pad_sequences(seq, maxlen=MAX_LEN)
    p = model.predict(X, verbose=0)[0]
    labels = ["부정", "중립", "긍정"]
    i = int(np.argmax(p))
    probs = {labels[j]: float(p[j]) for j in range(3)}
    return labels[i], float(p[i]), probs


@st.cache_data(show_spinner=False)
def get_word_freq(data_dir: str, title: str, top_n: int = 25, max_reviews: int = 1500):
    df = load_data(data_dir)
    reviews = df.loc[df["book_title"] == title, "review_content"].head(max_reviews).tolist()
    okt = get_okt()
    tags = {"Noun", "Verb", "Adjective"}
    stop = {"책", "정말", "진짜", "이", "그", "수", "것", "너무", "좀", "잘", "때",
            "더", "및", "등", "를", "을", "은", "는", "이다", "있다", "없다", "되다",
            "하다", "보다", "같다", "내용", "저"}
    counter = Counter()
    for text in reviews:
        for word, pos in okt.pos(str(text)):
            if pos in tags and len(word) > 1 and word not in stop:
                counter[word] += 1
    return counter.most_common(top_n)


def _sample_reviews(sub: pd.DataFrame, sentiment: str, k: int = 2):
    s = sub[sub["sentiment"] == sentiment]
    fit = s[s["review_content"].str.len().between(15, 160)]
    s = fit if len(fit) else s
    if len(s) == 0:
        return []
    return s.sample(min(k, len(s)), random_state=42)["review_content"].tolist()


# ════════════════════════════════════════════════════════════════════════
# 책 상세 : 총평(요약) + 단어 빈도수
# ════════════════════════════════════════════════════════════════════════
def render_book_detail(data_dir: str, title: str, show_wordfreq: bool = True):
    df = load_data(data_dir)
    sub = df[df["book_title"] == title]
    total = len(sub)
    if total == 0:
        st.warning("해당 책의 리뷰가 없습니다.")
        return

    pos = int((sub["sentiment"] == "긍정").sum())
    neu = int((sub["sentiment"] == "중립").sum())
    neg = int((sub["sentiment"] == "부정").sum())
    pp, up, gp = pos / total * 100, neu / total * 100, neg / total * 100
    verdict = max([("긍정", pos), ("중립", neu), ("부정", neg)], key=lambda x: x[1])[0]
    avg = sub["rating"].mean()

    st.subheader(title)
    c1, c2, c3 = st.columns(3)
    c1.metric("리뷰 수", f"{total:,} 개")
    c2.metric("평균 평점", f"{avg:.2f} / 5")
    c3.markdown(
        f"**종합 감성**<br><span style='font-size:28px;font-weight:700;"
        f"color:{SENTI_COLOR[verdict]}'>{verdict} 경향</span>",
        unsafe_allow_html=True)

    st.markdown(
        f"<div style='display:flex;height:26px;border-radius:6px;overflow:hidden;"
        f"font-size:12px;color:white;text-align:center;line-height:26px'>"
        f"<div style='width:{pp:.1f}%;background:{SENTI_COLOR['긍정']}'>긍정 {pp:.0f}%</div>"
        f"<div style='width:{up:.1f}%;background:{SENTI_COLOR['중립']}'>중립 {up:.0f}%</div>"
        f"<div style='width:{gp:.1f}%;background:{SENTI_COLOR['부정']}'>부정 {gp:.0f}%</div>"
        f"</div>", unsafe_allow_html=True)

    st.divider()

    # ── 리뷰 총평 (요약) ──
    st.markdown("#### 리뷰 총평")
    if pp >= 70:
        tone = "대체로 호평받는 책입니다."
    elif pp >= 50:
        tone = "긍정적인 평가가 우세한 책입니다."
    elif gp >= 40:
        tone = "혹평이 적지 않은 책입니다."
    else:
        tone = "평가가 다소 갈리는 책입니다."
    st.markdown(
        f"<div style='background:#F4F6F8;border-radius:8px;padding:14px 18px;font-size:15px;line-height:1.6'>"
        f"리뷰 <b>{total:,}개</b>를 분석한 결과, 긍정 <b style='color:{SENTI_COLOR['긍정']}'>{pp:.0f}%</b> · "
        f"중립 <b style='color:{SENTI_COLOR['중립']}'>{up:.0f}%</b> · "
        f"부정 <b style='color:{SENTI_COLOR['부정']}'>{gp:.0f}%</b>로, 평균 평점은 <b>{avg:.2f}점</b>입니다.<br>"
        f"→ <b>{tone}</b></div>", unsafe_allow_html=True)

    # 대표 리뷰
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"<b style='color:{SENTI_COLOR['긍정']}'>👍 대표 긍정 리뷰</b>", unsafe_allow_html=True)
        samples = _sample_reviews(sub, "긍정", 2)
        if samples:
            for r in samples:
                st.markdown(
                    f"<div style='background:{SENTI_BG['긍정']};border-radius:8px;padding:10px 14px;"
                    f"margin:6px 0;font-size:14px'>“ {r} ”</div>", unsafe_allow_html=True)
        else:
            st.caption("긍정 리뷰 없음")
    with cols[1]:
        st.markdown(f"<b style='color:{SENTI_COLOR['부정']}'>👎 대표 부정 리뷰</b>", unsafe_allow_html=True)
        samples = _sample_reviews(sub, "부정", 2)
        if samples:
            for r in samples:
                st.markdown(
                    f"<div style='background:{SENTI_BG['부정']};border-radius:8px;padding:10px 14px;"
                    f"margin:6px 0;font-size:14px'>“ {r} ”</div>", unsafe_allow_html=True)
        else:
            st.caption("부정 리뷰 없음")

    if not show_wordfreq:
        return
    st.divider()

    # ── 단어 빈도수 (막대그래프) ──
    st.markdown("#### 단어 빈도수")
    if st.checkbox("이 책 리뷰의 자주 나온 단어 분석", key=f"wf_{title}"):
        top_n = st.slider("표시할 단어 수", 10, 50, 25, key=f"tn_{title}")
        with st.spinner("형태소 분석 중..."):
            freq = get_word_freq(data_dir, title, top_n)
        if not freq:
            st.info("분석할 단어가 부족합니다.")
            return
        words = [w for w, _ in freq][::-1]
        counts = [c for _, c in freq][::-1]
        fig, ax = plt.subplots(figsize=(5, max(2.5, top_n * 0.18)))
        ax.barh(words, counts, color="#4A90D9")
        ax.set_xlabel("빈도수", fontsize=9)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        col_chart, _ = st.columns([1, 1])
        with col_chart:
            st.pyplot(fig, use_container_width=False)


# ════════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="알라딘 도서 리뷰 분석", layout="wide")
    st.title("📚 알라딘 도서 리뷰 분석")

    if not glob.glob(os.path.join(DATA_DIR, "*.csv")):
        st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_DIR}\n"
                 f"data/filtered/ 폴더에 분할 csv가 있는지 확인하세요.")
        st.stop()
    st.caption("데이터: 알라딘 도서 리뷰 약 100만 건 (전처리 완료) · 감성 모델: LSTM(정확도 약 62%)")

    stats = get_book_stats(DATA_DIR)
    tab_search, tab_rank, tab_rand = st.tabs(["🔍 책 검색", "🏆 책 순위", "🎲 랜덤 리뷰 감성 판단"])

    # ── 책 검색 ──
    with tab_search:
        query = st.text_input("책 제목 검색", placeholder="제목 일부만 입력해도 됩니다 (예: 사)")
        if query.strip():
            matches = stats[stats["book_title"].str.contains(
                query.strip(), case=False, na=False, regex=False)].sort_values("리뷰수", ascending=False)
            if matches.empty:
                st.info("검색 결과가 없습니다.")
            else:
                st.success(f"'{query.strip()}' 검색 결과: {len(matches):,}권")
                options = matches["book_title"].head(50).tolist()
                labels = {t: f"{t}  (리뷰 {int(r):,} · 평점 {p})"
                          for t, r, p in zip(matches["book_title"].head(50),
                                             matches["리뷰수"].head(50), matches["평균평점"].head(50))}
                selected = st.selectbox("책 선택", options, format_func=lambda t: labels.get(t, t))
                st.divider()
                render_book_detail(DATA_DIR, selected)
        else:
            st.info("위 검색창에 책 제목의 일부를 입력하세요.")

    # ── 책 순위 ──
    with tab_rank:
        c1, c2 = st.columns([2, 1])
        sort_by = c1.radio("정렬 기준", ["평점 높은순", "평점 낮은순", "리뷰 많은순"], horizontal=True)
        min_reviews = c2.slider("최소 리뷰 수", 1, 200, 10)
        ranked = stats[stats["리뷰수"] >= min_reviews].copy()
        if sort_by == "평점 높은순":
            ranked = ranked.sort_values(["평균평점", "리뷰수"], ascending=[False, False])
        elif sort_by == "평점 낮은순":
            ranked = ranked.sort_values(["평균평점", "리뷰수"], ascending=[True, False])
        else:
            ranked = ranked.sort_values("리뷰수", ascending=False)
        st.caption(f"리뷰 {min_reviews}개 이상인 책 {len(ranked):,}권 중 상위 100권")
        top = ranked.head(100).reset_index(drop=True)
        top.index = top.index + 1
        st.dataframe(top.rename(columns={"book_title": "책 제목"}), use_container_width=True, height=420)
        st.divider()
        st.markdown("#### 순위에서 책 선택해 상세 보기")
        pick = st.selectbox("책 선택", top["book_title"].tolist())
        if pick:
            render_book_detail(DATA_DIR, pick, show_wordfreq=False)

    # ── 랜덤 리뷰 감성 판단 ──
    with tab_rand:
        st.markdown("리뷰를 랜덤으로 뽑아 **학습한 LSTM 모델이 내용을 읽고** 긍정·중립·부정을 판단합니다.")
        df = load_data(DATA_DIR)
        if st.button("🎲 랜덤 리뷰 뽑기", type="primary"):
            st.session_state["rand_idx"] = random.randint(0, len(df) - 1)

        if "rand_idx" in st.session_state:
            row = df.iloc[st.session_state["rand_idx"]]
            st.markdown(
                f"<div style='background:#F4F6F8;border-radius:8px;padding:16px 18px;font-size:16px;line-height:1.6'>"
                f"“ {row['review_content']} ”</div>", unsafe_allow_html=True)
            st.caption(f"📖 {row['book_title']}  ·  실제 별점 {row['rating']}점")

            with st.spinner("학습 모델이 리뷰 내용을 판단하는 중..."):
                try:
                    label, conf, probs = judge_text(row["review_content"])
                    ok = True
                except Exception as e:
                    st.error(f"모델 판단 오류: {e}")
                    ok = False

            if ok:
                rating_senti = _rating_to_sentiment(int(row["rating"]))
                cc1, cc2 = st.columns(2)
                cc1.markdown(
                    f"<div style='background:{SENTI_BG[label]};border-radius:10px;padding:18px;text-align:center'>"
                    f"<div style='font-size:13px;color:#555'>학습 모델 판단</div>"
                    f"<div style='font-size:34px;font-weight:800;color:{SENTI_COLOR[label]}'>{label}</div>"
                    f"<div style='font-size:13px;color:#555'>확신도 {conf*100:.0f}%</div></div>",
                    unsafe_allow_html=True)
                cc2.markdown(
                    f"<div style='background:#EEF1F4;border-radius:10px;padding:18px;text-align:center'>"
                    f"<div style='font-size:13px;color:#555'>별점 기준</div>"
                    f"<div style='font-size:34px;font-weight:800;color:{SENTI_COLOR[rating_senti]}'>{rating_senti}</div>"
                    f"<div style='font-size:13px;color:#555'>{row['rating']}점</div></div>",
                    unsafe_allow_html=True)

                # 긍정/중립/부정 3개 확률 표시
                gp, up, pp = probs["긍정"] * 100, probs["중립"] * 100, probs["부정"] * 100
                st.markdown(
                    f"<div style='margin-top:12px;font-size:13px;color:#555'>학습 모델 판단 확률</div>"
                    f"<div style='display:flex;height:24px;border-radius:6px;overflow:hidden;"
                    f"font-size:12px;color:white;text-align:center;line-height:24px'>"
                    f"<div style='width:{gp:.1f}%;background:{SENTI_COLOR['긍정']}'>긍정 {gp:.0f}%</div>"
                    f"<div style='width:{up:.1f}%;background:{SENTI_COLOR['중립']}'>중립 {up:.0f}%</div>"
                    f"<div style='width:{pp:.1f}%;background:{SENTI_COLOR['부정']}'>부정 {pp:.0f}%</div>"
                    f"</div>", unsafe_allow_html=True)

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                if label == rating_senti:
                    st.success("학습 모델 판단과 별점 기준이 일치합니다.")
                else:
                    st.warning("학습 모델 판단과 별점이 다릅니다 — 내용과 별점이 어긋나는 리뷰일 수 있어요.")
            st.caption("학습 모델(LSTM, 정확도 약 62%)이 리뷰 텍스트만 보고 내린 결과입니다.")
        else:
            st.info("위 버튼을 눌러 랜덤 리뷰를 뽑아보세요.")


if __name__ == "__main__":
    main()
