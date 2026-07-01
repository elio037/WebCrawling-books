"""전역 설정 상수 — 여기만 수정하면 전체 동작이 바뀐다."""

# ── 기본 도메인 ──────────────────────────────────────────────
BASE_URL = "https://www.aladin.co.kr"

# ── 도서 목록 수집 URL (베스트셀러 아닌 전체 카테고리 브라우즈) ──
# SortOrder=6: 판매량순 / ViewRowsCount=50: 페이지당 50권
BROWSE_URL = (
    "https://www.aladin.co.kr/shop/wbrowse.aspx"
    "?BrowseTarget=List&ViewRowsCount=50&ViewType=Detail"
    "&SortOrder=6&Edu=0&PublishMonth=0&SearchKeyword="
    "&CID={cid}&page={page}"
)

# ── 리뷰 AJAX API 후보 (우선순위 순) ──────────────────────────
# 실행 시 probe()가 자동으로 작동하는 엔드포인트를 결정한다.
COMMENT_API_CANDIDATES = [
    # 100자평 (한줄평) — Aladin 내부 XHR 패턴
    BASE_URL + "/winside/wcomment.aspx?ItemId={item_id}&type=1&page={page}&maxResult=50",
    BASE_URL + "/shop/product/getCommentList.aspx?ItemId={item_id}&page={page}&maxResult=50",
    BASE_URL + "/shop/getReview.aspx?ItemId={item_id}&reviewType=comment&page={page}",
]

MYREVIEW_API_CANDIDATES = [
    BASE_URL + "/winside/wmyreview.aspx?ItemId={item_id}&page={page}&maxResult=10",
    BASE_URL + "/shop/product/getMyReviewList.aspx?ItemId={item_id}&page={page}&maxResult=10",
]

# ── 폴백: 도서 상세 페이지 직접 파싱 ────────────────────────
PRODUCT_URL = BASE_URL + "/shop/wproduct.aspx?ItemId={item_id}"

# ── 카테고리 CID (전체 브라우즈용) ──────────────────────────
CATEGORIES: dict[str, int] = {
    "소설/시/희곡":  1,
    "경제경영":      2,
    "자기계발":      3,
    "인문학":        4,
    "역사":          5,
    "사회과학":      6,
    "과학":          7,
    "예술/대중문화": 8,
    "만화":          9,
    "건강/취미":    10,
    "외국어":       11,
    "어린이":       13,
    "청소년":       20,
    "컴퓨터/IT":    21,
    "종교/역학":    26,
    "잡지":         50,
}

# ── 성능 설정 ────────────────────────────────────────────────
CONCURRENCY      = 20    # 동시 리뷰 수집 작업 수 (httpx는 브라우저보다 10x 빠름)
BROWSE_CONCUR    = 5     # 목록 페이지 동시 요청 수
MAX_BROWSE_PAGES = 200   # 카테고리당 최대 브라우즈 페이지 (200×50 = 10,000권)
MAX_REVIEW_PAGES = 20    # 도서당 최대 리뷰 페이지 (20×50 = 1,000건)

# ── 재시도 / 백오프 ──────────────────────────────────────────
MAX_RETRIES      = 4
BACKOFF_BASE     = 1.5   # 초: 1.5 → 3 → 6 → 12초 대기
RATE_LIMIT_CODES = {429, 503, 502}

# ── 출력 파일 ────────────────────────────────────────────────
OUTPUT_CSV       = "aladin_clean_reviews.csv"
PROCESSED_FILE   = "processed_books.txt"

# ── CSV 컬럼 ─────────────────────────────────────────────────
CSV_COLUMNS = [
    "book_id",
    "book_title",
    "review_id",
    "reviewer",
    "review_content",
    "rating",
    "created_at",
]
