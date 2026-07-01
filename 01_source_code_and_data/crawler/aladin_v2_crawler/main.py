"""
알라딘 고성능 비동기 리뷰 크롤러 v2
- 베스트셀러 탈피: 카테고리 전체 브라우즈 페이지 순회
- 완전한 중복 차단: ProcessedSet(book) + ReviewIdSet(review_id)
- 순수 httpx 비동기: Selenium 없음, 동시 요청 20개
- 지수 백오프: 429/503 자동 대기 후 재시도
- 실시간 CSV Append + RPS 로깅
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from config import (
    BASE_URL,
    BROWSE_CONCUR,
    BROWSE_URL,
    CATEGORIES,
    COMMENT_API_CANDIDATES,
    CONCURRENCY,
    CSV_COLUMNS,
    MAX_BROWSE_PAGES,
    MAX_REVIEW_PAGES,
    MYREVIEW_API_CANDIDATES,
    OUTPUT_CSV,
    PROCESSED_FILE,
    PRODUCT_URL,
)
from utils import (
    CsvWriter,
    ProcessedSet,
    ReviewIdSet,
    RpsTracker,
    get_with_backoff,
    log_status,
    random_ua,
)


# ══════════════════════════════════════════════════════════════
#  STEP 0: 스타트업 프로브 — 어느 API 엔드포인트가 작동하는지 탐지
# ══════════════════════════════════════════════════════════════
PROBE_ITEM_ID = "4342463"  # 1Q84 1권 (리뷰 많은 알려진 책)


async def probe_api(client: httpx.AsyncClient) -> tuple[str | None, str | None]:
    """작동하는 100자평 / 마이리뷰 API 엔드포인트를 반환."""
    print("[PROBE] 리뷰 API 엔드포인트 탐색 중...")

    comment_url = None
    for tpl in COMMENT_API_CANDIDATES:
        url = tpl.format(item_id=PROBE_ITEM_ID, page=1)
        resp = await get_with_backoff(client, url)
        if resp and len(resp.text) > 200:
            soup = BeautifulSoup(resp.text, "lxml")
            if soup.select("div.ritem, li.ritem, .review_cont"):
                comment_url = tpl
                print(f"  [OK] 100자평 API: {url[:70]}")
                break

    if not comment_url:
        # 폴백: 상세 페이지 직접 파싱
        comment_url = PRODUCT_URL
        print("  [폴백] 100자평 → 상세 페이지 직접 파싱")

    myreview_url = None
    for tpl in MYREVIEW_API_CANDIDATES:
        url = tpl.format(item_id=PROBE_ITEM_ID, page=1)
        resp = await get_with_backoff(client, url)
        if resp and len(resp.text) > 200:
            soup = BeautifulSoup(resp.text, "lxml")
            if soup.select("div.myreview_row, .review_row, .review_text"):
                myreview_url = tpl
                print(f"  [OK] 마이리뷰 API: {url[:70]}")
                break

    print()
    return comment_url, myreview_url


# ══════════════════════════════════════════════════════════════
#  STEP 1: 도서 목록 수집 — 카테고리 브라우즈 전체 페이지 순회
# ══════════════════════════════════════════════════════════════
def _extract_books_from_browse(html: str) -> list[dict]:
    """브라우즈 페이지 HTML에서 book_id, book_title, url 추출."""
    soup = BeautifulSoup(html, "lxml")
    books = []
    for a in soup.select("a.bo3"):
        href = a.get("href", "")
        m = re.search(r"ItemId=(\d+)", href)
        if not m:
            continue
        book_url = (BASE_URL + href) if href.startswith("/") else href
        books.append({
            "book_id":    m.group(1),
            "book_title": a.get_text(strip=True),
            "url":        book_url,
        })
    return books


async def collect_book_ids(
    client: httpx.AsyncClient,
    cid: int,
    cat_name: str,
    processed: ProcessedSet,
    max_pages: int = MAX_BROWSE_PAGES,
) -> list[dict]:
    """카테고리 전체 브라우즈 → 미수집 도서 목록 반환."""
    sem = asyncio.Semaphore(BROWSE_CONCUR)
    books_all: list[dict] = []

    async def fetch_page(page_no: int) -> list[dict]:
        async with sem:
            url = BROWSE_URL.format(cid=cid, page=page_no)
            resp = await get_with_backoff(client, url)
            if not resp:
                return []
            return _extract_books_from_browse(resp.text)

    # 1페이지 먼저 → 결과 없으면 카테고리 끝
    first = await fetch_page(1)
    if not first:
        return []
    books_all.extend(first)

    # 나머지 페이지 병렬 수집
    tasks = [fetch_page(p) for p in range(2, max_pages + 1)]
    for batch in await asyncio.gather(*tasks):
        if not batch:
            break
        books_all.extend(batch)

    # book_id 기준 중복 제거 (같은 책이 여러 페이지에 등장 가능)
    seen: set[str] = set()
    unique: list[dict] = []
    for b in books_all:
        if b["book_id"] not in seen:
            seen.add(b["book_id"])
            unique.append(b)

    # 이미 수집 완료한 도서 제외
    new_books = [b for b in unique if not processed.has(b["book_id"])]
    print(
        f"[{cat_name}] 총 {len(unique):,}권 발견 "
        f"→ 신규 {len(new_books):,}권 수집 대상 "
        f"(스킵 {len(unique)-len(new_books):,}권)"
    )
    return new_books


# ══════════════════════════════════════════════════════════════
#  STEP 2: 리뷰 파싱
# ══════════════════════════════════════════════════════════════
def _parse_reviews(
    html: str,
    book_id: str,
    title: str,
    review_ids: ReviewIdSet,
) -> list[dict]:
    """HTML 응답에서 리뷰 추출 + review_id 더블 체크."""
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []

    # 100자평 (div.ritem)
    for item in soup.select("div.ritem, li.ritem"):
        # 본문
        text_el = item.select_one(".review_cont, .txt_comment, span.review_cont")
        if not text_el:
            candidates = [s for s in item.find_all("span") if len(s.get_text(strip=True)) > 5]
            text_el = max(candidates, key=lambda s: len(s.get_text()), default=None)
        if not text_el:
            continue
        text = text_el.get_text(strip=True)
        if len(text) < 4:
            continue

        # review_id
        rid_m = re.search(r"commentNo=(\d+)|reviewNo=(\d+)|itemId=\d+.*?no=(\d+)", str(item))
        review_id = next((g for g in (rid_m.groups() if rid_m else ()) if g), "")

        # 더블 체크: review_id 중복
        if review_ids.is_dup(review_id):
            continue

        # 별점
        star_el = item.select_one("em[style*='width'], .star_score, [class*='star']")
        rating = ""
        if star_el:
            w_m = re.search(r"width:\s*(\d+)%", star_el.get("style", ""))
            rating = str(round(int(w_m.group(1)) / 20)) if w_m else ""

        # 작성일
        date_el = item.select_one(".date, .review_date, [class*='date']")
        date = date_el.get_text(strip=True) if date_el else ""

        # 작성자
        rev_el = item.select_one(".reviewer, .id, [class*='nick'], [class*='user']")
        reviewer = rev_el.get_text(strip=True) if rev_el else ""

        rows.append({
            "book_id":        book_id,
            "book_title":     title,
            "review_id":      review_id,
            "reviewer":       reviewer,
            "review_content": text,
            "rating":         rating,
            "created_at":     date,
        })
        review_ids.add(review_id)

    # 마이리뷰 (div.myreview_row)
    for item in soup.select("div.myreview_row, div.review_row"):
        content_el = item.select_one(".review_cont, .review_text, p.cont")
        if not content_el:
            continue
        text = content_el.get_text(strip=True)
        if len(text) < 4:
            continue

        rid_m = re.search(r"reviewNo=(\d+)|no=(\d+)", str(item))
        review_id = next((g for g in (rid_m.groups() if rid_m else ()) if g), "")
        if review_ids.is_dup(review_id):
            continue

        star_el = item.select_one("em[style*='width'], .star_score")
        rating = ""
        if star_el:
            w_m = re.search(r"width:\s*(\d+)%", star_el.get("style", ""))
            rating = str(round(int(w_m.group(1)) / 20)) if w_m else ""

        date_el = item.select_one(".review_date, .date")
        date = date_el.get_text(strip=True) if date_el else ""

        rev_el = item.select_one(".reviewer_id, .nick, [class*='nick']")
        reviewer = rev_el.get_text(strip=True) if rev_el else ""

        rows.append({
            "book_id":        book_id,
            "book_title":     title,
            "review_id":      review_id,
            "reviewer":       reviewer,
            "review_content": text,
            "rating":         rating,
            "created_at":     date,
        })
        review_ids.add(review_id)

    return rows


# ══════════════════════════════════════════════════════════════
#  STEP 3: 단일 도서 리뷰 수집
# ══════════════════════════════════════════════════════════════
async def fetch_book_reviews(
    client: httpx.AsyncClient,
    book: dict,
    comment_api_tpl: str,
    myreview_api_tpl: Optional[str],
    review_ids: ReviewIdSet,
) -> list[dict]:
    """한 도서의 모든 리뷰 페이지를 비동기로 수집."""
    item_id = book["book_id"]
    title   = book["book_title"]
    rows: list[dict] = []

    # ── 100자평 ────────────────────────────────────────────
    is_fallback = comment_api_tpl == PRODUCT_URL

    if is_fallback:
        # 상세 페이지 직접 파싱 (API 없을 때)
        url  = PRODUCT_URL.format(item_id=item_id)
        resp = await get_with_backoff(client, url)
        if resp:
            rows.extend(_parse_reviews(resp.text, item_id, title, review_ids))
    else:
        # 페이지별 순회
        for page in range(1, MAX_REVIEW_PAGES + 1):
            url  = comment_api_tpl.format(item_id=item_id, page=page)
            resp = await get_with_backoff(client, url)
            if not resp:
                break
            batch = _parse_reviews(resp.text, item_id, title, review_ids)
            rows.extend(batch)
            if len(batch) < 10:   # 마지막 페이지
                break

    # ── 마이리뷰 ──────────────────────────────────────────
    if myreview_api_tpl and not is_fallback:
        for page in range(1, MAX_REVIEW_PAGES + 1):
            url  = myreview_api_tpl.format(item_id=item_id, page=page)
            resp = await get_with_backoff(client, url)
            if not resp:
                break
            batch = _parse_reviews(resp.text, item_id, title, review_ids)
            rows.extend(batch)
            if len(batch) < 3:
                break

    return rows


# ══════════════════════════════════════════════════════════════
#  STEP 4: 워커 — 세마포어로 동시성 제어
# ══════════════════════════════════════════════════════════════
async def worker(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    book: dict,
    comment_api: str,
    myreview_api: Optional[str],
    processed: ProcessedSet,
    review_ids: ReviewIdSet,
    writer: CsvWriter,
    rps: RpsTracker,
    counters: dict,
) -> None:
    book_id = book["book_id"]

    if processed.has(book_id):
        counters["skipped"] += 1
        return

    async with sem:
        rows = await fetch_book_reviews(
            client, book, comment_api, myreview_api, review_ids
        )
        await rps.tick()

    if rows:
        saved = writer.append(rows)
        counters["reviews"] += saved

    processed.add(book_id)

    if counters["reviews"] % 500 == 0 and counters["reviews"] > 0:
        print(
            f"\n[체크포인트] 누적 {counters['reviews']:,}건 | "
            f"스킵 {counters['skipped']:,}권 | RPS {rps.rps:.1f}",
            flush=True,
        )


# ══════════════════════════════════════════════════════════════
#  메인 오케스트레이터
# ══════════════════════════════════════════════════════════════
async def run(args: argparse.Namespace) -> None:
    processed   = ProcessedSet(args.processed)
    review_ids  = ReviewIdSet(args.output)
    writer      = CsvWriter(args.output)
    rps_tracker = RpsTracker()
    sem         = asyncio.Semaphore(args.concurrency)
    counters    = {"reviews": 0, "skipped": 0}

    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)

    async with httpx.AsyncClient(
        limits=limits,
        timeout=httpx.Timeout(20.0),
        follow_redirects=True,
    ) as client:

        # ── API 엔드포인트 자동 탐지 ────────────────────────
        comment_api, myreview_api = await probe_api(client)

        # ── 카테고리 필터 ────────────────────────────────────
        cats = (
            {k: v for k, v in CATEGORIES.items() if k in args.categories}
            if args.categories else CATEGORIES
        )

        t_start = time.time()

        for cat_name, cid in cats.items():
            print(f"\n{'='*60}")
            print(f"  카테고리: {cat_name} (CID={cid})")
            print(f"{'='*60}")

            # 도서 목록 수집
            books = await collect_book_ids(
                client, cid, cat_name, processed, args.max_pages
            )
            if not books:
                print(f"  → 신규 도서 없음, 다음 카테고리로")
                continue

            # 병렬 리뷰 수집
            tasks = [
                worker(
                    sem, client, b,
                    comment_api, myreview_api,
                    processed, review_ids, writer, rps_tracker, counters,
                )
                for b in books
            ]

            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                await coro
                if i % 10 == 0:
                    log_status(
                        cat_name, i,
                        counters["reviews"],
                        counters["skipped"],
                        rps_tracker.rps,
                    )

            print()  # 줄바꿈

        elapsed = (time.time() - t_start) / 3600
        print(f"\n{'='*60}")
        print(f"[완료] 총 {counters['reviews']:,}건 리뷰")
        print(f"       스킵 {counters['skipped']:,}권 | 소요 {elapsed:.2f}시간")
        print(f"       저장 위치: {args.output}")
        print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
#  CLI 진입점
# ══════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="알라딘 고성능 비동기 리뷰 크롤러 v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--categories", nargs="*",
        help="수집 카테고리 (미지정 시 전체 16개). 예: 소설/시/희곡 경제경영",
    )
    ap.add_argument("--max-pages",   type=int, default=MAX_BROWSE_PAGES,
                    help="카테고리당 최대 브라우즈 페이지 수")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY,
                    help="동시 리뷰 수집 작업 수")
    ap.add_argument("--output",      default=OUTPUT_CSV,  help="출력 CSV 경로")
    ap.add_argument("--processed",   default=PROCESSED_FILE, help="처리 완료 book_id 파일")
    return ap


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
