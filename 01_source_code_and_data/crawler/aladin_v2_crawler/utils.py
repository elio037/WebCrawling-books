"""유틸리티: 중복 차단, CSV 저장, RPS 추적, UA 회전, 백오프 요청"""
from __future__ import annotations

import asyncio
import csv
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Iterable

import httpx

from config import (
    BACKOFF_BASE,
    CSV_COLUMNS,
    MAX_RETRIES,
    OUTPUT_CSV,
    PROCESSED_FILE,
    RATE_LIMIT_CODES,
)

# ── User-Agent 회전 ───────────────────────────────────────────
try:
    from fake_useragent import UserAgent
    _ua_gen = UserAgent()
    def random_ua() -> str:
        return _ua_gen.random
except Exception:
    _UA_FALLBACK = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    ]
    def random_ua() -> str:
        return random.choice(_UA_FALLBACK)


def _default_headers() -> dict:
    return {
        "User-Agent": random_ua(),
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
        "Referer": "https://www.aladin.co.kr/",
    }


# ── 지수 백오프 HTTP 요청 ─────────────────────────────────────
async def get_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    params: dict | None = None,
) -> httpx.Response | None:
    """429/503 시 지수 백오프 재시도. 최종 실패 시 None 반환."""
    delay = BACKOFF_BASE
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(
                url,
                params=params,
                headers=_default_headers(),
                follow_redirects=True,
                timeout=15.0,
            )
            if resp.status_code in RATE_LIMIT_CODES:
                wait = delay * (2 ** attempt)
                print(f"  [백오프] {resp.status_code} → {wait:.1f}s 대기 (시도 {attempt+1})")
                await asyncio.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp
            return None
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            if attempt == MAX_RETRIES - 1:
                print(f"  [요청실패] {url[:60]} | {exc}")
                return None
            await asyncio.sleep(delay * (2 ** attempt))
    return None


# ── 도서 중복 차단: 메모리 Set + processed_books.txt ──────────
class ProcessedSet:
    """수집 완료한 book_id를 메모리(set)와 파일에 동시 기록.
    재시작 시 파일을 읽어 즉시 복원 → 중복 수집 원천 차단."""

    def __init__(self, filepath: str = PROCESSED_FILE):
        self._path = Path(filepath)
        self._seen: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            lines = self._path.read_text(encoding="utf-8").splitlines()
            self._seen = set(filter(None, lines))
            print(f"[ProcessedSet] 기존 {len(self._seen):,}권 로드 → 즉시 스킵 대상")

    def has(self, book_id: str) -> bool:
        return book_id in self._seen

    def add(self, book_id: str) -> None:
        if book_id not in self._seen:
            self._seen.add(book_id)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(book_id + "\n")

    @property
    def count(self) -> int:
        return len(self._seen)


# ── 리뷰 ID 중복 차단 (저장 직전 더블 체크) ──────────────────
class ReviewIdSet:
    """review_id 기준 중복 체크. 기존 CSV에서 review_id를 로드."""

    def __init__(self, csv_path: str = OUTPUT_CSV):
        self._seen: set[str] = set()
        self._load_from_csv(csv_path)

    def _load_from_csv(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            with p.open(encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = row.get("review_id", "").strip()
                    if rid:
                        self._seen.add(rid)
            print(f"[ReviewIdSet] 기존 review_id {len(self._seen):,}개 로드")
        except Exception as exc:
            print(f"[ReviewIdSet] CSV 로드 실패: {exc}")

    def is_dup(self, review_id: str) -> bool:
        return bool(review_id) and review_id in self._seen

    def add(self, review_id: str) -> None:
        if review_id:
            self._seen.add(review_id)


# ── 실시간 CSV 저장 (Append 모드) ────────────────────────────
class CsvWriter:
    """도서 1권 완료 즉시 Append. 헤더는 파일 없을 때만 자동 생성."""

    def __init__(self, filepath: str = OUTPUT_CSV):
        self._path = Path(filepath)
        if not self._path.exists():
            with self._path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    def append(self, rows: Iterable[dict]) -> int:
        """rows를 파일에 추가. 실제 저장 건수 반환."""
        saved = 0
        with self._path.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            for row in rows:
                w.writerow(row)
                saved += 1
        return saved


# ── RPS 추적기 (초당 요청 수 실시간 계산) ────────────────────
class RpsTracker:
    def __init__(self, window: float = 10.0):
        self._window = window
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def tick(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._timestamps.append(now)
            cutoff = now - self._window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

    @property
    def rps(self) -> float:
        if not self._timestamps:
            return 0.0
        return len(self._timestamps) / self._window


# ── 상태 출력 ────────────────────────────────────────────────
def log_status(
    cat: str,
    page: int,
    total_reviews: int,
    skipped_books: int,
    rps: float,
) -> None:
    print(
        f"\r[{cat}] p{page:>4} | "
        f"리뷰 {total_reviews:>7,}건 | "
        f"스킵 {skipped_books:>5,}권 | "
        f"RPS {rps:>5.1f}",
        end="",
        flush=True,
    )
