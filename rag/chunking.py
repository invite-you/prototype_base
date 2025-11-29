"""문단 기반 청킹 유틸리티."""

from dataclasses import dataclass
import re
from typing import List, Optional, Tuple

# 전역 설정값
MIN_CHUNK_SIZE = 1000
MAX_CHUNK_SIZE = 1500
OVERLAP_TARGET = 250
MIN_OVERLAP = 200
MAX_OVERLAP = 300
BLOCK_SEPARATOR = "\n\n"


@dataclass
class Chunk:
    """텍스트 청크와 메타데이터."""

    index: int
    text: str
    file_path: Optional[str] = None
    overlap_from: Optional[int] = None
    overlap_range: Optional[Tuple[int, int]] = None
    source_range: Optional[Tuple[int, int]] = None


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    last = 0
    for match in re.finditer(r"\n\s*\n+", text):
        if match.start() > last:
            spans.append((last, match.start()))
        last = match.end()
    if last < len(text):
        spans.append((last, len(text)))
    if not spans and text:
        spans.append((0, len(text)))
    return spans


def _split_sentences(paragraph: str, base_offset: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for match in re.finditer(r"[.!?。？！]", paragraph):
        end = match.end()
        while end < len(paragraph) and paragraph[end].isspace():
            end += 1
        spans.append((base_offset + cursor, base_offset + end))
        cursor = end
    if cursor < len(paragraph):
        spans.append((base_offset + cursor, base_offset + len(paragraph)))
    return spans


def _split_blocks(text: str) -> List[Tuple[str, int, int]]:
    blocks: List[Tuple[str, int, int]] = []
    for start, end in _paragraph_spans(text):
        if start >= end:
            continue
        paragraph = text[start:end]
        if len(paragraph) > MAX_CHUNK_SIZE:
            for sent_start, sent_end in _split_sentences(paragraph, start):
                sentence = text[sent_start:sent_end]
                if sentence.strip():
                    blocks.append((sentence, sent_start, sent_end))
        else:
            blocks.append((paragraph, start, end))
    return blocks


def _joined_length(parts: List[str]) -> int:
    if not parts:
        return 0
    return sum(len(part) for part in parts) + (len(parts) - 1) * len(BLOCK_SEPARATOR)


def _candidate_length(parts: List[str], addition: str) -> int:
    if not parts:
        return len(addition)
    return _joined_length(parts) + len(BLOCK_SEPARATOR) + len(addition)


def _choose_overlap_length(length: int) -> int:
    if length <= MIN_OVERLAP:
        return length
    if length < OVERLAP_TARGET:
        return min(length, MAX_OVERLAP)
    return min(MAX_OVERLAP, max(MIN_OVERLAP, OVERLAP_TARGET))


def chunk_text(text: str) -> List[Chunk]:
    if not text:
        return []

    blocks = _split_blocks(text)
    if not blocks:
        return []

    chunks: List[Chunk] = []
    parts: List[str] = []
    chunk_start: Optional[int] = None
    chunk_end: Optional[int] = None
    pending_overlap_from: Optional[Chunk] = None
    overlap_from: Optional[int] = None
    overlap_range: Optional[Tuple[int, int]] = None

    def finalize_current() -> None:
        nonlocal parts, chunk_start, chunk_end, overlap_from, overlap_range, pending_overlap_from
        if not parts:
            return
        chunk_text_value = BLOCK_SEPARATOR.join(parts)
        chunk = Chunk(
            index=len(chunks),
            text=chunk_text_value,
            file_path=None,
            overlap_from=overlap_from,
            overlap_range=overlap_range,
            source_range=(chunk_start or 0, chunk_end or 0),
        )
        chunks.append(chunk)
        pending_overlap_from = chunk
        parts = []
        chunk_start = None
        chunk_end = None
        overlap_from = None
        overlap_range = None

    for block_text, start, end in blocks:
        if not parts and pending_overlap_from:
            overlap_text = pending_overlap_from.text[-_choose_overlap_length(len(pending_overlap_from.text)) :]
            if overlap_text:
                parts.append(overlap_text)
                overlap_from = pending_overlap_from.index
                overlap_range = (
                    max(0, len(pending_overlap_from.text) - len(overlap_text)),
                    len(pending_overlap_from.text),
                )
                chunk_start = pending_overlap_from.source_range[1] - len(overlap_text) if pending_overlap_from.source_range else 0
                chunk_end = chunk_start + len(overlap_text)
            pending_overlap_from = None

        candidate_len = _candidate_length(parts, block_text)
        current_len = _joined_length(parts)
        if candidate_len > MAX_CHUNK_SIZE and current_len >= MIN_CHUNK_SIZE:
            finalize_current()
            # 다시 오버랩을 추가한 뒤 길이를 계산한다.
            if pending_overlap_from:
                overlap_text = pending_overlap_from.text[-_choose_overlap_length(len(pending_overlap_from.text)) :]
                if overlap_text:
                    parts.append(overlap_text)
                    overlap_from = pending_overlap_from.index
                    overlap_range = (
                        max(0, len(pending_overlap_from.text) - len(overlap_text)),
                        len(pending_overlap_from.text),
                    )
                    chunk_start = pending_overlap_from.source_range[1] - len(overlap_text) if pending_overlap_from.source_range else 0
                    chunk_end = chunk_start + len(overlap_text)
                pending_overlap_from = None
            candidate_len = _candidate_length(parts, block_text)

        parts.append(block_text)
        if chunk_start is None:
            chunk_start = start
        chunk_end = end

        if candidate_len >= MIN_CHUNK_SIZE:
            finalize_current()

    if parts:
        finalize_current()

    return chunks
