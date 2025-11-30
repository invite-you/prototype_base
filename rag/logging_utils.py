"""단계별 로그 포맷을 통일하는 유틸리티."""

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator


def _iso_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def log_phase_start(phase_name: str, start_time: float) -> None:
    """단계 시작 시각을 ISO 포맷으로 기록한다."""

    print(f"[PHASE] {phase_name} 시작 | {_iso_timestamp(start_time)}")


def log_phase(phase_name: str, start_time: float, end_time: float) -> None:
    """단계 시작/종료 시각과 소요 시간을 기록한다."""

    elapsed = end_time - start_time
    print(
        f"[PHASE] {phase_name} 종료 | 시작: {_iso_timestamp(start_time)} | 종료: {_iso_timestamp(end_time)} | 소요 {elapsed:.2f}s"
    )


def log_phase_end(phase_name: str, start_time: float, end_time: float) -> None:
    """단계 종료 로그를 남긴다."""

    log_phase(phase_name, start_time, end_time)


@contextmanager
def log_phase_context(phase_name: str) -> Iterator[None]:
    """with 블록으로 간단하게 단계 로그를 남기는 컨텍스트 매니저."""

    start_time = time.time()
    log_phase_start(phase_name, start_time)
    try:
        yield
    finally:
        end_time = time.time()
        log_phase_end(phase_name, start_time, end_time)
