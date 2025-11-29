"""간단한 RAG 파이프라인 CLI.

지침에 따라 각 단계의 타임스탬프와 소요 시간을 기록하고,
폴더 내 텍스트 파일을 재귀적으로 처리한다.
"""

import argparse
import json
import importlib.util
import math
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# 전역 설정값
MODEL_NAME = "MongoDB/mdbr-leaf-ir"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}
DEFAULT_SIMILARITY_THRESHOLD = 0.3


class StepTimer:
    """단계별 시작/종료 시각과 소요 시간을 기록하는 컨텍스트 매니저."""

    def __init__(self, label: str):
        self.label = label
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        now = datetime.now().isoformat(timespec="seconds")
        print(f"[STEP] {self.label} 시작 | {now}")
        return self

    def __exit__(self, exc_type, exc, tb):
        end = time.perf_counter()
        now = datetime.now().isoformat(timespec="seconds")
        elapsed = end - self.start
        print(f"[STEP] {self.label} 종료 | {now} | 소요 {elapsed:.2f}s")


def collect_text_files(root_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in root_dir.rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            files.append(path)
    return sorted(files)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    step = max(1, chunk_size - max(0, chunk_overlap))
    chunks: List[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks


def load_embedding_model(model_name: str):
    if importlib.util.find_spec("sentence_transformers") is None:
        raise ImportError("sentence_transformers 패키지를 설치해야 합니다. (pip install sentence-transformers)")

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(model, texts: Sequence[str]) -> List[List[float]]:
    if not texts:
        return []
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return [vec.tolist() for vec in vectors]


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def cluster_embeddings(embeddings: List[List[float]], threshold: float) -> List[List[int]]:
    clusters: List[List[int]] = []
    representatives: List[List[float]] = []

    for idx, emb in enumerate(embeddings):
        placed = False
        for rep_idx, rep in enumerate(representatives):
            if cosine_similarity(rep, emb) >= threshold:
                clusters[rep_idx].append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
            representatives.append(emb)
    return clusters


def run_pipeline(input_dir: Path, output_file: Path, similarity_threshold: float, chunk_size: int, chunk_overlap: int) -> None:
    with StepTimer("모델 로드"):
        model = load_embedding_model(MODEL_NAME)

    with StepTimer("파일 로딩"):
        files = collect_text_files(input_dir)
        texts = {path: read_text(path) for path in files}

    with StepTimer("청킹"):
        chunk_records: List[Dict[str, str]] = []
        for path, text in texts.items():
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            for idx, chunk in enumerate(chunks):
                chunk_records.append({"file": str(path), "chunk_index": idx, "text": chunk})

    with StepTimer("임베딩"):
        texts_for_embedding = [record["text"] for record in chunk_records]
        embeddings = embed_texts(model, texts_for_embedding)

    with StepTimer("클러스터링"):
        clusters = cluster_embeddings(embeddings, similarity_threshold)

    with StepTimer("출력"):
        output = {
            "model": MODEL_NAME,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "similarity_threshold": similarity_threshold,
            "clusters": [
                {
                    "members": [
                        {
                            "file": chunk_records[idx]["file"],
                            "chunk_index": chunk_records[idx]["chunk_index"],
                            "text_preview": chunk_records[idx]["text"][:200],
                        }
                        for idx in cluster
                    ]
                }
                for cluster in clusters
            ],
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RESULT] {output_file} 에 결과를 저장했습니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="폴더 내 텍스트를 단순 RAG 파이프라인으로 처리하는 도구")
    parser.add_argument("--input-dir", required=True, type=Path, help="입력 텍스트 폴더 경로")
    parser.add_argument("--output-file", required=True, type=Path, help="결과 JSON 파일 경로")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="클러스터링 유사도 임계값 (0~1)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="청킹에 사용할 문자 단위 청크 크기",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="청크 간 겹치는 문자 수",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_dir=args.input_dir,
        output_file=args.output_file,
        similarity_threshold=args.similarity_threshold,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
