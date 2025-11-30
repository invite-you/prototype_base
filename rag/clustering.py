"""청크 클러스터링 및 결과 내보내기 유틸리티."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from rag.logging_utils import log_phase_context

# 전역 설정값
MODEL_NAME = "MongoDB/mdbr-leaf-ir"
DEFAULT_SIM_THRESHOLD = 0.75
DEFAULT_JSON_PATH = Path("cluster_result.json")
DEFAULT_MD_PATH = Path("cluster_result.md")
@dataclass
class ClusterMember:
    index: int
    text: str
    overlap_from: Optional[int]
    overlap_range: Optional[Tuple[int, int]]
    source_range: Optional[Tuple[int, int]]


@dataclass
class ClusterResult:
    cluster_id: int
    members: List[ClusterMember]


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _build_forbidden_pairs(chunks: Sequence) -> Set[Tuple[int, int]]:
    forbidden: Set[Tuple[int, int]] = set()
    for chunk in chunks:
        if getattr(chunk, "overlap_from", None) is None:
            continue
        prev_idx = getattr(chunk, "overlap_from")
        curr_idx = getattr(chunk, "index", None)
        if curr_idx is None:
            continue
        pair = tuple(sorted((prev_idx, curr_idx)))
        forbidden.add(pair)
    return forbidden


def _is_allowed(candidate: int, cluster_members: Iterable[int], forbidden_pairs: Set[Tuple[int, int]]) -> bool:
    for member in cluster_members:
        if tuple(sorted((candidate, member))) in forbidden_pairs:
            return False
    return True


def cluster_chunks(chunks: Sequence, embeddings: Sequence[Sequence[float]], sim_threshold: float = DEFAULT_SIM_THRESHOLD) -> List[ClusterResult]:
    """코사인 유사도 기반 그래프/그리디 방식 클러스터링.

    금지된 오버랩 짝을 동일 클러스터에 배치하지 않는다.
    """

    if not chunks or not embeddings:
        return []

    if len(chunks) != len(embeddings):
        raise ValueError("chunks와 embeddings 길이가 일치해야 합니다.")

    forbidden_pairs = _build_forbidden_pairs(chunks)
    visited: Set[int] = set()
    clusters: List[ClusterResult] = []

    for start_idx in range(len(chunks)):
        if start_idx in visited:
            continue

        queue = [start_idx]
        visited.add(start_idx)
        members: List[int] = []

        while queue:
            current = queue.pop()
            members.append(current)
            for neighbor in range(len(chunks)):
                if neighbor in visited:
                    continue
                if not _is_allowed(neighbor, members, forbidden_pairs):
                    continue
                sim = _cosine_similarity(embeddings[current], embeddings[neighbor])
                if sim >= sim_threshold:
                    visited.add(neighbor)
                    queue.append(neighbor)

        cluster_members = [
            ClusterMember(
                index=chunks[idx].index if hasattr(chunks[idx], "index") else idx,
                text=getattr(chunks[idx], "text", ""),
                overlap_from=getattr(chunks[idx], "overlap_from", None),
                overlap_range=getattr(chunks[idx], "overlap_range", None),
                source_range=getattr(chunks[idx], "source_range", None),
            )
            for idx in members
        ]
        clusters.append(ClusterResult(cluster_id=len(clusters), members=cluster_members))

    return clusters


def load_embedding_model(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import model_info

    with log_phase_context("임베딩 모델 로드"):
        model = SentenceTransformer(model_name)
        try:
            info = model_info(model_name)
            revision = getattr(info, "sha", None) or getattr(info, "revision", "unknown")
            print(f"[MODEL] 로드 완료 | id={model_name} | revision={revision}")
        except Exception as exc:  # noqa: BLE001
            print(f"[MODEL] 로드 완료 | id={model_name} | revision 조회 실패: {exc}")
        print(f"[MODEL] 사용 경로: {model_name}")
    return model


def embed_chunks(model, chunks: Sequence) -> List[List[float]]:
    texts = [getattr(chunk, "text", "") for chunk in chunks]
    with log_phase_context("청크 임베딩 계산"):
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return [vec.tolist() for vec in vectors]


def export_clusters(
    clusters: Sequence[ClusterResult],
    json_path: Path = DEFAULT_JSON_PATH,
    md_path: Path = DEFAULT_MD_PATH,
    model_name: str = MODEL_NAME,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
) -> None:
    json_payload = {
        "model": model_name,
        "similarity_threshold": sim_threshold,
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "members": [
                    {
                        "index": member.index,
                        "overlap_from": member.overlap_from,
                        "overlap_range": member.overlap_range,
                        "source_range": member.source_range,
                        "text": member.text,
                    }
                    for member in cluster.members
                ],
            }
            for cluster in clusters
        ],
    }

    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RESULT] JSON 클러스터 결과를 {json_path} 에 저장했습니다.")

    md_lines = [f"# 클러스터 결과 (모델: {model_name}, 임계값: {sim_threshold})", ""]
    for cluster in clusters:
        md_lines.append(f"## 클러스터 {cluster.cluster_id}")
        for member in cluster.members:
            overlap_note = (
                f" (overlap from {member.overlap_from} {member.overlap_range})" if member.overlap_from is not None else ""
            )
            source_note = f" {member.source_range}" if member.source_range else ""
            md_lines.append(f"- 청크 {member.index}{overlap_note}{source_note}")
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(member.text)
            md_lines.append("```")
            md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[RESULT] Markdown 클러스터 결과를 {md_path} 에 저장했습니다.")
