from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(env_name: str, default: str) -> Path:
    value = Path(os.getenv(env_name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Settings:
    data_dir: Path = _project_path("UNIGUIDE_DATA_DIR", "data")
    db_path: Path = _project_path("UNIGUIDE_DB_PATH", "storage/uniguide.db")
    embedding_model: str = os.getenv(
        "UNIGUIDE_EMBEDDING_MODEL", "qwen3-embedding-0.6b"
    )
    chat_model: str = os.getenv("UNIGUIDE_CHAT_MODEL", "phi-4-mini")
    top_k: int = int(os.getenv("UNIGUIDE_TOP_K", "3"))
    min_similarity: float = float(os.getenv("UNIGUIDE_MIN_SIMILARITY", "0.35"))
    chunk_size: int = int(os.getenv("UNIGUIDE_CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("UNIGUIDE_CHUNK_OVERLAP", "180"))

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
