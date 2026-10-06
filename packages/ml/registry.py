"""File-based model registry (README §13.3).

Layout: ``<root>/<model_name>/<model_version>/{model.pkl, metadata.json}``.
A model starts as ``PENDING_REVIEW``; only an explicit approval (human, with
the approver recorded) makes it usable by downstream steps.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "models" / "registry"
STATUSES = ("PENDING_REVIEW", "APPROVED", "REJECTED", "RETIRED")


@dataclass
class ModelMetadata:
    model_name: str
    model_version: str
    model_type: str
    training_data_version: str
    feature_schema_version: str
    feature_names: list[str]
    train_start_date: str
    train_end_date: str
    validation_dates: list[str]
    test_dates: list[str]
    metrics: dict[str, object]
    feature_importance: dict[str, float]
    hyperparameters: dict[str, object]
    git_commit: str
    approval_status: str = "PENDING_REVIEW"
    approved_by: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    deployed_at: str | None = None
    retired_at: str | None = None


class ModelRegistry:
    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def _dir(self, name: str, version: str) -> Path:
        return self.root / name / version

    def register(self, model, metadata: ModelMetadata) -> Path:
        path = self._dir(metadata.model_name, metadata.model_version)
        if path.exists():
            raise FileExistsError(
                f"{metadata.model_name} {metadata.model_version} already registered"
            )
        path.mkdir(parents=True)
        with (path / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        self._write(metadata)
        return path

    def _write(self, metadata: ModelMetadata) -> None:
        path = self._dir(metadata.model_name, metadata.model_version) / "metadata.json"
        path.write_text(json.dumps(asdict(metadata), indent=2, default=str))

    def metadata(self, name: str, version: str) -> ModelMetadata:
        data = json.loads((self._dir(name, version) / "metadata.json").read_text())
        return ModelMetadata(**data)

    def load(self, name: str, version: str, require_approved: bool = True):
        meta = self.metadata(name, version)
        if require_approved and meta.approval_status != "APPROVED":
            raise PermissionError(
                f"{name} {version} is {meta.approval_status}; approve it before use"
            )
        with (self._dir(name, version) / "model.pkl").open("rb") as handle:
            return pickle.load(handle), meta

    def set_status(self, name: str, version: str, status: str, by: str) -> ModelMetadata:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        meta = self.metadata(name, version)
        now = datetime.now(UTC).isoformat()
        meta.approval_status = status
        meta.approved_by = by
        if status == "APPROVED":
            meta.deployed_at = now
        if status == "RETIRED":
            meta.retired_at = now
        self._write(meta)
        return meta

    def list(self) -> list[ModelMetadata]:
        if not self.root.exists():
            return []
        out = [
            self.metadata(p.parent.name, p.name)
            for p in sorted(self.root.glob("*/*"))
            if (p / "metadata.json").exists()
        ]
        return sorted(out, key=lambda m: m.created_at)

    def next_version(self, name: str) -> str:
        existing = [m.model_version for m in self.list() if m.model_name == name]
        return f"v{len(existing) + 1:03d}"
