from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]
REPO_ROOT = Path(__file__).resolve().parent


def _resolve_workspace_root(workspace: Optional[PathLike]) -> Path:
    if workspace in (None, ""):
        return Path.cwd().resolve()

    candidate = Path(workspace)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class WorkspacePaths:
    repo_root: Path
    workspace_root: Path

    @classmethod
    def from_value(cls, workspace: Optional[PathLike] = None) -> "WorkspacePaths":
        return cls(repo_root=REPO_ROOT, workspace_root=_resolve_workspace_root(workspace))

    def display_path(self, path: PathLike) -> str:
        candidate = Path(path).resolve()
        for base in (Path.cwd().resolve(), self.workspace_root.resolve(), self.repo_root.resolve()):
            try:
                relative = candidate.relative_to(base)
            except ValueError:
                continue
            return str(relative) if str(relative) else "."
        return str(candidate)

    def display_workspace_root(self) -> str:
        cwd = Path.cwd().resolve()
        try:
            relative = self.workspace_root.resolve().relative_to(cwd)
            return str(relative) if str(relative) else "."
        except ValueError:
            return self.display_path(self.workspace_root)

    @property
    def config_dir(self) -> Path:
        return self.workspace_root / "config"

    @property
    def data_dir(self) -> Path:
        return self.workspace_root / "data"

    @property
    def experiments_dir(self) -> Path:
        return self.workspace_root / "experiments"

    @property
    def models_dir(self) -> Path:
        return self.workspace_root / "models"

    @property
    def accepted_models_dir(self) -> Path:
        return self.models_dir / "accepted"

    @property
    def train_py_path(self) -> Path:
        return self.workspace_root / "train.py"

    @property
    def feature_spec_path(self) -> Path:
        return self.config_dir / "feature_spec.json"

    @property
    def task_context_path(self) -> Path:
        return self.config_dir / "task_context.md"

    @property
    def feature_spec_schema_path(self) -> Path:
        return self.repo_root / "config" / "feature_spec.schema.json"

    @property
    def baseline_train_path(self) -> Path:
        return self.repo_root / "baselines" / "train.generic.py"

    @property
    def train_data_path(self) -> Path:
        return self.data_dir / "train.parquet"

    @property
    def test_data_path(self) -> Path:
        return self.data_dir / "test.parquet"

    @property
    def meta_path(self) -> Path:
        return self.data_dir / "columns.json"

    @property
    def session_baseline_path(self) -> Path:
        return self.experiments_dir / "session_baseline.json"

    @property
    def search_memory_path(self) -> Path:
        return self.experiments_dir / "search_memory.jsonl"

    @property
    def search_summary_path(self) -> Path:
        return self.experiments_dir / "search_summary.json"
