"""Per-run reproducibility manifest: git SHAs of the synced repos + image SHA.

Each run writes ``<output_dir>/manifest.json`` capturing the exact source-state
that produced the dump. A future rerun reading the manifest can:
- ``git checkout <sha>`` each repo to reproduce the same code
- pull the same image SHA (recorded in canonical config)
- rehydrate the spec from ``run_config_snapshot``
- expect to land at the same dump bytes (modulo intrinsic kernel non-determinism)

Manifest schema is intentionally small and stable -- kept in this module instead
of inlined into the registry RunRecord so it can be queried independently
(e.g. "what miles SHA produced experiment X?") without parsing the registry.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class GitInfo:
    repo_dir: str
    head_sha: str
    head_short: str
    is_dirty: bool


def collect_git(repo_dir: Path) -> GitInfo | None:
    """Return ``GitInfo`` for ``repo_dir`` or ``None`` if not a git repo."""
    if not (repo_dir / ".git").exists():
        return None
    head = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    short = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo_dir), "status", "--porcelain"], text=True)
    return GitInfo(
        repo_dir=str(repo_dir),
        head_sha=head,
        head_short=short,
        is_dirty=bool(status.strip()),
    )


def write_manifest(
    *,
    output_dir: Path,
    run_config_snapshot: dict,
    spec_snapshot: dict,
    git_infos: dict[str, GitInfo],
    image: str | None = None,
    started_at: str,
    finished_at: str,
) -> Path:
    """Serialise a manifest to ``<output_dir>/manifest.json``.

    ``git_infos`` keys are short repo labels (e.g. ``sglang``, ``miles``,
    ``megatron``). Returns the manifest path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_snapshot": spec_snapshot,
        "run_config_snapshot": run_config_snapshot,
        "git": {label: dataclasses.asdict(info) for label, info in git_infos.items()},
        "image": image,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, default=str))
    return manifest_path
