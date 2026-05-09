"""Dispatch many experiments across a pool of rcli-managed pods.

Each experiment is a single ``rcli exec`` invocation: cd into the miles repo,
run ``python -m miles.utils.debug_utils.experiment_runner run --exp <name>
--registry <registry_module>``. Concurrency is bounded by the size of the pod
pool the user passes in. Pods are *named* (rcli) -- the dispatcher does not
create or destroy pods, only schedules work onto pods the user has already
launched.

A per-pod log file accumulates all rcli stdout for that pod. The dispatcher
reports per-experiment status to its own stdout: which pod picked it up, when it
finished, and the rc.

Single-pod-pool failure mode: if all pods for an experiment fail, that experiment
is skipped (not re-routed to another pod) -- crashes are usually deterministic
and worth investigating, not papering over.
"""

from __future__ import annotations

import dataclasses
import shlex
import subprocess
import threading
from collections.abc import Iterable
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class DispatchResult:
    """Outcome of one experiment on one pod."""

    experiment: str
    pod: str
    return_code: int
    log_path: str


def dispatch(
    *,
    experiments: Iterable[str],
    pods: list[str],
    registry_module: str,
    miles_repo_dir: str = "/workspace/miles",
    log_dir: Path | None = None,
    rcli_binary: str = "rcli",
) -> list[DispatchResult]:
    """Run a set of experiments concurrently across the given pod pool.

    Each experiment is dispatched to one pod via ``rcli exec``. When more
    experiments are queued than pods are available, experiments wait until a
    pod is free. Per-experiment log goes to
    ``<log_dir>/<pod>__<experiment>.log``.

    Returns one DispatchResult per experiment in completion order.
    """
    queue = list(experiments)
    if not queue:
        return []
    if not pods:
        raise ValueError("dispatch requires at least one pod in the pool")

    if log_dir is None:
        log_dir = Path("/tmp/experiment_runner/dispatch")
    log_dir.mkdir(parents=True, exist_ok=True)

    pod_locks: dict[str, threading.Lock] = {p: threading.Lock() for p in pods}
    pod_cursor = threading.Semaphore(len(pods))
    results: list[DispatchResult] = []
    results_lock = threading.Lock()

    def _run_one(exp: str) -> None:
        pod_cursor.acquire()
        chosen_pod: str | None = None
        try:
            for p in pods:
                if pod_locks[p].acquire(blocking=False):
                    chosen_pod = p
                    break
            if chosen_pod is None:
                raise RuntimeError("no pod available even though semaphore allowed entry")
            log_path = log_dir / f"{chosen_pod}__{exp}.log"
            cmd = [
                rcli_binary,
                "exec",
                chosen_pod,
                _build_remote_command(experiment=exp, miles_repo_dir=miles_repo_dir, registry_module=registry_module),
            ]
            with log_path.open("w") as fh:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
            with results_lock:
                results.append(
                    DispatchResult(
                        experiment=exp,
                        pod=chosen_pod,
                        return_code=rc,
                        log_path=str(log_path),
                    )
                )
            print(f"[dispatch] {exp} on {chosen_pod} rc={rc} log={log_path}", flush=True)
        finally:
            if chosen_pod is not None:
                pod_locks[chosen_pod].release()
            pod_cursor.release()

    threads = [threading.Thread(target=_run_one, args=(exp,)) for exp in queue]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _build_remote_command(*, experiment: str, miles_repo_dir: str, registry_module: str) -> str:
    """Build the shell command to run on a remote pod via ``rcli exec``."""
    return (
        f"cd {shlex.quote(miles_repo_dir)} && "
        f"python -m miles.utils.debug_utils.experiment_runner run "
        f"--exp {shlex.quote(experiment)} "
        f"--registry-module {shlex.quote(registry_module)}"
    )
