"""Launch and manage an sglang HTTP server subprocess from a ``RunConfig``.

Provides three primitives:

- ``build_sg_launch_command``: pure function turning a RunConfig into the shell
  command string. No side effects -- unit-testable.
- ``probe_sg_ready``: TCP connect probe to determine whether a server is already
  running at ``host:port``. Used to support attach-to-existing-server.
- ``SgServerProcess``: context manager that starts the server in the background
  (or attaches to an existing one), waits for readiness, exposes its URL.

The runner uses these together: probe first, attach if up, otherwise start +
wait. The user has stated they prefer pod-level reuse of sglang servers; the
context manager exits without killing the server unless it was started by this
process AND ``stop_on_exit`` is True (default False).
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import socket
import subprocess
from pathlib import Path

from miles.utils.debug_utils.experiment_runner.sg_trigger import wait_for_sg_ready
from miles.utils.debug_utils.experiment_runner.spec import RunConfig


@dataclasses.dataclass(frozen=True)
class SgLaunchCommand:
    """Composed sglang launch command + the env it should be run with."""

    command: str
    """Full shell command including ``python -m sglang.launch_server`` and all args.
    Pipes stdout+stderr through ``tee`` to ``log_path`` so logs are persisted even
    when the process is detached.
    """

    env: dict[str, str]
    """Environment variables to export before invoking the command. The full os.environ
    plus run_config.sg_env (spec deltas merged onto canonical) plus DUMPER_* if
    grafter is enabled.
    """

    log_path: Path
    """Server log path; ``tee``'d to in the command."""

    server_url: str
    """e.g. ``http://0.0.0.0:30000``. Probed for readiness; passed to triggers."""


def build_sg_launch_command(
    run_config: RunConfig,
    *,
    sg_repo_dir: str = "/workspace/sglang",
    miles_repo_dir: str = "/workspace/miles",
    server_host: str = "0.0.0.0",
    server_port: int = 30000,
    output_subdir: str = "sg",
) -> SgLaunchCommand:
    """Compose the full sglang launch command for ``run_config``.

    The output dump directory is ``<output_root>/<run_name>-<output_subdir>/``;
    the server log goes to ``<dump_dir>/server.log``. The command ``cd``s into
    the sglang source tree (so ``-m sglang.launch_server`` resolves to the
    in-tree, editable-installed version) and adds ``miles_repo_dir`` to
    PYTHONPATH so V4 plugin imports succeed.

    If ``run_config.kind == grafter_pair``, DUMPER_GRAFTER_* env vars are wired
    in, the spec's b2t/t2b filters set, and ROLE=target.
    """
    dump_dir = Path(run_config.output_root) / f"{run_config.name}-{output_subdir}"
    log_path = dump_dir / "server.log"

    env: dict[str, str] = {**os.environ, **run_config.sg_env}
    env["DUMPER_ENABLE"] = "1"
    env["DUMPER_NON_INTRUSIVE_MODE"] = "off"
    env["DUMPER_DIR"] = str(dump_dir)
    if run_config.dumper_filter:
        env["DUMPER_FILTER"] = run_config.dumper_filter

    if run_config.kind.value == "grafter_pair":
        env.update(run_config.grafter_env)
        env["DUMPER_GRAFTER_ENABLE"] = "1"
        env["DUMPER_GRAFTER_ROLE"] = "target"
        if run_config.grafter_b2t_filter:
            env["DUMPER_GRAFTER_B2T_FILTER"] = run_config.grafter_b2t_filter
        if run_config.grafter_t2b_filter:
            env["DUMPER_GRAFTER_T2B_FILTER"] = run_config.grafter_t2b_filter

    pythonpath_parts = [miles_repo_dir]
    if "PYTHONPATH" in env:
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    server_args: list[str] = []
    if run_config.sg_model_path:
        server_args += ["--model-path", run_config.sg_model_path]
    server_args += ["--host", server_host, "--port", str(server_port)]
    server_args += list(run_config.sg_server_args)

    server_args_quoted = " ".join(shlex.quote(a) for a in server_args)
    cmd = (
        f"mkdir -p {shlex.quote(str(dump_dir))} && "
        f"cd {shlex.quote(sg_repo_dir)} && "
        f"python -m sglang.launch_server {server_args_quoted} 2>&1 | "
        f"tee {shlex.quote(str(log_path))}"
    )
    server_url = f"http://{server_host}:{server_port}"

    return SgLaunchCommand(command=cmd, env=env, log_path=log_path, server_url=server_url)


def probe_sg_ready(*, host: str, port: int, timeout_s: float = 2.0) -> bool:
    """Return True iff a TCP connection to ``host:port`` succeeds within ``timeout_s``."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


class SgServerProcess:
    """Context manager that brings up (or attaches to) one sglang HTTP server.

    Use as::

        launch = build_sg_launch_command(run_config)
        with SgServerProcess(launch, ready_timeout_s=1800) as server:
            do_things(server.server_url)

    On enter, probes ``server_url``. If reachable, attaches without spawning. If
    not, runs ``launch.command`` in the background via ``bash -c`` and waits up
    to ``ready_timeout_s`` for the port to bind.

    On exit, leaves an attached-to or self-spawned server running by default
    (caller controls via ``stop_on_exit``). The user has stated they want
    pod-level reuse of sglang servers across multiple sequential experiments.
    """

    def __init__(
        self,
        launch: SgLaunchCommand,
        *,
        host: str = "0.0.0.0",
        port: int = 30000,
        ready_timeout_s: int = 1800,
        stop_on_exit: bool = False,
    ) -> None:
        self._launch = launch
        self._host = host
        self._port = port
        self._ready_timeout_s = ready_timeout_s
        self._stop_on_exit = stop_on_exit
        self._proc: subprocess.Popen | None = None
        self._attached = False

    @property
    def server_url(self) -> str:
        return self._launch.server_url

    @property
    def attached(self) -> bool:
        """True if we attached to a pre-existing server (did not spawn)."""
        return self._attached

    def __enter__(self) -> SgServerProcess:
        if probe_sg_ready(host=self._host, port=self._port):
            self._attached = True
            return self
        self._launch.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            ["bash", "-c", self._launch.command],
            env=self._launch.env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wait_for_sg_ready(host=self._host, port=self._port, timeout_s=self._ready_timeout_s)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._proc is None or self._attached:
            return
        if not self._stop_on_exit:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
