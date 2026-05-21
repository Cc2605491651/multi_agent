"""沙箱抽象（spec v4 §7.2）。

阶段 1 只实现 ``LocalBackend``；阶段 4 加 ``E2BBackend`` / ``CubeSandboxBackend``
时上层零改动。接口完整覆盖 ``cancel`` / ``read_file`` / ``write_file`` /
``exec_command`` / ``run_code``——spec §13 明确这五样必须一次到位。
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class SandboxHandle:
    sandbox_id: str
    backend: str
    created_at: str
    metadata: Optional[dict] = field(default_factory=dict)


class SandboxBackend(ABC):
    @abstractmethod
    async def create(self, context_package: str) -> SandboxHandle: ...

    @abstractmethod
    async def destroy(self, handle: SandboxHandle) -> None: ...

    @abstractmethod
    async def cancel(self, handle: SandboxHandle, timeout: float = 5.0) -> bool: ...

    @abstractmethod
    async def exec_command(self, handle: SandboxHandle, cmd: str) -> str: ...

    @abstractmethod
    async def run_code(self, handle: SandboxHandle, code: str) -> str: ...

    @abstractmethod
    async def read_file(self, handle: SandboxHandle, path: str) -> str: ...

    @abstractmethod
    async def write_file(
        self, handle: SandboxHandle, path: str, content: str
    ) -> None: ...


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LocalBackend(SandboxBackend):
    """阶段 1：本地隔离最小化实现。

    - ``create``：在系统临时目录开一个唯一 workdir，写入上下文 package
      （供 Worker 读取），返回 handle。
    - ``read_file`` / ``write_file``：相对路径锚定在 workdir 内，防止越界。
    - ``exec_command``：通过 ``asyncio.create_subprocess_shell`` 执行，
      cwd 锁在 workdir，stdout+stderr 合并返回。
    - ``cancel``：标记 handle 为 cancelled，正在执行的命令通过子进程组终止；
      若 ``timeout`` 内没退出返回 False，上层应转 ``destroy`` 强杀。
    - ``run_code``：把 ``code`` 写到临时 ``.py`` 后用当前 Python 解释器跑。
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root = Path(root_dir) if root_dir else Path(tempfile.gettempdir()) / "ma_local_sandbox"
        self._root.mkdir(parents=True, exist_ok=True)
        # handle -> 当前在跑的 asyncio.subprocess.Process（若有）
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._cancelled: set[str] = set()

    async def create(self, context_package: str) -> SandboxHandle:
        sid = str(uuid.uuid4())
        workdir = self._root / sid
        workdir.mkdir(parents=True, exist_ok=False)
        (workdir / "context.txt").write_text(context_package, encoding="utf-8")
        return SandboxHandle(
            sandbox_id=sid,
            backend="local",
            created_at=_utcnow_iso(),
            metadata={"workdir": str(workdir)},
        )

    async def destroy(self, handle: SandboxHandle) -> None:
        proc = self._procs.pop(handle.sandbox_id, None)
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        self._cancelled.discard(handle.sandbox_id)
        workdir = Path(handle.metadata["workdir"]) if handle.metadata else None
        if workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)

    async def cancel(self, handle: SandboxHandle, timeout: float = 5.0) -> bool:
        self._cancelled.add(handle.sandbox_id)
        proc = self._procs.get(handle.sandbox_id)
        if proc is None or proc.returncode is not None:
            return True
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
        except ProcessLookupError:
            return True

    async def exec_command(self, handle: SandboxHandle, cmd: str) -> str:
        if handle.sandbox_id in self._cancelled:
            raise asyncio.CancelledError(f"sandbox {handle.sandbox_id} cancelled")
        workdir = self._workdir(handle)
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(workdir),
        )
        self._procs[handle.sandbox_id] = proc
        try:
            stdout_b, _ = await proc.communicate()
        finally:
            self._procs.pop(handle.sandbox_id, None)
        if handle.sandbox_id in self._cancelled:
            raise asyncio.CancelledError(f"sandbox {handle.sandbox_id} cancelled")
        return stdout_b.decode("utf-8", errors="replace")

    async def run_code(self, handle: SandboxHandle, code: str) -> str:
        workdir = self._workdir(handle)
        script = workdir / f"_run_{uuid.uuid4().hex[:8]}.py"
        script.write_text(code, encoding="utf-8")
        try:
            return await self.exec_command(
                handle, f"{shlex_quote(sys.executable)} {shlex_quote(str(script.name))}"
            )
        finally:
            try:
                script.unlink()
            except FileNotFoundError:
                pass

    async def read_file(self, handle: SandboxHandle, path: str) -> str:
        target = self._resolve_in_workdir(handle, path)
        return target.read_text(encoding="utf-8")

    async def write_file(
        self, handle: SandboxHandle, path: str, content: str
    ) -> None:
        target = self._resolve_in_workdir(handle, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def _workdir(self, handle: SandboxHandle) -> Path:
        if not handle.metadata or "workdir" not in handle.metadata:
            raise RuntimeError("handle missing workdir; was create() called?")
        return Path(handle.metadata["workdir"])

    def _resolve_in_workdir(self, handle: SandboxHandle, path: str) -> Path:
        workdir = self._workdir(handle).resolve()
        target = (workdir / path).resolve()
        if workdir not in target.parents and target != workdir:
            raise ValueError(f"path {path!r} escapes workdir")
        return target


def shlex_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)
