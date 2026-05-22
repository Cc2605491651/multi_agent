"""E2B 云沙箱后端（spec v4 §7.2，阶段 4b 任务 4b.3）。

走 ``e2b`` 官方 Python SDK 的 ``AsyncSandbox``，把 ``SandboxBackend`` 6 个抽象
方法都映射上。配置：

- ``E2B_API_KEY``：必填，去 https://e2b.dev/dashboard → API Keys 拿一个
- ``E2B_TEMPLATE``：可选，自定义模板 ID；不填用 e2b 默认
- ``E2B_SANDBOX_TIMEOUT``：可选，sandbox 生存时长（秒），默认 3600

``cancel`` 实现：简化为 ``kill`` 整个 sandbox（spec §5.3 「取消是尽力而为」
允许，避免追逐 e2b 内部进程 pid 的复杂度）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone

from worker.sandbox import SandboxBackend, SandboxHandle

_log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 3600
DEFAULT_WORKDIR = "/home/user"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class E2BBackend(SandboxBackend):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        template: str | None = None,
        timeout: int | None = None,
        sandbox_factory=None,
    ) -> None:
        """``sandbox_factory`` 仅供测试注入（避免打真实 API）；默认走 e2b SDK。"""
        self._api_key = api_key or os.environ.get("E2B_API_KEY")
        if not self._api_key and sandbox_factory is None:
            raise RuntimeError(
                "E2B_API_KEY not set; export it or pass api_key explicitly. "
                "Get a key at https://e2b.dev/dashboard"
            )
        self._template = template or os.environ.get("E2B_TEMPLATE")
        self._timeout = timeout or int(
            os.environ.get("E2B_SANDBOX_TIMEOUT", str(DEFAULT_TIMEOUT))
        )
        self._sandbox_factory = sandbox_factory  # 测试时注入 mock
        self._handles: dict[str, object] = {}

    async def create(self, context_package: str) -> SandboxHandle:
        sb = await self._create_async_sandbox()
        sid = getattr(sb, "sandbox_id", None) or f"sb_{uuid.uuid4().hex[:12]}"
        self._handles[sid] = sb
        # 把 context 包写进 sandbox 默认 workdir
        await sb.files.write(f"{DEFAULT_WORKDIR}/context.txt", context_package)
        return SandboxHandle(
            sandbox_id=sid,
            backend="e2b",
            created_at=_utcnow_iso(),
            metadata={"workdir": DEFAULT_WORKDIR, "template": self._template or ""},
        )

    async def destroy(self, handle: SandboxHandle) -> None:
        sb = self._handles.pop(handle.sandbox_id, None)
        if sb is None:
            return
        try:
            await sb.kill()
        except Exception as e:  # noqa: BLE001
            _log.warning("e2b kill failed for %s: %s", handle.sandbox_id, e)

    async def cancel(self, handle: SandboxHandle, timeout: float = 5.0) -> bool:
        """spec §5.3：通知优雅 abort，超时返回 False 由编排器 destroy 强杀。

        e2b 没单独的「取消所有进程」原语，简化为「等 timeout 内 kill 完即视为成功」。
        """
        sb = self._handles.get(handle.sandbox_id)
        if sb is None:
            return True
        try:
            await asyncio.wait_for(sb.kill(), timeout=timeout)
            self._handles.pop(handle.sandbox_id, None)
            return True
        except asyncio.TimeoutError:
            return False
        except Exception as e:  # noqa: BLE001
            _log.warning("e2b cancel failed for %s: %s", handle.sandbox_id, e)
            return False

    async def exec_command(self, handle: SandboxHandle, cmd: str) -> str:
        sb = self._require(handle)
        result = await sb.commands.run(cmd, cwd=DEFAULT_WORKDIR)
        return _format_command_output(result)

    async def run_code(self, handle: SandboxHandle, code: str) -> str:
        sb = self._require(handle)
        path = f"{DEFAULT_WORKDIR}/_run_{uuid.uuid4().hex[:8]}.py"
        await sb.files.write(path, code)
        result = await sb.commands.run(f"python3 {path}", cwd=DEFAULT_WORKDIR)
        return _format_command_output(result)

    async def read_file(self, handle: SandboxHandle, path: str) -> str:
        sb = self._require(handle)
        return await sb.files.read(_resolve(path))

    async def write_file(
        self, handle: SandboxHandle, path: str, content: str
    ) -> None:
        sb = self._require(handle)
        await sb.files.write(_resolve(path), content)

    def _require(self, handle: SandboxHandle):
        sb = self._handles.get(handle.sandbox_id)
        if sb is None:
            raise RuntimeError(
                f"unknown e2b sandbox handle: {handle.sandbox_id}; "
                "was create() called or is it already destroyed?"
            )
        return sb

    async def _create_async_sandbox(self):
        if self._sandbox_factory is not None:
            return await self._sandbox_factory()
        from e2b import AsyncSandbox

        return await AsyncSandbox.create(
            template=self._template,
            timeout=self._timeout,
            api_key=self._api_key,
        )


def _resolve(path: str) -> str:
    """相对路径锚定到 DEFAULT_WORKDIR，绝对路径直接透传。"""
    return path if path.startswith("/") else f"{DEFAULT_WORKDIR}/{path}"


def _format_command_output(result) -> str:
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    if stderr and stdout:
        return f"{stdout}{stderr}"
    return stdout or stderr
