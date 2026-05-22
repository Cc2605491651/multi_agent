"""E2BBackend 单测（阶段 4b 任务 4b.3）。

不打真实 E2B API：用 ``sandbox_factory`` 注入 mock 对象，验证接口契约。

要跑「真实 E2B 烟雾测试」请单独写脚本并设 ``E2B_API_KEY``。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from worker.sandbox import SandboxHandle
from worker.sandbox_e2b import DEFAULT_WORKDIR, E2BBackend


# ---------- mock AsyncSandbox ----------


@dataclass
class _FakeCommandResult:
    stdout: str = ""
    stderr: str = ""


class _FakeCommands:
    def __init__(self, outputs: dict[str, _FakeCommandResult] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[dict] = []

    async def run(self, cmd: str, cwd: str | None = None, **kwargs) -> _FakeCommandResult:
        self.calls.append({"cmd": cmd, "cwd": cwd, "kwargs": kwargs})
        for prefix, out in self.outputs.items():
            if cmd.startswith(prefix):
                return out
        return _FakeCommandResult(stdout=f"ran: {cmd}")


class _FakeFiles:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.reads: list[str] = []

    async def read(self, path: str, **kwargs) -> str:
        self.reads.append(path)
        if path not in self.store:
            raise FileNotFoundError(path)
        return self.store[path]

    async def write(self, path: str, data, **kwargs):
        self.store[path] = data if isinstance(data, str) else data.decode()
        return None


class _FakeAsyncSandbox:
    """对齐 e2b.AsyncSandbox 的公开接口子集。"""

    counter = 0

    def __init__(self) -> None:
        _FakeAsyncSandbox.counter += 1
        self.sandbox_id = f"fake_sb_{_FakeAsyncSandbox.counter}"
        self.commands = _FakeCommands()
        self.files = _FakeFiles()
        self.killed = False
        self.kill_delay = 0.0

    async def kill(self) -> bool:
        if self.kill_delay:
            await asyncio.sleep(self.kill_delay)
        self.killed = True
        return True


def _factory_for(sb: _FakeAsyncSandbox):
    async def _f():
        return sb
    return _f


@pytest.fixture
def fake_sb():
    return _FakeAsyncSandbox()


@pytest.fixture
def backend(fake_sb):
    return E2BBackend(
        api_key="fake_key",  # 跳过环境变量校验
        sandbox_factory=_factory_for(fake_sb),
    )


# ---------- 测试 ----------


async def test_create_writes_context_package(backend, fake_sb) -> None:
    handle = await backend.create("context payload 123")
    assert handle.backend == "e2b"
    assert handle.sandbox_id == fake_sb.sandbox_id
    assert handle.metadata["workdir"] == DEFAULT_WORKDIR
    assert fake_sb.files.store[f"{DEFAULT_WORKDIR}/context.txt"] == "context payload 123"


async def test_destroy_calls_kill_and_drops_handle(backend, fake_sb) -> None:
    h = await backend.create("x")
    await backend.destroy(h)
    assert fake_sb.killed is True
    # second destroy 是幂等的（已不在 handles 里）
    await backend.destroy(h)


async def test_cancel_returns_true_on_quick_kill(backend, fake_sb) -> None:
    h = await backend.create("x")
    assert await backend.cancel(h, timeout=1.0) is True
    assert fake_sb.killed is True


async def test_cancel_times_out_when_kill_hangs(backend, fake_sb) -> None:
    fake_sb.kill_delay = 2.0
    h = await backend.create("x")
    assert await backend.cancel(h, timeout=0.1) is False


async def test_cancel_unknown_handle_ok(backend) -> None:
    ghost = SandboxHandle(
        sandbox_id="never_created", backend="e2b", created_at="2026-05-22"
    )
    assert await backend.cancel(ghost) is True


async def test_exec_command_returns_stdout(backend, fake_sb) -> None:
    fake_sb.commands.outputs["echo"] = _FakeCommandResult(stdout="hello from sandbox\n")
    h = await backend.create("x")
    out = await backend.exec_command(h, "echo hello from sandbox")
    assert "hello from sandbox" in out
    assert fake_sb.commands.calls[-1]["cwd"] == DEFAULT_WORKDIR


async def test_run_code_writes_script_then_runs_python(backend, fake_sb) -> None:
    fake_sb.commands.outputs["python3"] = _FakeCommandResult(stdout="3\n")
    h = await backend.create("x")
    out = await backend.run_code(h, "print(1+2)")
    assert "3" in out
    # 应该有一个 _run_*.py 落进 sandbox
    py_files = [p for p in fake_sb.files.store if p.startswith(f"{DEFAULT_WORKDIR}/_run_") and p.endswith(".py")]
    assert len(py_files) == 1
    assert fake_sb.files.store[py_files[0]] == "print(1+2)"


async def test_read_write_relative_paths_anchored_to_workdir(backend, fake_sb) -> None:
    h = await backend.create("x")
    await backend.write_file(h, "sub/data.txt", "payload")
    assert fake_sb.files.store[f"{DEFAULT_WORKDIR}/sub/data.txt"] == "payload"
    got = await backend.read_file(h, "sub/data.txt")
    assert got == "payload"


async def test_absolute_paths_passthrough(backend, fake_sb) -> None:
    h = await backend.create("x")
    await backend.write_file(h, "/tmp/abs.txt", "abs")
    assert fake_sb.files.store["/tmp/abs.txt"] == "abs"


async def test_exec_on_unknown_handle_raises(backend) -> None:
    ghost = SandboxHandle(
        sandbox_id="never_created", backend="e2b", created_at="2026-05-22"
    )
    with pytest.raises(RuntimeError, match="unknown e2b sandbox handle"):
        await backend.exec_command(ghost, "echo")


def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="E2B_API_KEY not set"):
        E2BBackend()


def test_env_var_picks_up_key(monkeypatch) -> None:
    monkeypatch.setenv("E2B_API_KEY", "env-supplied-key")
    backend = E2BBackend()
    assert backend._api_key == "env-supplied-key"


# ---------- 工厂开关 ----------


def test_make_sandbox_default_is_local(monkeypatch) -> None:
    from worker.sandbox import LocalBackend, make_sandbox

    monkeypatch.delenv("SANDBOX_BACKEND", raising=False)
    assert isinstance(make_sandbox(), LocalBackend)


def test_make_sandbox_e2b_requires_key(monkeypatch) -> None:
    from worker.sandbox import make_sandbox

    monkeypatch.setenv("SANDBOX_BACKEND", "e2b")
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="E2B_API_KEY"):
        make_sandbox()


def test_make_sandbox_unknown_backend_raises(monkeypatch) -> None:
    from worker.sandbox import make_sandbox

    monkeypatch.setenv("SANDBOX_BACKEND", "bogus")
    with pytest.raises(ValueError, match="unknown SANDBOX_BACKEND"):
        make_sandbox()


def test_make_sandbox_e2b_with_key(monkeypatch) -> None:
    from worker.sandbox import make_sandbox
    from worker.sandbox_e2b import E2BBackend

    monkeypatch.setenv("SANDBOX_BACKEND", "e2b")
    monkeypatch.setenv("E2B_API_KEY", "test")
    assert isinstance(make_sandbox(), E2BBackend)
