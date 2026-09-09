"""Tests for shell utilities."""

import errno
import inspect
import subprocess
import time

import pytest

from slurm_gen.shell import DEFAULT_COMMAND_TIMEOUT_S, run_command


class TestRunCommand:
    """Tests for run_command function."""

    def test_successful_command(self):
        """Test running a successful command."""
        result = run_command(["echo", "hello"])
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_command_with_args(self):
        """Test command with multiple arguments."""
        result = run_command(["echo", "-n", "test"])
        assert result.returncode == 0
        assert "test" in result.stdout

    def test_failing_command(self):
        """Test a command that returns non-zero."""
        result = run_command(["false"])
        assert result.returncode != 0

    def test_check_raises(self):
        """Test that check=True raises on failure."""
        with pytest.raises(subprocess.CalledProcessError):
            run_command(["false"], check=True)

    def test_captures_stderr(self):
        """Test that stderr is captured."""
        result = run_command(["ls", "/nonexistent_path_12345"])
        assert result.returncode != 0
        assert result.stderr  # Should have error message

    def test_captures_stdout(self):
        """Test that stdout is captured."""
        result = run_command(["echo", "output"])
        assert "output" in result.stdout

    def test_timeout(self):
        """Test command timeout."""
        with pytest.raises(subprocess.TimeoutExpired):
            run_command(["sleep", "10"], timeout=0.1)


class TestSpawnRetries:
    """Transient fork() failures on a saturated login node are retried."""

    def test_retries_eagain_then_succeeds(self, monkeypatch):
        calls = {"n": 0}
        real_run = subprocess.run

        def flaky_run(argv, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise OSError(errno.EAGAIN, "Resource temporarily unavailable")
            return real_run(argv, **kwargs)

        monkeypatch.setattr(subprocess, "run", flaky_run)
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)

        result = run_command(["echo", "hello"], spawn_retry_backoff=0.0)

        assert calls["n"] == 3
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_gives_up_after_max_retries(self, monkeypatch):
        calls = {"n": 0}

        def always_eagain(argv, **kwargs):
            calls["n"] += 1
            raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

        monkeypatch.setattr(subprocess, "run", always_eagain)
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)

        with pytest.raises(OSError):
            run_command(["echo", "hello"], max_spawn_retries=2, spawn_retry_backoff=0.0)

        assert calls["n"] == 3  # the initial attempt plus two retries

    def test_permanent_oserror_is_not_retried(self, monkeypatch):
        """ENOENT means the command does not exist; retrying cannot help."""
        calls = {"n": 0}

        def missing_command(argv, **kwargs):
            calls["n"] += 1
            raise OSError(errno.ENOENT, "No such file or directory")

        monkeypatch.setattr(subprocess, "run", missing_command)

        with pytest.raises(OSError):
            run_command(["definitely-not-a-command"])

        assert calls["n"] == 1

    def test_default_timeout_is_bounded(self):
        """A wedged slurmctld must not hang the caller forever."""
        assert DEFAULT_COMMAND_TIMEOUT_S > 0
        assert inspect.signature(run_command).parameters["timeout"].default == (
            DEFAULT_COMMAND_TIMEOUT_S
        )
