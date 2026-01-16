"""Tests for SLURM client implementations."""

from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import pytest

from slurm_gen import (
    FakeSlurmClient,
    FakeSlurmClientConfig,
    SlurmClient,
    SlurmClientConfig,
    SlurmConfig,
    SlurmJob,
)
import slurm_gen.client


@dataclass
class MockResult:
    """Mock result for run_command calls."""
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def make_mock_run(responses: dict[str, MockResult]) -> Callable:
    """Create a mock_run function that returns different responses based on command.

    Args:
        responses: Dict mapping command name (e.g., "sbatch") to MockResult.
    """
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        return responses.get(cmd[0], MockResult())

    mock_run.calls = calls
    return mock_run


@pytest.fixture
def slurm_config(tmp_path: Path) -> SlurmConfig:
    """Standard SlurmConfig for tests."""
    return SlurmConfig(
        template_path="templates/base.sbatch",
        script_dir=str(tmp_path / "scripts"),
        log_dir=str(tmp_path / "logs"),
    )


@pytest.fixture
def configured_client(tmp_path: Path, slurm_config: SlurmConfig) -> SlurmClient:
    """Pre-configured SlurmClient for tests."""
    client = SlurmClient(SlurmClientConfig())
    client.configure(slurm_config)
    return client


class TestSlurmJob:
    """Tests for SlurmJob dataclass."""

    def test_job_creation(self):
        """Test basic job creation."""
        job = SlurmJob(
            job_id="12345",
            name="test_job",
            script_path="/path/to/script.sh",
            log_path="/path/to/log.txt",
        )
        assert job.job_id == "12345"
        assert job.name == "test_job"
        assert job.state == "PENDING"
        assert job.return_code is None

    def test_job_with_state(self):
        """Test job creation with custom state."""
        job = SlurmJob(
            job_id="12345",
            name="test_job",
            script_path="/path/to/script.sh",
            log_path="/path/to/log.txt",
            state="RUNNING",
            return_code=None,
        )
        assert job.state == "RUNNING"


class TestFakeSlurmClient:
    """Tests for FakeSlurmClient."""

    def test_submit_job(self, tmp_path: Path):
        """Test submitting a single job."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        job_id = client.submit(
            "test_job",
            str(tmp_path / "job.sbatch"),
            str(tmp_path / "log.txt"),
        )
        assert job_id == "1"
        assert client.squeue()[job_id] == "PENDING"

    def test_submit_multiple_jobs(self, tmp_path: Path):
        """Test submitting multiple jobs."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        job1 = client.submit("job1", str(tmp_path / "j1.sh"), str(tmp_path / "l1.txt"))
        job2 = client.submit("job2", str(tmp_path / "j2.sh"), str(tmp_path / "l2.txt"))
        assert job1 == "1"
        assert job2 == "2"
        assert len(client.squeue()) == 2

    def test_state_transitions(self, tmp_path: Path):
        """Test job state transitions."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        job_id = client.submit("demo", str(tmp_path / "job.sbatch"), str(tmp_path / "log.txt"))

        client.set_state(job_id, "RUNNING")
        assert client.squeue()[job_id] == "RUNNING"

        client.set_state(job_id, "COMPLETED", return_code=0)
        job = client.get_job(job_id)
        assert job.state == "COMPLETED"
        assert job.return_code == 0

    def test_cancel_job(self, tmp_path: Path):
        """Test cancelling a job."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        job_id = client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))
        client.cancel(job_id)
        assert client.squeue()[job_id] == "CANCELLED"

    def test_remove_job(self, tmp_path: Path):
        """Test removing a job from tracking."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        job_id = client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))
        assert job_id in client.squeue()
        client.remove(job_id)
        assert job_id not in client.squeue()

    def test_job_ids_by_name(self, tmp_path: Path):
        """Test finding jobs by name."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        client.submit("job_a", str(tmp_path / "a.sh"), str(tmp_path / "a.txt"))
        client.submit("job_b", str(tmp_path / "b.sh"), str(tmp_path / "b.txt"))
        client.submit("job_a", str(tmp_path / "a2.sh"), str(tmp_path / "a2.txt"))

        job_a_ids = client.job_ids_by_name("job_a")
        assert len(job_a_ids) == 2

    def test_submit_array(self, tmp_path: Path):
        """Test submitting a job array."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        script_path = str(tmp_path / "array.sbatch")
        log_paths = [str(tmp_path / f"log_{i}.txt") for i in range(3)]
        task_names = ["task0", "task1", "task2"]

        job_ids = client.submit_array("test_array", script_path, log_paths, task_names)

        assert len(job_ids) == 3
        assert all(isinstance(jid, str) for jid in job_ids)
        # Array job IDs should be like "1_0", "1_1", "1_2"
        assert job_ids == ["1_0", "1_1", "1_2"]

        # Check all jobs are tracked
        for job_id in job_ids:
            job = client.get_job(job_id)
            assert job.state == "PENDING"
            assert job.script_path == script_path

    def test_submit_array_with_start_index(self, tmp_path: Path):
        """Test submitting an array job with custom start index."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        script_path = str(tmp_path / "array.sbatch")
        log_paths = [str(tmp_path / f"log_{i}.txt") for i in range(2)]
        task_names = ["task5", "task6"]

        job_ids = client.submit_array(
            "test_array", script_path, log_paths, task_names, start_index=5
        )

        assert job_ids == ["1_5", "1_6"]

    def test_register_job(self, tmp_path: Path):
        """Test registering an external job."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        job_id = client.register_job(
            "99999",
            "external_job",
            str(tmp_path / "external.sh"),
            str(tmp_path / "external.log"),
            state="RUNNING",
        )
        assert job_id == "99999"
        job = client.get_job(job_id)
        assert job.state == "RUNNING"
        assert job.name == "external_job"

    def test_persist_artifacts(self, tmp_path: Path):
        """Test artifact persistence."""
        log_path = tmp_path / "logs" / "job.log"
        client = FakeSlurmClient(FakeSlurmClientConfig(persist_artifacts=True))
        client.submit("test", str(tmp_path / "job.sh"), str(log_path))
        assert log_path.exists()

    def test_submit_array_persist_artifacts(self, tmp_path: Path):
        """Test artifact persistence for array jobs."""
        client = FakeSlurmClient(FakeSlurmClientConfig(persist_artifacts=True))
        script_path = str(tmp_path / "array.sbatch")
        log_paths = [str(tmp_path / "logs" / f"log_{i}.txt") for i in range(3)]
        task_names = ["task0", "task1", "task2"]

        job_ids = client.submit_array("test_array", script_path, log_paths, task_names)

        assert len(job_ids) == 3
        for log_path in log_paths:
            assert Path(log_path).exists()

    def test_register_job_with_non_numeric_id(self, tmp_path: Path):
        """Test registering a job with a non-numeric ID."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        job_id = client.register_job(
            "abc_xyz",
            "external_job",
            str(tmp_path / "external.sh"),
            str(tmp_path / "external.log"),
        )
        assert job_id == "abc_xyz"
        # Next job should use _next_id (not crash on ValueError)
        next_job_id = client.submit("new_job", str(tmp_path / "j.sh"), str(tmp_path / "l.txt"))
        assert next_job_id is not None


class TestSlurmClient:
    """Tests for SlurmClient (mocked)."""

    def test_submit_parses_job_id(self, configured_client, tmp_path, monkeypatch):
        """Test that submit correctly parses job ID from sbatch output."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="Submitted batch job 12345")})
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))
        assert job_id == "12345"

    def test_submit_handles_failure(self, configured_client, tmp_path, monkeypatch):
        """Test that submit raises on sbatch failure."""
        mock_run = make_mock_run({
            "sbatch": MockResult(returncode=1, stderr="sbatch: error: invalid option")
        })
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        with pytest.raises(RuntimeError, match="sbatch failed"):
            configured_client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))

    def test_submit_array(self, configured_client, tmp_path, monkeypatch):
        """Test that submit_array correctly calls sbatch with --array."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="Submitted batch job 10000")})
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        script_path = str(tmp_path / "array.sbatch")
        log_paths = [str(tmp_path / f"log_{i}.txt") for i in range(3)]
        task_names = ["task0", "task1", "task2"]

        job_ids = configured_client.submit_array("test_array", script_path, log_paths, task_names)

        # Verify sbatch was called with --array flag
        assert len(mock_run.calls) == 1
        assert mock_run.calls[0][0] == "sbatch"
        assert "--array=0-2" in mock_run.calls[0]
        assert script_path in mock_run.calls[0]

        # Verify job IDs were generated
        assert job_ids == ["10000_0", "10000_1", "10000_2"]

        # Verify jobs are tracked
        for idx, job_id in enumerate(job_ids):
            job = configured_client.get_job(job_id)
            assert f"task{idx}" in job.name
            assert job.script_path == script_path

    def test_cancel(self, configured_client, monkeypatch):
        """Test cancelling a job."""
        mock_run = make_mock_run({})
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        configured_client.cancel("12345")
        assert mock_run.calls[-1] == ["scancel", "12345"]

    def test_parse_job_id_variations(self):
        """Test parsing various sbatch output formats."""
        assert SlurmClient._parse_job_id("Submitted batch job 12345") == "12345"
        assert SlurmClient._parse_job_id("12345") == "12345"
        assert SlurmClient._parse_job_id("Job 12345_0 submitted") == "12345_0"
        assert SlurmClient._parse_job_id("No job id here") is None

    def test_submit_with_persist_artifacts(self, tmp_path, slurm_config, monkeypatch):
        """Test that submit creates log file when persist_artifacts is True."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="Submitted batch job 12345")})
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        client = SlurmClient(SlurmClientConfig(persist_artifacts=True))
        client.configure(slurm_config)

        log_path = tmp_path / "logs" / "job.log"
        job_id = client.submit("test", str(tmp_path / "job.sh"), str(log_path))
        assert job_id == "12345"
        assert log_path.exists()

    def test_submit_unable_to_parse_job_id(self, configured_client, tmp_path, monkeypatch):
        """Test that submit raises when job ID can't be parsed."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="Some output without job id")})
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        with pytest.raises(RuntimeError, match="Unable to parse job id"):
            configured_client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))

    def test_submit_array_sbatch_failure(self, configured_client, tmp_path, monkeypatch):
        """Test that submit_array raises on sbatch failure."""
        mock_run = make_mock_run({
            "sbatch": MockResult(returncode=1, stderr="sbatch: error: invalid option")
        })
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        with pytest.raises(RuntimeError, match="sbatch failed for array"):
            configured_client.submit_array(
                "test_array",
                str(tmp_path / "array.sbatch"),
                [str(tmp_path / "log.txt")],
                ["task0"],
            )

    def test_submit_array_parse_failure(self, configured_client, tmp_path, monkeypatch):
        """Test that submit_array raises when job ID can't be parsed."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="No valid job id here")})
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        with pytest.raises(RuntimeError, match="Unable to parse job id"):
            configured_client.submit_array(
                "test_array",
                str(tmp_path / "array.sbatch"),
                [str(tmp_path / "log.txt")],
                ["task0"],
            )

    def test_submit_array_with_persist_artifacts(self, tmp_path, slurm_config, monkeypatch):
        """Test submit_array creates log files when persist_artifacts is True."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="Submitted batch job 10000")})
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        client = SlurmClient(SlurmClientConfig(persist_artifacts=True))
        client.configure(slurm_config)

        log_paths = [str(tmp_path / "logs" / f"log_{i}.txt") for i in range(3)]
        job_ids = client.submit_array(
            "test_array",
            str(tmp_path / "array.sbatch"),
            log_paths,
            ["task0", "task1", "task2"],
        )

        assert len(job_ids) == 3
        for log_path in log_paths:
            assert Path(log_path).exists()

    def test_remove(self, configured_client, tmp_path, monkeypatch):
        """Test removing a job from tracking."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="Submitted batch job 12345")})
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))
        assert job_id == "12345"

        configured_client.remove(job_id)
        with pytest.raises(KeyError):
            configured_client.get_job(job_id)

    def test_job_ids_by_name(self, configured_client, tmp_path, monkeypatch):
        """Test finding jobs by name."""
        call_count = [0]

        def mock_run(cmd, **kwargs):
            call_count[0] += 1
            return MockResult(stdout=f"Submitted batch job {10000 + call_count[0]}")

        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        configured_client.submit("job_a", str(tmp_path / "a.sh"), str(tmp_path / "a.txt"))
        configured_client.submit("job_b", str(tmp_path / "b.sh"), str(tmp_path / "b.txt"))
        configured_client.submit("job_a", str(tmp_path / "a2.sh"), str(tmp_path / "a2.txt"))

        job_a_ids = configured_client.job_ids_by_name("job_a")
        assert len(job_a_ids) == 2

    def test_register_job(self, configured_client, tmp_path):
        """Test registering an external job."""
        job_id = configured_client.register_job(
            "99999",
            "external_job",
            str(tmp_path / "external.sh"),
            str(tmp_path / "external.log"),
            state="RUNNING",
        )

        assert job_id == "99999"
        job = configured_client.get_job(job_id)
        assert job.state == "RUNNING"
        assert job.name == "external_job"

    def test_squeue_no_jobs(self, configured_client):
        """Test squeue with no tracked jobs."""
        statuses = configured_client.squeue()
        assert statuses == {}

    def test_squeue_with_jobs(self, configured_client, tmp_path, monkeypatch):
        """Test squeue with tracked jobs."""
        mock_run = make_mock_run({
            "sbatch": MockResult(stdout="Submitted batch job 12345"),
            "squeue": MockResult(stdout="12345 RUNNING"),
        })
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))
        statuses = configured_client.squeue()

        assert job_id in statuses
        assert statuses[job_id] == "RUNNING"

    def test_squeue_failed_falls_back_to_sacct(self, configured_client, tmp_path, monkeypatch):
        """Test squeue falls back to sacct when squeue fails."""
        mock_run = make_mock_run({
            "sbatch": MockResult(stdout="Submitted batch job 12345"),
            "squeue": MockResult(returncode=1, stderr="Invalid job id"),
            "sacct": MockResult(stdout="12345|COMPLETED"),
        })
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))
        statuses = configured_client.squeue()

        assert job_id in statuses
        assert statuses[job_id] == "COMPLETED"

    def test_squeue_jobs_missing_from_queue_check_sacct(self, configured_client, tmp_path, monkeypatch):
        """Test squeue checks sacct for jobs not in queue."""
        mock_run = make_mock_run({
            "sbatch": MockResult(stdout="Submitted batch job 12345"),
            "squeue": MockResult(stdout=""),
            "sacct": MockResult(stdout="12345|FAILED"),
        })
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))
        statuses = configured_client.squeue()

        assert job_id in statuses
        assert statuses[job_id] == "FAILED"

    def test_squeue_sacct_returns_various_states(self, configured_client, tmp_path, monkeypatch):
        """Test sacct parsing for various job states."""
        mock_run = make_mock_run({
            "sbatch": MockResult(stdout="Submitted batch job 10000"),
            "squeue": MockResult(stdout=""),
            "sacct": MockResult(stdout="10000_0|CANCELLED by 12345\n10000_1|COMPLETED\n10000_2|FAILED\n10000_3|TIMEOUT"),
        })
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        log_paths = [str(tmp_path / f"log_{i}.txt") for i in range(4)]
        configured_client.submit_array(
            "test_array",
            str(tmp_path / "array.sbatch"),
            log_paths,
            ["task0", "task1", "task2", "task3"],
        )

        statuses = configured_client.squeue()

        assert statuses.get("10000_0") == "CANCELLED"
        assert statuses.get("10000_1") == "COMPLETED"
        assert statuses.get("10000_2") == "FAILED"
        assert statuses.get("10000_3") == "TIMEOUT"

    def test_squeue_sacct_failure(self, configured_client, tmp_path, monkeypatch):
        """Test squeue when sacct also fails."""
        mock_run = make_mock_run({
            "sbatch": MockResult(stdout="Submitted batch job 12345"),
            "squeue": MockResult(returncode=1, stderr="Error"),
            "sacct": MockResult(returncode=1, stderr="sacct error"),
        })
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        configured_client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))
        statuses = configured_client.squeue()

        assert statuses == {}

    def test_check_sacct_for_missing_jobs_empty_list(self, configured_client):
        """Test _check_sacct_for_missing_jobs with empty list."""
        result = configured_client._check_sacct_for_missing_jobs([], {})
        assert result == {}

    def test_squeue_with_empty_and_malformed_lines(self, configured_client, tmp_path, monkeypatch):
        """Test squeue handles empty lines and lines without state."""
        mock_run = make_mock_run({
            "sbatch": MockResult(stdout="Submitted batch job 12345"),
            "squeue": MockResult(stdout="12345 RUNNING\n\n99999\n"),
        })
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))
        statuses = configured_client.squeue()

        assert statuses[job_id] == "RUNNING"

    def test_squeue_unknown_job_id_logged(self, configured_client, tmp_path, monkeypatch):
        """Test squeue logs warning for unknown job IDs."""
        mock_run = make_mock_run({
            "sbatch": MockResult(stdout="Submitted batch job 12345"),
            "squeue": MockResult(stdout="12345 RUNNING\n99999 PENDING"),
        })
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))
        statuses = configured_client.squeue()

        assert job_id in statuses
        assert "99999" not in statuses

    def test_sacct_with_empty_and_malformed_lines(self, configured_client, tmp_path, monkeypatch):
        """Test sacct handles empty lines and lines with less than 2 parts."""
        mock_run = make_mock_run({
            "sbatch": MockResult(stdout="Submitted batch job 12345"),
            "squeue": MockResult(stdout=""),
            "sacct": MockResult(stdout="12345|COMPLETED\n\nmalformed_line_no_pipe\n99999.batch|COMPLETED"),
        })
        monkeypatch.setattr(slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit("test", str(tmp_path / "job.sh"), str(tmp_path / "log.txt"))
        statuses = configured_client.squeue()

        assert job_id in statuses
        assert statuses[job_id] == "COMPLETED"

class TestSlurmClientConfiguration:
    """Tests for SLURM client configuration."""

    def test_unconfigured_client_raises(self):
        """Test that accessing slurm_config before configure raises."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        with pytest.raises(RuntimeError, match="has not been configured"):
            _ = client.slurm_config

    def test_configure_sets_config(self):
        """Test that configure properly sets the config."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        config = SlurmConfig(
            template_path="templates/base.sbatch",
            script_dir="/tmp/scripts",
            log_dir="/tmp/logs",
        )
        client.configure(config)
        assert client.slurm_config == config
