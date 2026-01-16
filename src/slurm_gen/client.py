"""SLURM client implementations for job submission and management."""

from __future__ import annotations

import logging
import re
import shlex
import time
from dataclasses import MISSING, dataclass, field
from pathlib import Path

from compoconf import ConfigInterface, register

from slurm_gen.schema import SlurmClientInterface, SlurmConfig
from slurm_gen.shell import run_command

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SlurmJob:
    """Metadata tracked for submitted SLURM jobs.

    Attributes:
        job_id: SLURM job ID.
        name: Job name.
        script_path: Path to the job script.
        log_path: Path to the log file.
        state: Current job state.
        return_code: Exit code if completed.
        submitted_at: Timestamp when submitted.
    """

    job_id: str = field(default_factory=MISSING)
    name: str = field(default_factory=MISSING)
    script_path: str = field(default_factory=MISSING)
    log_path: str = field(default_factory=MISSING)
    state: str = "PENDING"
    return_code: int | None = None
    submitted_at: float = field(default_factory=time.time)


class BaseSlurmClient(SlurmClientInterface):
    """Base functionality shared by SLURM client implementations.

    Subclasses must implement:
        - submit(): Submit a single job
        - submit_array(): Submit an array job
        - cancel(): Cancel a job
        - remove(): Remove job from tracking
        - squeue(): Query job statuses
        - job_ids_by_name(): Find jobs by name
        - get_job(): Get job metadata
    """

    config: ConfigInterface
    supports_array: bool = False

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config
        self._slurm_config: SlurmConfig | None = None

    def configure(self, slurm_config: SlurmConfig) -> None:
        """Configure the client with SLURM settings."""
        self._slurm_config = slurm_config

    def submit(self, name: str, script_path: str, log_path: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def submit_array(
        self,
        array_name: str,
        script_path: str,
        log_paths: list[str],
        task_names: list[str],
    ) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def cancel(self, job_id: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def remove(self, job_id: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def squeue(self) -> dict[str, str]:  # pragma: no cover - interface
        raise NotImplementedError

    def job_ids_by_name(self, name: str) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_job(self, job_id: str) -> SlurmJob:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def slurm_config(self) -> SlurmConfig:
        """Get the SLURM configuration, raising if not configured."""
        if self._slurm_config is None:
            raise RuntimeError("SLURM client has not been configured with SlurmConfig")
        return self._slurm_config


@dataclass(kw_only=True)
class BaseSlurmClientConfig(ConfigInterface):
    """Shared configuration fields for SLURM clients."""

    class_name: str = "BaseSlurmClient"
    persist_artifacts: bool = False


@dataclass(kw_only=True)
class FakeSlurmClientConfig(BaseSlurmClientConfig):
    """Configuration for the fake SLURM client."""

    class_name: str = "FakeSlurmClient"
    persist_artifacts: bool = False


@register
class FakeSlurmClient(BaseSlurmClient):
    """In-memory SLURM simulator for testing.

    This client simulates SLURM behavior without actually submitting jobs.
    Useful for unit tests and development without a SLURM cluster.
    """

    config: FakeSlurmClientConfig
    supports_array = True

    def __init__(self, config: FakeSlurmClientConfig) -> None:
        super().__init__(config)
        self._jobs: dict[str, SlurmJob] = {}
        self._next_id = 1

    def submit(self, name: str, script_path: str, log_path: str) -> str:
        job_id = str(self._next_id)
        self._next_id += 1
        job = SlurmJob(job_id=job_id, name=name, script_path=script_path, log_path=log_path)
        job.state = "PENDING"
        self._jobs[job_id] = job
        if self.config.persist_artifacts:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).touch(exist_ok=True)
        return job_id

    def submit_array(
        self,
        array_name: str,
        script_path: str,
        log_paths: list[str],
        task_names: list[str],
        start_index: int = 0,
    ) -> list[str]:
        job_ids: list[str] = []
        base_id = str(self._next_id)
        self._next_id += 1

        for offset, (log_path, task_name) in enumerate(zip(log_paths, task_names)):
            array_idx = start_index + offset
            task_job_id = f"{base_id}_{array_idx}"
            job_name = f"{array_name}_{task_name}"

            job = SlurmJob(
                job_id=task_job_id,
                name=job_name,
                script_path=script_path,
                log_path=log_path,
            )
            job.state = "PENDING"
            self._jobs[task_job_id] = job

            if self.config.persist_artifacts:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                Path(log_path).touch(exist_ok=True)

            job_ids.append(task_job_id)

        return job_ids

    def cancel(self, job_id: str) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].state = "CANCELLED"

    def remove(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def squeue(self) -> dict[str, str]:
        return {job_id: job.state for job_id, job in self._jobs.items()}

    def job_ids_by_name(self, name: str) -> list[str]:
        return [job_id for job_id, job in self._jobs.items() if job.name == name]

    def get_job(self, job_id: str) -> SlurmJob:
        return self._jobs[job_id]

    def set_state(self, job_id: str, state: str, return_code: int | None = None) -> None:
        """Set the state of a job (for testing)."""
        job = self._jobs[job_id]
        job.state = state
        if return_code is not None:
            job.return_code = return_code

    def register_job(
        self,
        job_id: str,
        name: str,
        script_path: str,
        log_path: str,
        state: str = "PENDING",
    ) -> str:
        """Register an externally submitted job for tracking."""
        job = SlurmJob(
            job_id=job_id,
            name=name,
            script_path=script_path,
            log_path=log_path,
            state=state,
        )
        self._jobs[job_id] = job
        try:
            base_id = int(str(job_id).split("_")[0])
        except ValueError:
            base_id = self._next_id
        self._next_id = max(self._next_id, base_id + 1)
        return job_id


@dataclass(kw_only=True)
class SlurmClientConfig(BaseSlurmClientConfig):
    """Configuration for the real SLURM client."""

    class_name: str = "SlurmClient"


@register
class SlurmClient(BaseSlurmClient):
    """SLURM client that executes real SLURM commands.

    This client shells out to sbatch, squeue, scancel, and sacct
    to interact with a real SLURM cluster.
    """

    config: SlurmClientConfig
    supports_array = True

    def __init__(self, config: SlurmClientConfig) -> None:
        super().__init__(config)
        self._jobs: dict[str, SlurmJob] = {}

    def submit(self, name: str, script_path: str, log_path: str) -> str:
        slurm_conf = self.slurm_config
        submit_cmd = shlex.split(slurm_conf.submit_cmd)
        proc = run_command([*submit_cmd, str(script_path)])
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed for {script_path}: {proc.stderr.strip()}")
        job_id = self._parse_job_id(proc.stdout)
        if job_id is None:
            raise RuntimeError(f"Unable to parse job id from sbatch output: {proc.stdout.strip()}")
        if self.config.persist_artifacts:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).touch(exist_ok=True)
        self._jobs[job_id] = SlurmJob(job_id=job_id, name=name, script_path=script_path, log_path=log_path)
        return job_id

    def submit_array(
        self,
        array_name: str,
        script_path: str,
        log_paths: list[str],
        task_names: list[str],
        start_index: int = 0,
    ) -> list[str]:
        """Submit a SLURM job array.

        Args:
            array_name: Base name for the array job.
            script_path: Path to the array script.
            log_paths: List of log paths for each task.
            task_names: List of task names.
            start_index: Starting array index (default 0).

        Returns:
            List of job IDs (one per task).
        """
        slurm_conf = self.slurm_config
        num_tasks = len(task_names)

        submit_cmd = shlex.split(slurm_conf.submit_cmd)
        array_range = f"{start_index}-{start_index + num_tasks - 1}"
        proc = run_command([*submit_cmd, f"--array={array_range}", str(script_path)])

        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed for array {script_path}: {proc.stderr.strip()}")

        base_job_id = self._parse_job_id(proc.stdout)
        if base_job_id is None:
            raise RuntimeError(f"Unable to parse job id from sbatch output: {proc.stdout.strip()}")

        LOGGER.info(
            f"submit_array: submitted array job with base_id={base_job_id}, num_tasks={num_tasks}, "
            f"start_index={start_index}"
        )

        job_ids: list[str] = []
        for offset, (log_path, task_name) in enumerate(zip(log_paths, task_names)):
            array_idx = start_index + offset
            task_job_id = f"{base_job_id}_{array_idx}"
            job_name = f"{array_name}_{task_name}"

            if self.config.persist_artifacts:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                Path(log_path).touch(exist_ok=True)

            job = SlurmJob(
                job_id=task_job_id,
                name=job_name,
                script_path=script_path,
                log_path=log_path,
            )
            self._jobs[task_job_id] = job
            job_ids.append(task_job_id)
            LOGGER.debug(
                f"submit_array: registered task offset={offset}, array_idx={array_idx}: "
                f"synthetic_id={task_job_id}, real_id={base_job_id}_{array_idx}"
            )

        return job_ids

    def cancel(self, job_id: str) -> None:
        cmd = shlex.split(self.slurm_config.cancel_cmd)
        LOGGER.debug(f"cancel: cancelling job_id={job_id}")
        run_command([*cmd, str(job_id)])

    def remove(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def squeue(self) -> dict[str, str]:
        if not self._jobs:
            LOGGER.debug("squeue: no jobs tracked, returning empty")
            return {}

        LOGGER.info(f"squeue: tracking {len(self._jobs)} jobs: {list(self._jobs.keys())}")

        cmd = shlex.split(self.slurm_config.squeue_cmd)
        format_arg = ["--noheader", "--format", "%i %T"]

        job_ids = list(self._jobs.keys())
        job_id_to_key = {str(jid): jid for jid in job_ids}

        job_ids_str = ",".join(str(jid) for jid in job_ids)
        full_cmd = [*cmd, "--jobs", job_ids_str, *format_arg]
        LOGGER.info(f"squeue: querying jobs {job_ids_str} with command: {' '.join(full_cmd)}")

        proc = run_command(full_cmd)
        if proc.returncode != 0:
            LOGGER.warning(f"squeue: command failed with rc={proc.returncode}, stderr={proc.stderr}")
            return self._check_sacct_for_missing_jobs(job_ids, job_id_to_key)

        LOGGER.debug(f"squeue: output: {proc.stdout.strip()}")
        statuses: dict[str, str] = {}

        for line in proc.stdout.strip().splitlines():
            parts = line.strip().split(None, 1)
            if not parts:
                continue

            slurm_id = parts[0]
            state = parts[1] if len(parts) > 1 else "UNKNOWN"

            if slurm_id in job_id_to_key:
                statuses[job_id_to_key[slurm_id]] = state
            else:
                LOGGER.warning(f"squeue: received unknown job ID {slurm_id}")

        missing_jobs = [jid for jid in job_ids if jid not in statuses]
        if missing_jobs:
            LOGGER.info(f"squeue: {len(missing_jobs)} jobs not in queue, checking sacct for recent completion")
            missing_id_to_key = {str(jid): jid for jid in missing_jobs}
            sacct_statuses = self._check_sacct_for_missing_jobs(missing_jobs, missing_id_to_key)
            statuses.update(sacct_statuses)

        LOGGER.debug(f"squeue: parsed statuses: {statuses}")
        return statuses

    def _check_sacct_for_missing_jobs(
        self, job_ids: list[str], job_id_to_key: dict[str, str]
    ) -> dict[str, str]:
        """Check sacct for jobs that are no longer in squeue."""
        if not job_ids:
            return {}

        sacct_cmd = shlex.split(self.slurm_config.sacct_cmd)

        job_ids_str = ",".join(str(jid) for jid in job_ids)
        full_cmd = [
            *sacct_cmd,
            "--jobs",
            job_ids_str,
            "--noheader",
            "--format",
            "JobID,State",
            "--parsable2",
        ]

        LOGGER.info(f"sacct: checking for {len(job_ids)} missing jobs: {' '.join(full_cmd)}")

        proc = run_command(full_cmd)
        if proc.returncode != 0:
            LOGGER.warning(f"sacct: command failed with rc={proc.returncode}, stderr={proc.stderr}")
            return {}

        LOGGER.debug(f"sacct: output: {proc.stdout.strip()}")
        statuses: dict[str, str] = {}

        for line in proc.stdout.strip().splitlines():
            if not line.strip():
                continue

            parts = line.strip().split("|")
            if len(parts) < 2:
                continue

            slurm_id = parts[0].strip()
            state = parts[1].strip()

            if "CANCELLED" in state:
                state = "CANCELLED"
            elif "COMPLETED" in state:
                state = "COMPLETED"
            elif "FAILED" in state:
                state = "FAILED"
            elif "TIMEOUT" in state:
                state = "TIMEOUT"

            if slurm_id in job_id_to_key:
                statuses[job_id_to_key[slurm_id]] = state
                LOGGER.info(f"sacct: found job {slurm_id} in state {state}")
            else:
                LOGGER.debug(f"sacct: ignoring irrelevant job {slurm_id}")

        return statuses

    def job_ids_by_name(self, name: str) -> list[str]:
        return [job_id for job_id, job in self._jobs.items() if job.name == name]

    def get_job(self, job_id: str) -> SlurmJob:
        return self._jobs[job_id]

    def register_job(
        self,
        job_id: str,
        name: str,
        script_path: str,
        log_path: str,
        state: str = "PENDING",
    ) -> str:
        """Register an externally submitted job for tracking."""
        job = SlurmJob(
            job_id=job_id,
            name=name,
            script_path=script_path,
            log_path=log_path,
            state=state,
        )
        self._jobs[job_id] = job
        return job_id

    @staticmethod
    def _parse_job_id(output: str) -> str | None:
        """Parse job ID from sbatch output."""
        for token in output.split():
            if re.match(r"\d+(?:_\d+)?", token):
                return token
        return None


# Backward compatibility alias
FakeSlurm = FakeSlurmClient


__all__ = [
    "BaseSlurmClient",
    "BaseSlurmClientConfig",
    "FakeSlurmClient",
    "FakeSlurmClientConfig",
    "SlurmClient",
    "SlurmClientConfig",
    "SlurmJob",
    "FakeSlurm",
]
