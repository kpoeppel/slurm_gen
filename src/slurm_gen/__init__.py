"""slurm_gen - SLURM job generation and management library.

This library provides:
- SLURM client implementations for job submission and management
- Template rendering for SBATCH scripts
- Script validation utilities
- Configuration schema for SLURM settings

Example usage:
    from slurm_gen import FakeSlurmClient, FakeSlurmClientConfig, SlurmConfig

    # Create a fake client for testing
    client = FakeSlurmClient(FakeSlurmClientConfig())
    client.configure(SlurmConfig(
        template_path="templates/job.sbatch",
        script_dir="/tmp/scripts",
        log_dir="/tmp/logs",
    ))

    # Submit a job
    job_id = client.submit("my_job", "/path/to/script.sh", "/path/to/log.txt")
    print(f"Submitted job: {job_id}")

    # Check status
    statuses = client.squeue()
    print(f"Job status: {statuses[job_id]}")
"""

from slurm_gen.client import (
    BaseSlurmClient,
    BaseSlurmClientConfig,
    FakeSlurmClient,
    FakeSlurmClientConfig,
    SlurmClient,
    SlurmClientConfig,
    SlurmJob,
    FakeSlurm,
)
from slurm_gen.schema import (
    SlurmClientInterface,
    SlurmConfig,
    SrunConfig,
    SbatchConfig,
)
from slurm_gen.template_renderer import (
    render_template,
    render_template_file,
    SbatchTemplateError,
)
from slurm_gen.validator import (
    validate_job_script,
    SlurmValidationError,
)

__version__ = "0.1.0"

__all__ = [
    # Client classes
    "BaseSlurmClient",
    "BaseSlurmClientConfig",
    "FakeSlurmClient",
    "FakeSlurmClientConfig",
    "SlurmClient",
    "SlurmClientConfig",
    "SlurmJob",
    "FakeSlurm",
    # Schema
    "SlurmClientInterface",
    "SlurmConfig",
    "SrunConfig",
    "SbatchConfig",
    # Template rendering
    "render_template",
    "render_template_file",
    "SbatchTemplateError",
    # Validation
    "validate_job_script",
    "SlurmValidationError",
]
