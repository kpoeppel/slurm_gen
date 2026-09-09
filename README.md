# slurm_gen

SLURM job generation and management library for Python.

## Features

- **SLURM Client Implementations**: Both real (`SlurmClient`) and fake (`FakeSlurmClient`) implementations for job submission
- **Template Rendering**: Generate SBATCH scripts from templates with variable substitution
- **Script Validation**: Validate rendered scripts for common issues
- **Configuration Schema**: Type-safe configuration using `compoconf`

## Installation

```bash
pip install slurm_gen
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

### Basic Job Submission (Testing)

```python
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
```

### Real SLURM Submission

```python
from slurm_gen import SlurmClient, SlurmClientConfig, SlurmConfig

# Create a real SLURM client
client = SlurmClient(SlurmClientConfig())
client.configure(SlurmConfig(
    template_path="templates/job.sbatch",
    script_dir="./scripts",
    log_dir="./logs",
    submit_cmd="sbatch",
    squeue_cmd="squeue",
))

# Submit a job
job_id = client.submit("training_job", "./scripts/train.sbatch", "./logs/train.log")
```

### Template Rendering

```python
from slurm_gen import render_template, render_template_file

# Render a template string
template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}

python train.py --epochs {epochs}
"""

rendered = render_template(template, {
    "job_name": "training",
    "nodes": "2",
    "epochs": "100",
})

# Or render from a file
render_template_file(
    "templates/job.sbatch",
    "scripts/my_job.sbatch",
    {"job_name": "my_job", "nodes": "4"},
)
```

### Script Validation

```python
from slurm_gen import validate_job_script, SlurmValidationError

script = """#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=2

python train.py
"""

try:
    validate_job_script(script, "my_job", required_tokens=["python train.py"])
    print("Script is valid!")
except SlurmValidationError as e:
    print(f"Validation failed: {e}")
```

### Job Arrays

```python
from slurm_gen import FakeSlurmClient, FakeSlurmClientConfig

client = FakeSlurmClient(FakeSlurmClientConfig())
client.configure(...)

# Submit an array job
job_ids = client.submit_array(
    array_name="sweep",
    script_path="./scripts/array.sbatch",
    log_paths=["./logs/task_0.log", "./logs/task_1.log", "./logs/task_2.log"],
    task_names=["lr_0.01", "lr_0.001", "lr_0.0001"],
)

# job_ids will be like ["1_0", "1_1", "1_2"]
```

## API Reference

### Clients

- `BaseSlurmClient`: Abstract base class for SLURM clients
- `SlurmClient`: Real SLURM client using system commands
- `FakeSlurmClient`: In-memory simulator for testing

### Configuration

- `SlurmConfig`: Main configuration for SLURM settings
- `SlurmClientConfig`: Configuration for real SLURM client
- `FakeSlurmClientConfig`: Configuration for fake client
- `SbatchConfig`: SBATCH directive configuration (non-strict: any extra key
  becomes an `#SBATCH --key=value` directive)
- `SrunConfig`: srun options configuration

### Script generation

- `build_sbatch_directives()`: Render `sbatch` into `#SBATCH` directives
- `build_srun_args()`: Render `SlurmConfig.srun_args` into srun flags
- `build_replacements()`: Build the template placeholder mapping
- `generate_script()`: Render a `SlurmConfig` into an sbatch script
- `merge_slurm_config()`: Deep-merge two config dictionaries

`SlurmConfig` has two srun fields. `srun_args` is a **mapping**, so Hydra merges
it key-by-key across the defaults list and a cluster group and an experiment can
each contribute flags; it is rendered into the `{srun_opts}` placeholder ahead of
the verbatim `srun_opts` string, and a `False` value removes a flag an inherited
group set. The `srun` field is **legacy and deliberately not rendered** - see
`SrunConfig` for why switching it on would change every cluster at once.

### Node exclusion

- `read_exclude_nodes()`: Parse a node-exclusion list file into a SLURM nodelist

Set `SlurmConfig.exclude_file` to have `--exclude` resolved **from that file at
render time**, overriding any `sbatch.exclude` baked in earlier. This matters for
restarts: the script is re-rendered but the config was resolved at plan time, so
without it a node excluded after the first submission would never reach the
resubmitted job. A missing or empty file leaves any existing `sbatch.exclude`
untouched, so a bad path can never silently drop the exclusions.

`SlurmClient.update_excludes(job_id, nodelist)` applies a list to a job that is
already queued, via `scontrol update JobId=.. ExcNodeList=..`. Only *pending*
jobs can be edited live.

### Template & Validation

- `render_template()`: Render a template string
- `render_template_file()`: Render a template file to output
- `validate_job_script()`: Validate a rendered script
- `SbatchTemplateError`: Raised on template errors
- `SlurmValidationError`: Raised on validation errors

## Testing

```bash
pytest tests/ -v
```

## License and Attribution

Copyright 2026 Korbinian Poeppel.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
these files except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the [LICENSE](LICENSE)
file for the specific language governing permissions and limitations under the
License.

This library is derived from `oellm_autoexp/slurm_gen` in
[OpenEuroLLM/oellm-autoexp](https://github.com/OpenEuroLLM/oellm-autoexp),
Copyright 2026 OpenEuroLLM Consortium, also licensed under Apache 2.0. It is
maintained here as a standalone package and is periodically re-synced with
upstream.
