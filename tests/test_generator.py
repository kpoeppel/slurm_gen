"""Tests for SBATCH script generation helpers."""

from pathlib import Path

import pytest

from slurm_gen.generator import (
    build_replacements,
    build_sbatch_directives,
    build_srun_args,
    generate_script,
    merge_slurm_config,
)
from slurm_gen.schema import SbatchConfig, SlurmConfig, SrunConfig


def make_config(tmp_path: Path) -> SlurmConfig:
    return SlurmConfig(
        template_path=str(tmp_path / "template.sbatch"),
        script_dir=str(tmp_path / "scripts"),
        log_dir=str(tmp_path / "logs"),
    )


class TestBuildSbatchDirectives:
    """Tests for build_sbatch_directives."""

    def test_builds_directives_with_extras(self, tmp_path: Path):
        config = make_config(tmp_path)
        config.sbatch = SbatchConfig(
            account="acct",
            nodes=True,
            partition="gpu",
            time="1-00:00:00",
        )
        config.sbatch_extra_directives = [
            "--mail-type=END",
            "#SBATCH --exclusive",
        ]

        directives = build_sbatch_directives(config)

        assert "#SBATCH --account=acct" in directives
        assert "#SBATCH --nodes" in directives
        assert "#SBATCH --partition=gpu" in directives
        assert "#SBATCH --time=1-00:00:00" in directives
        assert "#SBATCH --mail-type=END" in directives
        assert "#SBATCH --exclusive" in directives

    def test_job_name_in_sbatch_skips_auto_emit(self, tmp_path: Path):
        config = make_config(tmp_path)
        config.sbatch = SbatchConfig(job_name="explicit-name")

        directives = build_sbatch_directives(config)

        assert "#SBATCH --job-name=explicit-name" in directives
        # config.name ("job") must NOT be auto-emitted as a second job-name directive
        job_name_directives = [d for d in directives if "--job-name=" in d]
        assert len(job_name_directives) == 1

    def test_non_strict_extras_render_as_directives(self, tmp_path: Path):
        """Extra sbatch keys must render as flags, not leak bookkeeping fields."""
        config = make_config(tmp_path)
        config.sbatch = SbatchConfig(nodes=4, exclude="node01,node02")

        directives = build_sbatch_directives(config)

        assert "#SBATCH --exclude=node01,node02" in directives
        assert not any("extras" in d or "non-strict" in d for d in directives)

    def test_exclude_file_is_read_at_render_time(self, tmp_path: Path):
        """The list on disk wins over whatever was resolved earlier."""
        exclude_file = tmp_path / "excluded_nodes.txt"
        exclude_file.write_text("# bad nodes\nnode07\nnode09 node11\n")
        config = make_config(tmp_path)
        config.sbatch = SbatchConfig(exclude="stale-node")
        config.exclude_file = str(exclude_file)

        directives = build_sbatch_directives(config)

        assert "#SBATCH --exclude=node07,node09,node11" in directives

    def test_missing_exclude_file_keeps_resolved_exclusions(self, tmp_path: Path):
        """A bad path must never silently drop the exclusions we already have."""
        config = make_config(tmp_path)
        config.sbatch = SbatchConfig(exclude="node01")
        config.exclude_file = str(tmp_path / "does-not-exist.txt")

        directives = build_sbatch_directives(config)

        assert "#SBATCH --exclude=node01" in directives


class TestBuildSrunArgs:
    """Tests for build_srun_args."""

    def test_renders_flags_and_values(self, tmp_path: Path):
        config = make_config(tmp_path)
        config.srun_args = SrunConfig(cpu_bind="cores", exclusive=True, kill_on_bad_exit=0)

        args = build_srun_args(config)

        assert "--cpu-bind=cores" in args
        assert "--exclusive" in args
        # 0 is a value, not a boolean: it must survive as --flag=0
        assert "--kill-on-bad-exit=0" in args

    def test_false_and_none_are_skipped(self, tmp_path: Path):
        """False turns off a flag an inherited config group set."""
        config = make_config(tmp_path)
        config.srun_args = SrunConfig(exclusive=False, label=None, wait=60)

        args = build_srun_args(config)

        assert args == ["--wait=60"]

    def test_legacy_srun_block_is_not_rendered(self, tmp_path: Path):
        """``srun`` is legacy and deliberately unrendered; only ``srun_args`` is."""
        config = make_config(tmp_path)
        config.srun = SrunConfig(wait=60)

        assert build_srun_args(config) == []


class TestBuildReplacements:
    """Tests for build_replacements."""

    def test_builds_expected_replacements(self, tmp_path: Path):
        config = make_config(tmp_path)
        config.env = {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "8"}
        config.launcher_cmd = "srun"
        config.srun_opts = "--cpu-bind=cores"
        config.launcher_env_passthrough = True

        replacements = build_replacements(
            config,
            job_name="job",
            log_path="/tmp/log.txt",
            command=["python", "train.py"],
            extra_args=["--epochs", "5"],
        )

        assert replacements["job_name"] == "job"
        assert replacements["log_path"] == "/tmp/log.txt"
        assert replacements["command"] == "python train.py --epochs 5"
        assert replacements["launcher_cmd"] == "srun"
        # Templates render `srun {srun_opts}bash -c ...`, so the value carries
        # its own trailing space.
        assert replacements["srun_opts"] == "--cpu-bind=cores "
        assert replacements["launcher_env_passthrough"] == "true"
        assert "export CUDA_VISIBLE_DEVICES=0" in replacements["env_exports"]
        assert "export OMP_NUM_THREADS=8" in replacements["env_exports"]

    def test_srun_args_render_before_srun_opts(self, tmp_path: Path):
        config = make_config(tmp_path)
        config.srun_args = SrunConfig(kill_on_bad_exit=0)
        config.srun_opts = "  --cpu-bind=cores  "

        replacements = build_replacements(
            config, job_name="job", log_path="/tmp/log.txt", command=["true"]
        )

        assert replacements["srun_opts"] == "--kill-on-bad-exit=0 --cpu-bind=cores "

    def test_empty_srun_opts_stays_empty(self, tmp_path: Path):
        """No flags means no stray space before the command in the template."""
        config = make_config(tmp_path)

        replacements = build_replacements(
            config, job_name="job", log_path="/tmp/log.txt", command=["true"]
        )

        assert replacements["srun_opts"] == ""


class TestGenerateScript:
    """Tests for generate_script."""

    def test_generates_script_in_script_dir(self, tmp_path: Path):
        config = make_config(tmp_path)
        template_path = Path(config.template_path)
        template_path.write_text("#!/bin/bash\n{sbatch_directives}\n{env_exports}\n{command}\n")

        script_path = generate_script(
            config,
            job_name="demo",
            log_path=str(tmp_path / "logs" / "demo.log"),
            command=["python", "train.py"],
        )

        assert Path(script_path).exists()
        contents = Path(script_path).read_text()
        assert "#SBATCH --time=0-01:00:00" in contents
        assert "python train.py" in contents

    def test_generates_script_in_custom_script_path(self, tmp_path: Path):
        config = make_config(tmp_path)
        template_path = Path(config.template_path)
        template_path.write_text("#!/bin/bash\n{command}\n")
        custom_path = str(tmp_path / "custom" / "my_script.sbatch")

        script_path = generate_script(
            config,
            job_name="demo",
            script_path=custom_path,
            log_path=str(tmp_path / "logs" / "demo.log"),
            command=["echo", "hello"],
        )

        assert Path(script_path).exists()
        assert "echo hello" in Path(script_path).read_text()

    def test_missing_template_path_raises(self, tmp_path: Path):
        config = make_config(tmp_path)
        config.template_path = ""

        with pytest.raises(ValueError, match="template_path is required"):
            generate_script(
                config,
                job_name="demo",
                log_path=str(tmp_path / "logs" / "demo.log"),
                command=["echo", "hello"],
            )


class TestMergeSlurmConfig:
    """Tests for merge_slurm_config."""

    def test_merge_none_inputs(self):
        assert merge_slurm_config(None, None) == {}
        assert merge_slurm_config(None, {"a": 1}) == {"a": 1}
        assert merge_slurm_config({"a": 1}, None) == {"a": 1}

    def test_merge_nested_dicts(self):
        base = {"sbatch": {"nodes": 2, "partition": "gpu"}, "array": True}
        override = {"sbatch": {"nodes": 4}, "array": False}
        merged = merge_slurm_config(base, override)
        assert merged["sbatch"]["nodes"] == 4
        assert merged["sbatch"]["partition"] == "gpu"
        assert merged["array"] is False
