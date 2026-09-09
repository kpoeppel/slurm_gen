"""Node-exclusion lists.

A node-exclusion list is a plain text file that grows over time - typically
appended to by whatever notices a node misbehaving - and is read back whenever a
job script is rendered, so a job never lands on a node that has already been
ruled out.

Kept in its own module (rather than inside :mod:`slurm_gen.generator`) because
the same file is read by the monitor when it refreshes a job's exclusions before
a resubmission.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def read_exclude_nodes(path: str | Path, sep: str = ",") -> str | None:
    """Read a node-exclusion list file and return a SLURM nodelist string.

    The file is expected to hold one node name per line; blank lines and lines
    starting with ``#`` are ignored, and entries may also be comma- or
    whitespace-separated within a line. Duplicates are dropped while preserving
    first-seen order. The resulting nodes are joined with ``sep`` (default
    ``","``) so the value can be dropped straight into ``#SBATCH --exclude=``.

    A missing or effectively empty file yields ``None`` so the caller omits the
    ``--exclude`` directive entirely rather than emitting an empty one.

    Args:
        path: Path to the exclusion list. ``~`` is expanded.
        sep: Separator for the returned node list.

    Returns:
        The joined node list, or ``None`` when there is nothing to exclude.
    """
    target = Path(str(path)).expanduser()
    if not target.exists():
        # WARN, do not fail. Returning None is correct - a site with no list
        # should not have an --exclude directive - but staying silent about it
        # is not: the list usually lives outside the repo, so a moved, renamed
        # or archived file, or an unmounted filesystem, turns into a large job
        # submitted with NO exclusions at all, and nothing anywhere says so. A
        # typo in the path looks identical to "we deliberately have no list".
        LOGGER.warning(
            "read_exclude_nodes: %s does not exist - no --exclude directive will be "
            "emitted for this job. If a list was expected, check the path.",
            target,
        )
        return None
    nodes: list[str] = []
    seen: set[str] = set()
    for line in target.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for token in re.split(r"[,\s]+", line):
            token = token.strip()
            if token and token not in seen:
                seen.add(token)
                nodes.append(token)
    if not nodes:
        return None
    return sep.join(nodes)


__all__ = ["read_exclude_nodes"]
