"""Emit the sbatch flags for one judge, from the config."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Host RAM a node may be given, per partition.
MEM_CAP_GIB = {
    "gpu_h100": 760,
    "gpu_h100_short": 760,
    "dev_gpu_h100": 760,
    "gpu_a100_il": 510,
    "gpu_h100_il": 510,
    "dev_gpu_a100_il": 510,
    "gpu_mi300": 510,
    # Cascade Lake A100 nodes carry only 384 GiB of host RAM.
    "gpu_a100_short": 370,
}

#: Total GPU memory on one node, per partition.
VRAM_PER_NODE_GIB = {
    "gpu_h100": 4 * 94,
    "gpu_h100_short": 4 * 94,
    "dev_gpu_h100": 4 * 94,
    "gpu_a100_il": 4 * 80,
    "gpu_h100_il": 4 * 80,
    "dev_gpu_a100_il": 4 * 80,
    # Cascade Lake: 40 GB cards. 160 GB a node holds neither judge.
    "gpu_a100_short": 4 * 40,
    "gpu_mi300": 4 * 128,
}

#: Partitions whose GPUs are Ampere or older, so have no FP8 kernel.
NO_FP8 = {"gpu_a100_il", "gpu_a100_short", "dev_gpu_a100_il", "gpu_mi300"}

#: Max walltime per partition, in minutes, from `scontrol show partition`.
MAX_TIME_MIN = {
    "gpu_h100": 3 * 24 * 60,
    "gpu_h100_short": 30,
    "dev_gpu_h100": 30,
    "gpu_a100_il": 2 * 24 * 60,
    "gpu_h100_il": 2 * 24 * 60,
    "gpu_a100_short": 30,
    "dev_gpu_a100_il": 30,
}


def _minutes(spec: str) -> int:
    """HH:MM:SS or D-HH:MM:SS -> minutes."""
    days = 0
    if "-" in spec:
        head, spec = spec.split("-", 1)
        days = int(head)
    parts = [int(p) for p in spec.split(":")]
    while len(parts) < 3:
        parts.append(0)
    hours, minutes, seconds = parts
    return days * 1440 + hours * 60 + minutes + (1 if seconds else 0)


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    name = argv[1]
    partition_override = argv[2] if len(argv) > 2 else ""

    import yaml
    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    # Generators (Experiment 3) are served the same way, so they are looked up here too.
    models = config["judges"] + config.get("generators", [])
    for judge in models:
        if judge["name"] == name:
            break
    else:
        sys.exit(f"no judge or generator named {name!r}; config has "
                 f"{', '.join(j['name'] for j in models)}")

    partition = partition_override or judge.get("partition", "gpu_h100")
    # A data-parallel generator needs tp x dp cards, not tp.
    gpus = int(judge.get("gpus") or judge.get("tensor_parallel", 4))
    # A short queue caps walltime well below the config default, and asking for more is
    # rejected.
    time = os.environ.get("JV_TIME") or judge.get("time", "08:00:00")
    mem = str(judge.get("mem", "480gb"))

    if judge.get("quantization") == "fp8" and partition in NO_FP8:
        sys.exit(f"{name} is FP8 and {partition} has no FP8 kernel (Ampere or "
                 f"older). Use a gpu_h100* partition, or set dtype: bfloat16 "
                 f"in the config and check it still fits.")

    vram = VRAM_PER_NODE_GIB.get(partition)
    weights = int(judge.get("approx_weight_gib", 0))
    if vram and weights:
        # Weights alone, before any KV cache. gpu_memory_utilization then has to leave
        # room on top, so anything above ~85% of the node is not viable.
        if weights > vram * 0.85:
            sys.exit(f"{name} needs ~{weights} GiB of weights but {partition} "
                     f"has {vram} GiB per node ({gpus} GPUs). It will not fit "
                     f"with room for a KV cache. Use a partition with more VRAM.")

    limit = MAX_TIME_MIN.get(partition)
    if limit and _minutes(time) > limit:
        hours, mins = divmod(limit, 60)
        sys.exit(f"{name} asks for {time} but {partition} caps a job at "
                 f"{hours}h{mins:02d}m. Either lower it with JV_TIME "
                 f"(the store resumes, so a short job that runs out is not "
                 f"lost work), or use a partition with a longer limit.")

    cap = MEM_CAP_GIB.get(partition)
    if cap:
        want = int("".join(c for c in mem if c.isdigit()) or 0)
        if want > cap:
            sys.exit(f"{name} asks for {mem} but {partition} caps a node at "
                     f"{cap} GB. Lower `mem:` for this judge in the config.")

    print(f"-p {partition} --gres=gpu:{gpus} --mem={mem} -t {time} -J jv-{name}")


if __name__ == "__main__":
    main(sys.argv)
