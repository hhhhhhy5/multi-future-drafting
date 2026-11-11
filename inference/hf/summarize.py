"""Tau and speedup tables from benchmark.py outputs.

Every benchmark file holds, per prompt, the autoregressive baseline and each "draft|method" run back to back on one
GPU. Per task: tau = accepted tokens per verification round (including the bonus token, pooled over the task's
rounds); speedup = the method's output tokens per second / the baseline's in the same file(s). Rows are means over the
tasks (macro average, as in the paper); stage times are ms per round (draft / tree build / verify).

  python summarize.py runs/qwen3-4b/*.t0*.pt

File names <task>.<anything>.pt group by task (the part before the first dot); shards of one task (benchmark.py
--shard) are pooled.
"""

import argparse
import collections
import os

import torch

STAGES = ["draft", "tree_build", "verify"]


def summarize(paths):
    out = {}
    for response in (r for path in paths for r in torch.load(path, weights_only=False)["responses"]):
        for key, o in response.items():
            r = out.setdefault(key, {"acc": 0, "rounds": 0, "tokens": 0, "time": 0.0, **{s: 0.0 for s in STAGES}})
            r["acc"] += sum(o.acceptance_lengths)
            r["rounds"] += len(o.acceptance_lengths)
            r["tokens"] += o.num_output_tokens
            r["time"] += o.num_output_tokens * o.time_per_output_token
            for s in STAGES:
                r[s] += (getattr(o, "stage_times", None) or {}).get(s, 0.0)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    by_task = collections.defaultdict(list)
    for path in args.files:
        by_task[os.path.basename(path).split(".")[0]].append(path)
    data = {task: summarize(paths) for task, paths in sorted(by_task.items())}
    keys = list(dict.fromkeys(k for d in data.values() for k in d if "|" in k))  # first-seen order (the CLI order)
    print(f"{len(data)} tasks: " + ", ".join(f"{t} ({d['baseline']['rounds']} AR tokens)" for t, d in data.items()))
    print("\n| draft | method | tau | speedup | draft ms | tree ms | verify ms |")
    print("|---|---|---|---|---|---|---|")
    for k in keys:
        rows = [(d[k], d["baseline"]) for d in data.values() if k in d]
        tau = sum(r["acc"] / r["rounds"] for r, _ in rows) / len(rows)
        speedup = sum((r["tokens"] / r["time"]) / (b["tokens"] / b["time"]) for r, b in rows) / len(rows)
        ms = [1e3 * sum(r[s] for r, _ in rows) / sum(r["rounds"] for r, _ in rows) for s in STAGES]
        name, method = k.split("|")
        print(f"| {name} | {method} | {tau:.3f} | {speedup:.2f}x | " + " | ".join(f"{x:.2f}" for x in ms) + " |")
    print("\nper-task speedup (tau):\n")
    print("| draft|method | " + " | ".join(data) + " |")
    print("|---" * (len(data) + 1) + "|")
    for k in keys:
        cells = []
        for d in data.values():
            if k not in d:
                cells.append("-")
                continue
            r, b = d[k], d["baseline"]
            cells.append(f"{(r['tokens'] / r['time']) / (b['tokens'] / b['time']):.2f}x ({r['acc'] / r['rounds']:.2f})")
        print(f"| {k} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
