"""Benchmark client for a running SGLang server (launch_server.sh): throughput, latency and acceptance length, and
an output comparison against a reference run (T = 0 losslessness check against autoregressive decoding).

Prompts: one of the paper's tasks (gsm8k, math500, humaneval, mbpp, mt-bench; the HF harness's prompt formats, first
turn only), shuffled with seed 0 and cut to --num-prompts as the HF benchmark does, or --prompts-file (a JSON list
of strings). The target's chat template is applied here (--thinking for thinking mode) and token ids are sent to
/generate. --concurrency C keeps C requests in flight (closed loop).

  python inference/sglang/bench_client.py --url http://127.0.0.1:30000 --model Qwen/Qwen3-4B --task gsm8k \
      --num-prompts 32 --concurrency 1 --temperature 0 --max-new-tokens 512 --out runs/mfd.json \
      [--top-logprobs 2] [--compare runs/ar.json]

Summary: output tokens / wall time (all requests), mean latency per request and per output token, and
tau = 1 + accepted draft tokens / verification rounds (pooled; speculative engines only). --top-logprobs N stores
each step's top-N log-probabilities (run it on the autoregressive reference). --compare REF.json reports, per prompt,
whether the outputs are identical and, where they are not, the first differing position; when REF holds top
log-probabilities, a divergence counts as a near-tie if the other run's token is in the reference's top-N there with
a log-probability within --tie-margin nats of the reference token (bf16 logits of magnitude 16-32 are spaced 0.125
apart, so verify and decode kernels flip such ties).
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
HF_UTILS = os.path.join(HERE, "..", "hf", "model", "utils.py")


def task_prompts(task, num_prompts):
    spec = importlib.util.spec_from_file_location("hf_utils", HF_UTILS)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    dataset = utils.load_and_process_dataset(task)
    if num_prompts is not None and len(dataset) > num_prompts:
        dataset = dataset.shuffle(seed=0).select(range(num_prompts))
    return [row["turns"][0] for row in dataset]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--model", required=True, help="tokenizer / chat template (the served target)")
    p.add_argument("--task", default=None)
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--num-prompts", type=int, default=None)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--top-logprobs", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--compare", default=None, help="reference result file (same prompts)")
    p.add_argument("--tie-margin", type=float, default=0.25)
    args = p.parse_args()
    if (args.task is None) == (args.prompts_file is None):
        p.error("give exactly one of --task and --prompts-file")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = task_prompts(args.task, args.num_prompts) if args.task else json.load(open(args.prompts_file))
    if args.prompts_file and args.num_prompts:
        texts = texts[: args.num_prompts]
    prompts = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=args.thinking) for t in texts]
    input_ids = [tok.encode(text, add_special_tokens=False) for text in prompts]

    session = requests.Session()

    def generate(index, ids, max_new_tokens):
        sampling = {"temperature": args.temperature, "max_new_tokens": max_new_tokens}
        body = {"input_ids": ids, "sampling_params": sampling}
        if args.top_logprobs:
            body.update(return_logprob=True, top_logprobs_num=args.top_logprobs)
        t = time.perf_counter()
        resp = session.post(f"{args.url}/generate", json=body, timeout=3600)
        resp.raise_for_status()
        out = resp.json()
        latency = time.perf_counter() - t
        meta = out["meta_info"]
        record = {"index": index, "prompt_tokens": len(ids), "output_ids": out.get("output_ids"),
                  "text": out.get("text"), "completion_tokens": meta["completion_tokens"],
                  "spec_verify_ct": meta.get("spec_verify_ct"),
                  "spec_num_correct_drafts": meta.get("spec_num_correct_drafts"),
                  "finish_reason": meta.get("finish_reason"), "latency": latency}
        if args.top_logprobs:
            record["top_logprobs"] = [[[lp, tid] for lp, tid, *_ in step] for step in meta["output_top_logprobs"]]
        return record

    for i in range(args.warmup):
        generate(-1 - i, input_ids[0], 32)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(pool.map(lambda pair: generate(pair[0], pair[1], args.max_new_tokens), enumerate(input_ids)))
    wall = time.perf_counter() - t0
    tokens = sum(r["completion_tokens"] for r in records)
    summary = {"requests": len(records), "output_tokens": tokens, "wall_s": wall, "throughput_tok_s": tokens / wall,
               "mean_latency_s": sum(r["latency"] for r in records) / len(records),
               "mean_ms_per_output_token": 1e3 * sum(r["latency"] for r in records) / tokens}
    rounds = sum(r["spec_verify_ct"] or 0 for r in records)
    if rounds:
        summary["tau"] = 1 + sum(r["spec_num_correct_drafts"] or 0 for r in records) / rounds
    print(json.dumps(summary))

    if args.compare:
        ref = {r["index"]: r for r in json.load(open(args.compare))["records"]}
        identical = ties = 0
        report = []
        for r in records:
            a, b = ref[r["index"]]["output_ids"], r["output_ids"]
            if a == b:
                identical += 1
                continue
            pos = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            entry = {"index": r["index"], "first_difference": pos, "ref_tokens": len(a), "tokens": len(b)}
            top = ref[r["index"]].get("top_logprobs")
            if top is not None and pos < min(len(a), len(b)) and pos < len(top):
                lps = {int(tid): lp for lp, tid in top[pos]}
                if b[pos] in lps and a[pos] in lps:
                    entry["margin"] = lps[a[pos]] - lps[b[pos]]
                    entry["near_tie"] = entry["margin"] <= args.tie_margin
                else:
                    entry["near_tie"] = False
                ties += entry["near_tie"]
            report.append(entry)
        summary["compare"] = {"reference": args.compare, "identical": identical, "different": len(report),
                              "near_tie_divergences": ties, "divergences": report}
        print(json.dumps({k: v for k, v in summary["compare"].items() if k != "divergences"}))
        for entry in report:
            print("  " + json.dumps(entry))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "summary": summary, "records": records}, f)


if __name__ == "__main__":
    sys.exit(main())
