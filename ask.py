"""
Ask your model (and optionally Jev) one question and see the answer + probabilities.

    python ask.py examples/charge-dispute.json
    python ask.py examples/charge-dispute.json --jev      # also ask TypeSafe Jev
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

import bench


def bar(p, width=30):
    return "#" * round(p * width) + "." * (width - round(p * width))


def show(title, probs, ms):
    print(f"\n{title}  ({ms:.0f} ms)")
    for key, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<22} {bar(p)} {p:6.1%}")


async def main():
    bench.load_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", type=Path)
    p.add_argument("--jev", action="store_true")
    p.add_argument("--label", default="My Jev (Unsloth 4B)")
    a = p.parse_args()
    task = json.loads(a.task.read_text(encoding="utf-8"))
    t = {"task": {k: task[k] for k in ("state", "question", "options")}}
    print("STATE:   ", str(task["state"])[:200])
    print("QUESTION:", task["question"])
    async with httpx.AsyncClient(timeout=60) as client:
        await bench.call_mine(client, t)  # warm the connection
        start = time.perf_counter()
        probs, _, _ = await bench.call_mine(client, t)
        show(a.label, probs, (time.perf_counter() - start) * 1000)
        if a.jev:
            await bench.call_jev(client, t)
            start = time.perf_counter()
            probs, _, _ = await bench.call_jev(client, t)
            show("TypeSafe Jev", probs, (time.perf_counter() - start) * 1000)


if __name__ == "__main__":
    asyncio.run(main())
