"""Inference and storage cost of a trained CIL method (frugality tables).

For each final checkpoint:
* parameters: backbone, method (adapters, heads, snapshots), and the bytes of
  every tensor the method stores (heads, class statistics, covariances...)
* backbone passes per image: patch-embed calls (full passes) and last-block
  calls (tail passes, which also counts trunk-shared multi-slot methods)
* GFLOPs per image (torch.utils.flop_counter on one image)
* latency (batch 1, median), throughput (batch 64), peak inference memory
* GPU energy per 1000 images (NVML; only meaningful alone on the GPU, see
  ``gpu_processes``)

    python scripts/cost_benchmark.py RUN_DIR [RUN_DIR ...] --device 0
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from composition_audit import load  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from utils.energy import measure  # noqa: E402


def tensor_bytes(obj) -> int:
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def count_params(modules) -> int:
    seen, total = set(), 0
    for module in modules:
        for p in module.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


@torch.no_grad()
def benchmark(method, device, batch=64, repeats=10):
    bb = method.backbone
    calls = {"embed": 0, "last": 0}
    hooks = [
        bb.patch_embed.register_forward_hook(
            lambda _m, i, _o: calls.__setitem__("embed", calls["embed"] + len(i[0]))
        ),
        bb.blocks[-1].register_forward_hook(
            lambda _m, i, _o: calls.__setitem__("last", calls["last"] + len(i[0]))
        ),
    ]
    x1 = torch.rand(1, 3, 224, 224, device=device)
    xb = torch.rand(batch, 3, 224, 224, device=device)
    method.logits(x1)  # warm-up (lazy init, cudnn)
    calls.update(embed=0, last=0)
    method.logits(xb)
    passes = {
        "full_passes_per_image": calls["embed"] / batch,
        "block_passes_per_image": calls["last"] / batch,
    }
    for h in hooks:
        h.remove()
    with FlopCounterMode(display=False) as flops:
        method.logits(x1)
    gflops = flops.get_total_flops() / 1e9
    lat = []
    for _ in range(30):
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        method.logits(x1)
        torch.cuda.synchronize(device)
        lat.append((time.perf_counter() - t) * 1000)
    with measure(device) as cost:
        for _ in range(repeats):
            method.logits(xb)
    images = batch * repeats
    return {
        **passes,
        "gflops_per_image": gflops,
        "latency_ms_b1_median": statistics.median(lat[5:]),
        "throughput_img_s_b64": images / cost["seconds"],
        "peak_mem_mb_b64": cost["peak_mem_mb"],
        "energy_j_per_1000_img": (
            cost["energy_j"] / images * 1000 if cost["energy_j"] is not None else None
        ),
        "gpu_processes": cost.get("gpu_processes"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path, default=ROOT / "results/cost_benchmark.json")
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.device}")
    out = json.loads(a.out.read_text()) if a.out.exists() else {}
    for run_arg in a.runs:
        run, method, config = load(run_arg, device)
        method.backbone.eval()
        for module in method.parameter_modules():
            module.eval()
        res = {
            "method": config["method"],
            "backbone_params": count_params([method.backbone]),
            "method_params": count_params(method.parameter_modules()),
            "stored_mb": tensor_bytes(method.state_dict()) / 2**20,
            **benchmark(method, device),
        }
        out[str(run)] = res
        print(
            run,
            json.dumps(
                {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}
            ),
            flush=True,
        )
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2))
        del method
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
