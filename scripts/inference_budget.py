"""E1: how much of TUNA's / EASE's accuracy needs their multi-pass inference?

Inference only, on the final checkpoint of a run.  Each adapter's test
features are computed once; the variants below recombine them with the
method's own trained heads (nothing is refit).  ``passes`` = ViT forward
passes per image.

TUNA (T snapshots + EMR-merged adapter):
  full            merged softmax + lowest-entropy snapshot softmax  (T+1)
  merged          merged adapter only                               (1)
  last            last snapshot (= the adapter after the last task) (1)
  merged+last     softmax sum of the two                            (2)
  specific        lowest-entropy snapshot only                      (T)
  oracle_specific snapshot of the true task (diagnostic upper bound)
EASE (one adapter per task, concatenated subspaces):
  full            EASE's reweighted cross-subspace scoring          (T)
  first           cosine on adapter 0's subspace (real prototypes
                  of every class: first-session adaptation + NCM)   (1)
  last            cosine on the last adapter's subspace             (1)

    python scripts/inference_budget.py --run-dir RUN [RUN ...] --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from scripts.composition_audit import load  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from datasets.protocol import TaskProtocol  # noqa: E402
from datasets.registry import DEFAULT_DATA_ROOT, build_datasets  # noqa: E402
from runner import make_loader  # noqa: E402
from utils.reproducibility import parse_device  # noqa: E402


def entropy(logit):
    prob = F.softmax(logit, dim=1)
    return -(prob * torch.log(prob + 1e-10)).sum(1)


@torch.no_grad()
def tuna_variants(m, loader):
    total, T = sum(m.task_sizes), len(m.snapshots)
    out = {
        k: []
        for k in [
            "full",
            "merged",
            "last",
            "merged+last",
            "specific",
            "oracle_specific",
        ]
    }
    labels = []
    for images, y in loader:
        y = y.to(m.device)
        spec = torch.stack(
            [m._head(m._features(images, s))[:, :total] * m.scale for s in m.snapshots]
        )
        merged = m._head(m._features(images, m.merged))[:, :total] * m.scale
        idx = torch.arange(len(y), device=y.device)
        chosen = spec[torch.stack([entropy(s) for s in spec]).argmin(0), idx]
        task = torch.bucketize(
            y, torch.tensor(m.task_sizes, device=y.device).cumsum(0), right=True
        )
        pm, pl = F.softmax(merged, 1), F.softmax(spec[-1], 1)
        preds = {
            "full": pm + F.softmax(chosen, 1),
            "merged": merged,
            "last": spec[-1],
            "merged+last": pm + pl,
            "specific": chosen,
            "oracle_specific": spec[task, idx],
        }
        for k, v in preds.items():
            out[k].append(v.argmax(1))
        labels.append(y)
    passes = {
        "full": T + 1,
        "merged": 1,
        "last": 1,
        "merged+last": 2,
        "specific": T,
        "oracle_specific": 1,
    }
    return out, torch.cat(labels), passes


@torch.no_grad()
def ease_variants(m, loader):
    T = len(m.task_sizes)
    w = m.classifier.weight
    out = {k: [] for k in ["full", "first", "last"]}
    labels = []
    for images, y in loader:
        y = y.to(m.device)
        sub = m._subspaces(images)
        full = m._reweighted_logits(sub) if m.use_reweight else m._cosine_logits(sub, w)
        out["full"].append(full.argmax(1))
        for k, t in [("first", 0), ("last", T - 1)]:
            cols = m._subspace_slice(t)
            out[k].append(m._cosine_logits(sub[:, cols], w[:, cols]).argmax(1))
        labels.append(y)
    return out, torch.cat(labels), {"full": T, "first": 1, "last": 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument(
        "--out", type=Path, default=ROOT / "research" / "data" / "inference_budget.json"
    )
    a = ap.parse_args()
    device = parse_device(a.device)
    results = json.loads(a.out.read_text()) if a.out.exists() else {}
    for run_dir in a.run_dir:
        run, method, config = load(run_dir, device)
        method.backbone.eval()
        for module in method.parameter_modules():
            module.eval()
        _, test = build_datasets(
            config["dataset"],
            str(DEFAULT_DATA_ROOT),
            preprocessing=config.get("preprocessing", "canonical"),
        )
        protocol = TaskProtocol(
            tuple(config["class_order"]),
            config["initial_classes"],
            config["increment_classes"],
        )
        last = protocol.num_tasks - 1
        loader = make_loader(
            test, protocol.seen_classes(last), protocol, config, False, last
        )
        fn = {"tuna": tuna_variants, "ease": ease_variants}[config["method"]]
        preds, y, passes = fn(method, loader)
        acc = {
            k: round(float((torch.cat(v) == y).float().mean() * 100), 2)
            for k, v in preds.items()
        }
        logged = json.loads((run / "metrics.json").read_text())["final_accuracy"]
        res = {
            "method": config["method"],
            "accuracy": acc,
            "passes": passes,
            "logged_final": round(logged, 2),
        }
        print(run_dir, json.dumps(res), flush=True)
        if abs(acc["full"] - logged) > 0.05:
            print(f"WARNING full {acc['full']} != logged {logged}", flush=True)
        results[str(run_dir)] = res
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
