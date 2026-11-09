"""Staleness of SRW's head: fresh ROLANN on all training data (final slots)
versus the incrementally built head.

    python scripts/srw_refit.py RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from composition_audit import load  # noqa: E402

from datasets.protocol import TaskProtocol  # noqa: E402
from datasets.registry import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    build_datasets,
    evaluation_view,
)
from methods.rolann import ROLANN  # noqa: E402
from runner import make_loader  # noqa: E402


@torch.no_grad()
def zfeats(method, loader):
    zs, ys = [], []
    for images, labels in loader:
        phi0, slots = method._parts(images)
        zs.append(method._z(phi0, slots))
        ys.append(labels.to(method.device))
    return torch.cat(zs), torch.cat(ys)


def main() -> None:
    out = {}
    for arg in sys.argv[1:]:
        run, method, config = load(Path(arg), torch.device("cuda:0"))
        method.slots.eval()
        train, test = build_datasets(
            config["dataset"],
            str(DEFAULT_DATA_ROOT),
            preprocessing=config["preprocessing"],
        )
        protocol = TaskProtocol(
            tuple(config["class_order"]),
            config["initial_classes"],
            config["increment_classes"],
        )
        last = protocol.num_tasks - 1
        classes = protocol.seen_classes(last)
        tr = make_loader(evaluation_view(train), classes, protocol, config, False, last)
        te = make_loader(test, classes, protocol, config, False, last)
        z_tr, y_tr = zfeats(method, tr)
        z_te, y_te = zfeats(method, te)
        actual = float((method.rolann(z_te).argmax(1) == y_te).float().mean() * 100)
        head = ROLANN(method.rolann.ridge)
        size = config["increment_classes"]
        for t in range(protocol.num_tasks):
            mask = (y_tr // size) == t
            head.update(z_tr[mask], y_tr[mask], y_tr[mask].unique())
        refit = float((head(z_te).argmax(1) == y_te).float().mean() * 100)
        out[str(run)] = {"actual": actual, "refit": refit, "staleness": refit - actual}
        print(run, json.dumps(out[str(run)]), flush=True)
    path = ROOT / "research" / "data" / "srw_refit.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(out)
    path.write_text(json.dumps(old, indent=2))


if __name__ == "__main__":
    main()
