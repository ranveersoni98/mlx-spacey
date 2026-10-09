"""Train the MLX transit classifier on raw MAST light curves.

Labels come from TIC lists, not data dumps:
  --planet-tics / --planet-catalog-n  confirmed planets (PLANET)
  --fp-tics / --fp-catalog-n          false positives, mostly EBs (FALSE_POSITIVE)
  --field-tics                        assumed-field stars (NO_SIGNAL)

Each TIC is fetched live via lightkurve, flattened, BLS-folded
(see data/), then fed to the dual-view CNN in models/.
"""

from __future__ import annotations

import argparse
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from ..data import FIELD_TICS, LABELS, PLANET_TICS, build_batch, iter_batches
from ..models import count_params, create_model


def accuracy(logits: mx.array, y: mx.array) -> float:
    return float(mx.mean(mx.argmax(logits, axis=-1) == y).item())


def evaluate(model, batch, batch_size=32) -> dict:
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for g, l, s, y in iter_batches(batch, batch_size, shuffle=False):
        gm, lm, sm, ym = mx.array(g), mx.array(l), mx.array(s), mx.array(y)
        logits = model(gm, lm, sm)
        loss = mx.mean(nn.losses.cross_entropy(logits, ym))
        mx.eval(loss, logits)
        tot_loss += float(loss.item()) * len(y)
        tot_acc += accuracy(logits, ym) * len(y)
        n += len(y)
    return {"loss": tot_loss / max(n, 1), "acc": tot_acc / max(n, 1)}


def train(
    planet_tics=None,
    fp_tics=None,
    field_tics=None,
    planet_catalog_n: int = 0,
    fp_catalog_n: int = 0,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    out_dir: str = "checkpoints",
    mission: str = "TESS",
    download_dir: str = "data/raw",
    seed: int = 0,
):
    planet_tics = list(planet_tics or PLANET_TICS)
    fp_tics = list(fp_tics or [])
    field_tics = list(field_tics or FIELD_TICS)
    if planet_catalog_n or fp_catalog_n:
        from ..data.catalog import select_targets

        if planet_catalog_n:
            planet_tics += select_targets(kind="planet", n=planet_catalog_n)
        if fp_catalog_n:
            fp_tics += select_targets(kind="fp", n=fp_catalog_n)
    # dedupe: catalog may overlap curated lists
    def _dedup(ids):
        return list(dict.fromkeys(int(t) for t in ids))

    planet_tics, fp_tics, field_tics = _dedup(planet_tics), _dedup(fp_tics), _dedup(field_tics)
    fp_tics = [t for t in fp_tics if t not in planet_tics]
    field_tics = [t for t in field_tics if t not in planet_tics and t not in fp_tics]
    np.random.seed(seed)
    mx.random.seed(seed)

    tics = planet_tics + fp_tics + field_tics
    labels = (
        [LABELS["PLANET"]] * len(planet_tics)
        + [LABELS["FALSE_POSITIVE"]] * len(fp_tics)
        + [LABELS["NO_SIGNAL"]] * len(field_tics)
    )
    print(f"[data] {len(planet_tics)} planet + {len(fp_tics)} fp + {len(field_tics)} field TICs")
    print(f"[data] fetching {len(tics)} TICs live from MAST ({mission}) ...")
    batch = build_batch(tics, labels, mission=mission, download_dir=download_dir)
    print(f"[data] got {len(batch.labels)} light curves: {batch.star_ids}")

    model = create_model()
    print(f"[model] params={count_params(model):,}")
    opt = optim.AdamW(learning_rate=lr)

    def step(g, l, s, y):
        def _loss(m):
            return mx.mean(nn.losses.cross_entropy(m(g, l, s), y))

        loss_and_grad = nn.value_and_grad(model, _loss)
        loss, grads = loss_and_grad(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        return loss

    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "transit_cnn_best.safetensors")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tot, nb = 0.0, 0
        for g, l, s, y in iter_batches(batch, batch_size, shuffle=True):
            loss = step(mx.array(g), mx.array(l), mx.array(s), mx.array(y))
            mx.eval(loss)
            tot += float(loss.item())
            nb += 1
        ev = evaluate(model, batch)
        dt = time.time() - t0
        print(f"[epoch {epoch:02d}] loss={tot / max(nb, 1):.4f} acc={ev['acc']:.3f} ({dt:.1f}s)")
        if ev["loss"] < best_loss:
            best_loss = ev["loss"]
            model.save_weights(best_path)
            print(f"  -> saved {best_path} (loss {best_loss:.4f})")

    print(f"done. best loss={best_loss:.4f} weights={best_path}")
    return best_path


def _parse_tics(s: str | None):
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description="Train MLX classifier on raw MAST light curves")
    p.add_argument("--planet-tics", default=None, help="comma-separated TIC ids, confirmed planets")
    p.add_argument("--fp-tics", default=None, help="comma-separated TIC ids, false positives")
    p.add_argument("--field-tics", default=None, help="comma-separated TIC ids, assumed field")
    p.add_argument("--planet-catalog-n", type=int, default=0, help="brightest confirmed TOIs to add")
    p.add_argument("--fp-catalog-n", type=int, default=0, help="brightest FP TOIs to add")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--mission", default="TESS")
    p.add_argument("--download-dir", default="data/raw")
    a = p.parse_args(argv)
    train(
        planet_tics=_parse_tics(a.planet_tics),
        fp_tics=_parse_tics(a.fp_tics),
        field_tics=_parse_tics(a.field_tics),
        planet_catalog_n=a.planet_catalog_n,
        fp_catalog_n=a.fp_catalog_n,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        out_dir=a.out_dir,
        mission=a.mission,
        download_dir=a.download_dir,
    )


if __name__ == "__main__":
    main()
