"""Discovery: scan TICs live from MAST, rank by P(PLANET).

  TIC list --(lightkurve)--> raw light curves --> BLS views --> MLX
    --> P(PLANET) ranking --> vetting verdicts

No data dumps anywhere. Output is a small ranked list for readability,
written with the stdlib csv module.
"""

from __future__ import annotations

import argparse
import csv
import os

import mlx.core as mx
import numpy as np

from ..data import INV_LABELS, LABELS, tic_to_inputs
from ..models import create_model
from ..vetting import vet_candidate


def softmax(x: mx.array) -> mx.array:
    x = x - mx.max(x, axis=-1, keepdims=True)
    e = mx.exp(x)
    return e / mx.sum(e, axis=-1, keepdims=True)


def scan_tics(weights_path: str, tic_ids, mission="TESS", download_dir="data/raw"):
    model = create_model()
    model.load_weights(weights_path)
    model.eval()
    tic_ids = [int(t) for t in tic_ids]
    rows = []
    try:
        for n, tic in enumerate(tic_ids, 1):
            print(f"[{n}/{len(tic_ids)}] TIC {tic} ...")
            try:
                g, l, s, meta = tic_to_inputs(tic, mission=mission, download_dir=download_dir)
            except Exception as e:
                print(f"[skip] TIC {tic}: {e}")
                continue
            probs = softmax(model(mx.array(g[None]), mx.array(l[None]), mx.array(s[None])))
            mx.eval(probs)
            p = np.array(probs.tolist(), dtype=np.float64)[0]
            row = {
                "star_id": f"TIC {tic}",
                "p_no_signal": float(p[LABELS["NO_SIGNAL"]]),
                "p_false_positive": float(p[LABELS["FALSE_POSITIVE"]]),
                "p_planet": float(p[LABELS["PLANET"]]),
                "pred_label": INV_LABELS[int(np.argmax(p))],
                "bls_period": meta.get("period", float("nan")),
                "n_points": meta.get("n_points", 0),
            }
            vet = vet_candidate(row, g, g, g, g)
            row["verdict"] = vet["verdict"]
            row["snr"] = round(max(vet["snr_global"], vet["snr_local"]), 2)
            rows.append(row)
            print(f"  TIC {tic}: p_planet={row['p_planet']:.3f} verdict={row['verdict']} P={row['bls_period']}")
    except KeyboardInterrupt:
        print(f"\ninterrupted — keeping {len(rows)} scanned rows ...")
    rows.sort(key=lambda r: r["p_planet"], reverse=True)
    return rows


def write_candidates(rows, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["rank", "star_id", "p_planet", "p_false_positive", "p_no_signal", "pred_label", "verdict", "snr", "bls_period", "n_points"],
        )
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"rank": i, **r})
    print(f"wrote {len(rows)} ranked candidates -> {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scan TICs live from MAST for planets")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tics", default=None, help="comma-separated TIC ids to scan")
    ap.add_argument("--tic-range", default=None, help="inclusive range START-END, e.g. 100014359-100014460")
    ap.add_argument("--catalog", default=None, help="TOI catalog kind to scan: planet|candidate|fp")
    ap.add_argument("--catalog-n", type=int, default=50, help="how many brightest catalog targets")
    ap.add_argument("--mission", default="TESS")
    ap.add_argument("--download-dir", default="data/raw")
    ap.add_argument("--out", default="candidates.csv")
    args = ap.parse_args(argv)
    tics: list[int] = []
    if args.tics:
        tics += [int(x.strip()) for x in args.tics.split(",") if x.strip()]
    if args.tic_range:
        try:
            start_s, end_s = args.tic_range.split("-")
            start, end = int(start_s), int(end_s)
        except ValueError:
            raise SystemExit("--tic-range must look like START-END, e.g. 100014359-100014460")
        if end < start or end - start > 100000:
            raise SystemExit("range must be ascending and <= 100000 wide")
        tics += list(range(start, end + 1))
    if args.catalog:
        from ..data.catalog import select_targets

        tics += select_targets(kind=args.catalog, n=args.catalog_n)
    if not tics:
        raise SystemExit("give --tics, --tic-range, and/or --catalog")
    # dedupe, keep order
    tics = list(dict.fromkeys(tics))
    print(f"scanning {len(tics)} TICs ... (Ctrl-C keeps partial results)")
    rows = scan_tics(args.weights, tics, mission=args.mission, download_dir=args.download_dir)
    write_candidates(rows, args.out)
    print("\nRanked:")
    for r in rows:
        print(f"  {r['star_id']}: p_planet={r['p_planet']:.3f} {r['verdict']} SNR={r['snr']}")


if __name__ == "__main__":
    main()
