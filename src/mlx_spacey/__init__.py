"""mlx-spacey: hunt exoplanets with MLX + raw MAST light curves."""

from __future__ import annotations

__version__ = "0.2.0"


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(prog="mlx-spacey", description="Hunt exoplanets with MLX (raw MAST only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show env + raw-data pipeline")

    t = sub.add_parser("train", help="train MLX classifier on raw TICs from MAST")
    t.add_argument("--planet-tics", default=None, help="comma-separated TIC ids, confirmed planets")
    t.add_argument("--fp-tics", default=None, help="comma-separated TIC ids, false positives")
    t.add_argument("--field-tics", default=None, help="comma-separated TIC ids, assumed field")
    t.add_argument("--planet-catalog-n", type=int, default=0)
    t.add_argument("--fp-catalog-n", type=int, default=0)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--out-dir", default="checkpoints")

    d = sub.add_parser("discover", help="scan TICs live from MAST")
    d.add_argument("--weights", required=True)
    d.add_argument("--tics", default=None, help="comma-separated TIC ids to scan")
    d.add_argument("--tic-range", default=None, help="inclusive range START-END")
    d.add_argument("--catalog", default=None, help="TOI catalog kind: planet|candidate|fp")
    d.add_argument("--catalog-n", type=int, default=50)
    d.add_argument("--out", default="candidates.csv")

    g = sub.add_parser("targets", help="list target TICs from the TOI catalog")
    g.add_argument("--kind", default="candidate", help="planet|candidate|fp")
    g.add_argument("--n", type=int, default=50)
    g.add_argument("--refresh", action="store_true")
    g.add_argument("--out", default=None)

    args = p.parse_args()

    if args.cmd == "info":
        import mlx.core as mx

        print(f"mlx-spacey {__version__} (raw MAST via lightkurve, no parquet, no CSVs)")
        try:
            print(f"mlx {mx.__version__}")
        except AttributeError:
            print("mlx ok")
        try:
            import lightkurve

            print(f"lightkurve {lightkurve.__version__}")
        except (ImportError, AttributeError):
            print("lightkurve missing?")
        print("try: mlx-spacey train --epochs 2 --planet-tics 279741379 --field-tics 100014359")
        return
    if args.cmd == "train":
        from .pipelines.train import train

        def _pt(s):
            return [int(x.strip()) for x in s.split(",") if x.strip()] if s else None

        train(
            planet_tics=_pt(args.planet_tics),
            fp_tics=_pt(args.fp_tics),
            field_tics=_pt(args.field_tics),
            planet_catalog_n=args.planet_catalog_n,
            fp_catalog_n=args.fp_catalog_n,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
        )
        return
    if args.cmd == "discover":
        from .pipelines.discover import main as dmain

        cmd = ["--weights", args.weights, "--out", args.out]
        if args.tics:
            cmd += ["--tics", args.tics]
        if args.tic_range:
            cmd += ["--tic-range", args.tic_range]
        if args.catalog:
            cmd += ["--catalog", args.catalog, "--catalog-n", str(args.catalog_n)]
        dmain(cmd)
        return
    if args.cmd == "targets":
        from .data.catalog import main as cmain

        cmd = ["--kind", args.kind, "--n", str(args.n)]
        if args.refresh:
            cmd += ["--refresh"]
        if args.out:
            cmd += ["--out", args.out]
        cmain(cmd)
        return
