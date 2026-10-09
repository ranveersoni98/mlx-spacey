# mlx-spacey

Hunt exoplanets with [MLX](https://github.com/ml-explore/mlx) and raw TESS light curves.
No pre-processed datasets, no CSV dumps. Every training example is fetched
live from MAST, folded, and fed to a small 1D CNN running on Apple Silicon.

## Quickstart

```bash
uv sync
uv run mlx-spacey info

# train on 2 TICs (1 confirmed planet + 1 field star)
uv run mlx-spacey train --epochs 2 --planet-tics 279741379 --field-tics 100014359

# train bigger: brightest confirmed + false-positive TOIs from the catalog
uv run mlx-spacey train --epochs 10 --planet-catalog-n 25 --fp-catalog-n 25

# hunt: scan ambiguous planet candidates, brightest first
uv run mlx-spacey discover --weights checkpoints/transit_cnn_best.safetensors \
  --catalog candidate --catalog-n 20 --out hunt.csv
```

## Commands

| Command | What it does |
|---|---|
| `mlx-spacey info` | env + pipeline sanity check |
| `mlx-spacey train` | fetch TICs from MAST, BLS-fold them, train the CNN |
| `mlx-spacey discover` | scan TICs (`--tics`, `--tic-range`, or `--catalog`), rank by P(PLANET) |
| `mlx-spacey targets` | list TICs from the TOI catalog (`planet` / `candidate` / `fp`) |

`discover` writes a ranked CSV (`p_planet`, vetting `verdict`, BLS period, SNR).
Ctrl-C keeps partial results. Re-runs reuse the MAST cache in `data/raw/`.

## How it works

```
MAST --(lightkurve)--> stitch + flatten --> BLS period search --> phase fold
  --> global view (201) + local view (81) --> MLX dual-view CNN --> P(PLANET)
  --> odd/even + SNR vetting --> PASS / REVIEW
```

- training positives: confirmed TOIs (`CP`/`KP`). Negatives: false positives
  (`FP`/`FA`, mostly eclipsing binaries) plus field stars.
- the hunting ground is `--catalog candidate`: ambiguous (`PC`/`APC`) TOIs.
  A high `p_planet` there is worth a closer look, not a discovery claim.

## Layout

```
src/mlx_spacey/
  data/        mast.py (MAST fetching) / views.py (folding) /
               batch.py (batching) / catalog.py (TOI catalog via TAP)
  models/      transit_cnn.py (MLX dual-view 1D CNN, ~74k params)
  pipelines/   train.py / discover.py
  vetting/     checks.py (BLS, odd/even, SNR, verdicts)
```

## Notes

- Needs Apple Silicon (MLX) and network access (MAST + NASA Exoplanet Archive TAP).
- `data/` (downloads, TOI cache) and `checkpoints/` (weights) are gitignored.
  Result CSVs (`candidates.csv`, sweeps) are gitignored too.
- `numpy` stays on purpose. mlx, astropy and lightkurve all compute on it.
  Nothing here imports pandas or scikit-learn directly.

## License

Code is MIT. See [LICENSE](LICENSE).

<p align="center">
  Built with <3 and a lot of bad decisions! by <a href="https://ranveersoni.me">Ranveer Soni</a>
</p>
