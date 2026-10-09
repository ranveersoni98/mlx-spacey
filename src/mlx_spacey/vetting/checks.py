"""Vetting checks: BLS period search, odd/even depth, SNR, verdicts.

The bridge from "ML says planet" to "plausible new planet":
  1. BLS periodogram (astropy, numpy fallback) on raw flux
  2. Odd/even transit depth consistency (eclipsing-binary check)
  3. Transit SNR heuristics -> PASS / REVIEW verdict
"""

from __future__ import annotations

import numpy as np


def odd_even_consistency(odd: np.ndarray, even: np.ndarray) -> dict:
    """Compare odd vs even folded depths. Large mismatch => likely EB, not planet."""
    o = np.asarray(odd, dtype=np.float64)
    e = np.asarray(even, dtype=np.float64)
    # depth = median - min (views are median-normalized; handle both scalings
    # by measuring the dip relative to the median)
    def depth(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan
        return float(np.median(v) - np.min(v))

    do, de = depth(o), depth(e)
    denom = max(abs(do) + abs(de), 1e-9) / 2
    rel_diff = abs(do - de) / denom if np.isfinite(do) and np.isfinite(de) else np.nan
    return {
        "odd_depth": do,
        "even_depth": de,
        "odd_even_rel_diff": float(rel_diff) if np.isfinite(rel_diff) else float("nan"),
        "passes_odd_even": bool(np.isfinite(rel_diff) and rel_diff < 0.5),
    }


def snr_estimate(folded: np.ndarray) -> float:
    """Rough transit SNR: depth / (std/sqrt(n_in_transit))."""
    v = np.asarray(folded, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 10:
        return 0.0
    depth = float(np.median(v) - np.min(v))
    # in-transit ~= lowest 10% of points
    k = max(3, len(v) // 10)
    in_tr = np.sort(v)[:k]
    noise = float(np.std(v) + 1e-9)
    return float(depth / (noise / np.sqrt(k)))


def bls_search(time: np.ndarray, flux: np.ndarray, min_period=0.5, max_period=30.0):
    """Box-Least-Squares period search (astropy, numpy fallback).

    Returns dict(period, power, depth, duration).
    """
    t = np.asarray(time, dtype=np.float64)
    f = np.asarray(flux, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 100:
        return {"period": float("nan"), "power": 0.0, "method": "none (too few points)"}
    try:
        from astropy.timeseries import BoxLeastSquares  # type: ignore

        dur = np.linspace(0.05, min(0.5, min_period * 0.5), 10)
        per = np.linspace(min_period, max_period, 2000)
        res = BoxLeastSquares(t, f).power(per, dur, objective="snr")
        i = int(np.argmax(res.power))
        return {
            "period": float(res.period[i]),
            "power": float(res.power[i]),
            "depth": float(res.depth[i]),
            "duration": float(res.duration[i]),
            "method": "astropy BLS",
        }
    except (ImportError, ValueError):
        pass
    # --- numpy fallback: trial periods, 2-bin box ---
    f = (f - np.median(f)) / (np.std(f) + 1e-9)
    best = {"period": float("nan"), "power": 0.0, "method": "numpy fallback grid"}
    for p in np.linspace(min_period, max_period, 200):
        phase = ((t - t[0]) % p) / p
        in_tr = phase < 0.05  # 5% duty cycle box
        if in_tr.sum() < 5 or (~in_tr).sum() < 5:
            continue
        s = abs(float(np.mean(f[in_tr]) - np.mean(f[~in_tr])))
        if s > best["power"]:
            best = {"period": float(p), "power": s, "method": "numpy fallback grid"}
    return best


def vet_candidate(row: dict, flux_global, flux_local, flux_odd, flux_even) -> dict:
    """Combine heuristics into one verdict dict for a discovery row."""
    oe = odd_even_consistency(np.asarray(flux_odd), np.asarray(flux_even))
    snr_g = snr_estimate(np.asarray(flux_global))
    snr_l = snr_estimate(np.asarray(flux_local))
    verdict = "PASS" if (oe["passes_odd_even"] and max(snr_g, snr_l) > 7.0) else "REVIEW"
    return {
        "star_id": row.get("star_id"),
        "p_planet": row.get("p_planet"),
        **oe,
        "snr_global": snr_g,
        "snr_local": snr_l,
        "verdict": verdict,
    }


RAW_TESS_HINT = """\
Raw only — live from MAST:
  import lightkurve as lk
  lcs = lk.search_lightcurve("TIC <id>", mission="TESS").download_all(download_dir="data/raw")
  lc = lcs.stitch().remove_nans().remove_outliers().flatten(window_length=1001)
"""
