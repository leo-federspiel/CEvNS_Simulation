"""
CEνNS Monte Carlo simulation
Samples neutrino energies of detected events from φ(E_ν)·σ(E_ν) via inverse CDF,
then samples recoil energies from dσ/dE_R via rejection sampling.
Physics and flux data are imported from cevns.py.
Sources: solar_nuclear | geo | reactor | cnb
Materials: Ar | Xe | Ge
"""

import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cevns import (
    E_thresh, E_R_max, E_v_min, ds_dE, flux_at, load_spectrum,
    nucleus, sigma_above_threshold,
)


# --- sampling ---

def build_event_cdf(source, m, Q_W, R, n_grid=400):
    """
    Piecewise-linear CDF of the neutrino energies behind detected events,
    p(E_ν) ∝ φ(E_ν) σ(E_ν), where σ counts only recoils above threshold.
    Returns (E_v grid, unnormalized weights, cdf), or None if no event is possible.
    """
    logE, logF = load_spectrum(source)
    lo = max(logE[0], math.log10(E_v_min(m)))
    hi = logE[-1]
    if lo >= hi:
        return None

    E_v = np.logspace(lo, hi, n_grid)
    w = np.array([flux_at(E, logE, logF) * sigma_above_threshold(E, m, Q_W, R)
                  for E in E_v])

    areas = 0.5 * (w[:-1] + w[1:]) * np.diff(E_v)
    cdf = np.concatenate([[0.0], np.cumsum(areas)])
    if cdf[-1] <= 0:
        return None
    return E_v, w, cdf / cdf[-1]


def sample_E_v(E_v, cdf):
    """Draw one neutrino energy via inverse CDF."""
    u   = random.random()
    idx = int(np.searchsorted(cdf, u)) - 1
    idx = max(0, min(idx, len(E_v) - 2))
    t   = (u - cdf[idx]) / (cdf[idx + 1] - cdf[idx] + 1e-300)
    return E_v[idx] + t * (E_v[idx + 1] - E_v[idx])


def sample_E_R(E_v, m, Q_W, R, n_env=40, max_tries=300):
    """
    Draw one recoil energy via rejection sampling over [E_thresh, E_R^max].
    Builds a coarse envelope grid to handle non-monotone ds_dE.
    Returns (E_R, proposals used); E_R is None if no recoil was accepted.
    """
    hi = E_R_max(E_v, m)
    if hi <= E_thresh:
        return None, 0

    grid  = np.linspace(E_thresh, hi, n_env)
    f_max = max(ds_dE(E_v, er, m, Q_W, R) for er in grid)
    if f_max <= 0:
        return None, 0

    for tries in range(1, max_tries + 1):
        E_R = random.uniform(E_thresh, hi)
        if random.random() < ds_dE(E_v, E_R, m, Q_W, R) / f_max:
            return E_R, tries
    return None, max_tries


# --- simulation ---

def run_mc(source, mat, n_samples=50_000):
    """
    Returns (E_v samples, E_R samples, (E_v grid, weights) or None, total proposals).
    """
    m, Q_W, R, _ = nucleus(mat)
    built = build_event_cdf(source, m, Q_W, R)
    if built is None:
        return np.array([]), np.array([]), None, 0
    E_v_grid, weights, cdf = built

    sampled_E_v = []
    sampled_E_R = []
    proposals = 0

    for _ in range(n_samples):
        Ev = sample_E_v(E_v_grid, cdf)
        E_R, tries = sample_E_R(Ev, m, Q_W, R)
        proposals += tries
        if E_R is not None:
            sampled_E_v.append(Ev)
            sampled_E_R.append(E_R)

    return np.array(sampled_E_v), np.array(sampled_E_R), (E_v_grid, weights), proposals


def plot_results(source, mat, E_v_samples, E_R_samples, spectrum, Ev_min):
    n = len(E_v_samples)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"CEνNS Monte Carlo  —  {source} / {mat}   ({n:,} accepted events)",
        fontsize=12,
    )

    # 1. Sampled neutrino energies vs expected event spectrum φ·σ
    ax = axes[0]
    if n > 0:
        ax.hist(np.log10(E_v_samples), bins=50, color="steelblue",
                edgecolor="none", density=True, label="MC samples")
    if spectrum is not None:
        E_grid, w = spectrum
        u    = np.log10(E_grid)
        dens = w * E_grid * math.log(10)   # density per unit log10(E_ν)
        norm = np.trapezoid(dens, u)
        if norm > 0:
            ax.plot(u, dens / norm, "r--", lw=1.5, label="φ·σ (expected)")
    ax.axvline(math.log10(Ev_min), color="gray", lw=1, linestyle=":", label="E_ν min")
    ax.set_xlabel("log₁₀(E_ν / eV)")
    ax.set_ylabel("density")
    ax.set_title("Neutrino energies (detected events)")
    ax.legend(fontsize=8)

    # 2. Recoil energy distribution
    ax = axes[1]
    if n > 0:
        ax.hist(E_R_samples, bins=50, color="darkorange", edgecolor="none", density=True)
    ax.axvline(E_thresh, color="k", lw=1, linestyle=":", label=f"threshold = {E_thresh} eV")
    ax.set_xlabel("E_R [eV]")
    ax.set_ylabel("density")
    ax.set_title("Nuclear recoil energies")
    ax.legend(fontsize=8)

    # 3. 2D joint (E_ν, E_R)
    ax = axes[2]
    if n > 1:
        h = ax.hist2d(np.log10(E_v_samples), E_R_samples, bins=40, cmap="inferno")
        fig.colorbar(h[3], ax=ax, label="counts")
    else:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("log₁₀(E_ν / eV)")
    ax.set_ylabel("E_R [eV]")
    ax.set_title("E_ν vs E_R joint distribution")

    plt.tight_layout()
    out = Path(__file__).resolve().with_name(f"cevns_mc_{source}_{mat}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out}")
    plt.show()


if __name__ == "__main__":
    source = input("Neutrino source (solar_nuclear | geo | reactor | cnb): ").strip()
    mat    = input("Target material (Ar | Xe | Ge): ").strip()
    n      = int(input("Number of MC samples [default 50000]: ").strip() or 50000)

    m = nucleus(mat)[0]
    Ev_min = E_v_min(m)
    print(f"Minimum E_ν for detectable recoil: {Ev_min:.3e} eV  (log10 = {math.log10(Ev_min):.2f})")

    print(f"Running {n:,} samples...")
    E_v_s, E_R_s, spectrum, proposals = run_mc(source, mat, n_samples=n)

    if spectrum is None:
        print("No events possible: the flux has no support above the kinematic threshold for this material.")
    else:
        print(f"Accepted: {len(E_v_s):,} of {n:,}")
        if proposals > 0:
            print(f"Rejection sampler efficiency: {100 * len(E_v_s) / proposals:.1f}%")
        if len(E_v_s) > 0:
            print(f"Mean E_ν (sampled): {E_v_s.mean():.3e} eV")
            print(f"Mean E_R:           {E_R_s.mean():.3e} eV")
            print(f"E_R range:          [{E_R_s.min():.3e}, {E_R_s.max():.3e}] eV")
        plot_results(source, mat, E_v_s, E_R_s, spectrum, Ev_min)
