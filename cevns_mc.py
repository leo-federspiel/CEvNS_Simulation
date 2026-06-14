"""
CEνNS Monte Carlo simulation
Samples neutrino energies from the digitized flux spectrum via inverse CDF,
then samples recoil energies from dσ/dE_R via rejection sampling.
Sources: solar_nuclear | geo | reactor | cnb
Materials: Ar | Xe | Ge
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt

# --- constants ---
G_F    = 1.166e13    # Fermi constant [eV^-2]
N_A    = 6.02214e23
hbar_c = 197.3e6     # eV·fm
to_cm2 = 3.894e-52   # natural units -> cm^2

s        = 0.9    # Helm skin thickness [fm]
M_det    = 23     # detector mass [kg]
E_thresh = 100    # recoil threshold [eV]

# --- flux data (log10 scale) ---
FLUX_DATA = {
    "solar_nuclear": {
        "E_log": [
            2.9501023855524675, 3.018394961871426, 3.0911271152616724,
            3.180113802932315, 3.241061131989504, 3.31722125875509,
            3.388423809961113, 3.465563972647738, 3.54803047990082,
            3.6252926089905024, 3.7560334185736846, 3.9019870258652674,
            3.971045838175197, 4.038114445767004, 4.08179707174671,
            4.1591826021383085, 4.313371093984552, 4.395653934183619,
            4.466120382274717, 4.531781354085357, 4.603900805662597,
            4.679878700273029, 4.7346416152457795, 4.792099270276667,
            4.864432521783973, 4.937012575895112, 5.0112743314695845,
            5.094323407659623, 5.19050324331293, 5.262317061433098,
            5.343284099378005, 5.411424576417858, 5.458411774470987,
            5.529246991568977, 5.5568271825473925, 5.5568271825473925,
            5.569041041639441, 5.659343531564243, 5.754147299211292,
            5.837103106975464, 5.883141836882464, 6.128337353993523,
            6.147224927681098, 6.145755591248971, 6.198376202224491,
            6.383328925618375, 6.717724930330142, 6.802211775177398,
            6.802211775177398, 6.932217916544516, 7.035071466793351,
            7.075783852133894, 7.1056613161863655, 7.138200517623208,
            7.170372384952017,
        ],
        "flux_log": [
            0.9842365244579341, 1.1472610458990076, 1.2940283622998408,
            1.3781350549119722, 1.5595627112903543, 1.719536360756912,
            1.863819055489003, 1.9847320179757517, 2.142112558565451,
            2.304232137929084, 2.612102958378543, 2.8433156591631814,
            2.98590489503691, 3.140462122186193, 3.2267147446953928,
            3.3256127420234947, 3.7257215036346665, 3.8263129598211165,
            3.992835532216482, 4.124475498159143, 4.240532996575446,
            4.388202609024255, 4.402652575939086, 4.501889265038855,
            4.632060684627781, 4.780521459886977, 4.885063438066446,
            4.981138121954125, 5.104649485376669, 5.139420958591792,
            5.243513111762013, 5.282235106970148, 5.1762245824369515,
            4.83425967231571, 3.9429367085431863, 3.9429367085431863,
            3.0755470812922994, 2.3750187182084446, 2.4112587377773025,
            2.6718264894597823, 2.460369044669683, 1.443163875072912,
            0.7351869390417534, 0.04628787546177904, -0.5556811820656478,
            -0.5367144428520376, -0.31453264063548403, -0.3587874834574194,
            -0.3587874834574194, -0.7031258820098216, -1.3441556265170576,
            -1.9497365142658225, -2.7157435225198654, -3.613390496716569,
            -4.499408171409399,
        ],
    },
    "geo": {
        "E_log": [
            4.854697220062208, 4.976554989042839, 5.0974937703237675,
            5.2305933435246725, 5.368257700940196, 5.526169942033084,
            5.65651398805316, 5.7931057145129365, 5.955736383076488,
            6.033298944931429, 6.1248904283896515, 6.268436307083276,
            6.297629961649937, 6.434037890569773, 6.468377876254772,
        ],
        "flux_log": [
            1.2566855866189286, 1.4627524442455382, 1.687411978993886,
            1.7767550606210492, 1.7336045227931365, 1.8192322377521677,
            1.8904368306491506, 1.9220953459186418, 1.7518565586212063,
            1.1921274598937188, 0.6256383478965759, 0.30735526999367835,
            -0.5124098860499586, -0.9317551623342055, -2.430112103553391,
        ],
    },
    "reactor": {
        "E_log": [
            3.862434610490599, 3.999487266718507, 4.134149119008908,
            4.291909152764033, 4.492033083557189, 4.671297429485367,
            4.876966876678921, 5.054760842287573, 5.247624770253076,
            5.430351115156228, 5.61886564664923, 5.799232777817052,
            6.014489851371414, 6.19641926162873, 6.371578317192139,
            6.523271596210943, 6.660691847518734, 6.756143079315708,
            6.81986051710731, 6.8627456524812676,
        ],
        "flux_log": [
            -0.20437773451925523, 0.04461520736966662, 0.3864474252486474,
            0.6957180323353995, 0.9258964841671755, 0.8680983707116141,
            0.9557514490213634, 1.1649738597172075, 1.2666011529578185,
            1.2824871841415089, 1.3073856861413446, 1.3007392199695786,
            1.1451453676696133, 0.9616110073984601, 0.6872680157483302,
            0.222628009927206, -0.37799916908170417, -1.0616609642094872,
            -1.7749559112560505, -2.390117118794713,
        ],
    },
    "cnb": {
        "E_log": [
            -6.289163014279655, -6.167580941608936, -5.974685423441255,
            -5.793828644139686, -5.636589848720487, -5.449024459211085,
            -5.258058815212781, -5.112858758659692, -4.941927046514916,
            -4.749215325887176, -4.548048918422168, -4.370592393609501,
            -4.173561430793157, -3.999597059239361, -3.768747349073943,
            -3.5809062632546302, -3.4047363212215465, -3.211657005513926,
            -3.0941184787219003, -2.986688816626608, -2.877972571751732,
            -2.8055563410151274, -2.707224657146896, -2.6248833592534995,
            -2.5680899194118476, -2.5190159762477027, -2.445037466421603,
            -2.4025802346953204, -2.3755619963240493, -2.3136222253640604,
            -2.2288915594514345,
        ],
        "flux_log": [
            10.761039790189216, 11.167654588359007, 11.3366549201004,
            11.79903982774485, 12.199232613308467, 12.409469025994753,
            12.823519838761165, 13.217628612382093, 13.532307230084562,
            13.680689521353507, 14.165382472787819, 14.531099190676187,
            14.790683700230971, 15.036410182582955, 15.283150666925387,
            15.283150666925387, 15.052634214430128, 14.596671319391852,
            14.036942220664358, 13.427527024404903, 12.48619517660535,
            11.516471273073243, 10.304062893160491, 8.880404098571006,
            7.37630114607262, 5.8728741949012, 4.074372664509305,
            2.7629300901961003, 0.8380163116616473, -0.9783992538948283,
            3.658631848425479,
        ],
    },
}

MATERIALS = {
    "Ar": {"m": 37_210_000_000, "N": 22, "Z": 18, "M": 0.03995},
    "Xe": {"m": 1.223e11,       "N": 77, "Z": 54, "M": 0.131293},
    "Ge": {"m": 6.765e10,       "N": 42, "Z": 32, "M": 0.07263},
}


# --- physics ---

def helm_F(q, R):
    """Helm form factor; q in fm^-1, R in fm."""
    qR = q * R
    j1_qR = math.sin(qR) / qR**2 - math.cos(qR) / qR
    return 3 * j1_qR * math.exp(-(q * s)**2 / 2) / qR


def ds_dE(E_v, E_R, m, Q_W, R):
    """dσ/dE_R [cm^2/eV]."""
    q   = math.sqrt(2 * m * E_R) / hbar_c
    F   = helm_F(q, R)
    kin = 1 - m * E_R / (2 * E_v**2)
    return G_F**2 / (4 * math.pi) * kin * Q_W**2 * F**2 * to_cm2


def min_E_v_for_threshold(m, thresh=E_thresh):
    """Minimum neutrino energy to produce a recoil above thresh (non-relativistic nucleus)."""
    return math.sqrt(thresh * m / 2)


def build_flux_cdf(source, E_v_min=None):
    """
    Piecewise-linear CDF from the flux spectrum for inverse-CDF sampling.
    Optionally restrict to E_v >= E_v_min (to skip unphysical low-energy region).
    Returns (E_v array, cdf array).
    """
    d    = FLUX_DATA[source]
    E_v  = np.array([10**x for x in d["E_log"]])
    flux = np.array([10**x for x in d["flux_log"]])
    flux = np.maximum(flux, 0)

    if E_v_min is not None:
        mask = E_v >= E_v_min
        if mask.sum() < 2:
            # fall back to full spectrum if too few points pass
            pass
        else:
            E_v  = E_v[mask]
            flux = flux[mask]

    widths = np.diff(E_v)
    areas  = 0.5 * (flux[:-1] + flux[1:]) * widths
    cdf    = np.concatenate([[0], np.cumsum(areas)])
    total  = cdf[-1]
    if total == 0:
        raise ValueError("Flux integrates to zero over the usable energy range.")
    cdf /= total
    return E_v, cdf


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
    Returns None if no valid recoil is possible.
    """
    E_R_max = 2 * E_v**2 / (m + 2 * E_v)
    if E_R_max <= E_thresh:
        return None

    grid  = np.linspace(E_thresh, E_R_max, n_env)
    f_max = max(ds_dE(E_v, er, m, Q_W, R) for er in grid)
    if f_max <= 0:
        return None

    for _ in range(max_tries):
        E_R = random.uniform(E_thresh, E_R_max)
        if random.random() < ds_dE(E_v, E_R, m, Q_W, R) / f_max:
            return E_R
    return None


# --- simulation ---

def run_mc(source, mat, n_samples=50_000):
    p = MATERIALS[mat]
    m, N, Z, M = p["m"], p["N"], p["Z"], p["M"]
    A   = N + Z
    Q_W = N - 0.072 * Z
    R   = math.sqrt((1.2 * A**(1/3))**2 - 5 * s**2)

    # Only sample neutrinos energetic enough to produce a detectable recoil
    Ev_min = min_E_v_for_threshold(m)
    E_v_grid, cdf = build_flux_cdf(source, E_v_min=Ev_min)

    sampled_E_v = []
    sampled_E_R = []

    for _ in range(n_samples):
        Ev  = sample_E_v(E_v_grid, cdf)
        E_R = sample_E_R(Ev, m, Q_W, R)
        if E_R is not None:
            sampled_E_v.append(Ev)
            sampled_E_R.append(E_R)

    return np.array(sampled_E_v), np.array(sampled_E_R), Ev_min


def plot_results(source, mat, E_v_samples, E_R_samples, Ev_min):
    n = len(E_v_samples)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"CEνNS Monte Carlo  —  {source} / {mat}   ({n:,} accepted events)",
        fontsize=12,
    )

    # 1. Sampled neutrino energies vs digitized spectrum
    ax = axes[0]
    if n > 0:
        ax.hist(np.log10(E_v_samples), bins=50, color="steelblue",
                edgecolor="none", density=True, label="MC samples")

    d    = FLUX_DATA[source]
    E_v  = np.array([10**x for x in d["E_log"]])
    flux = np.array([10**x for x in d["flux_log"]])
    flux = np.maximum(flux, 0)
    mask = E_v >= Ev_min
    if mask.sum() >= 2:
        Ev_r  = E_v[mask]
        fl_r  = flux[mask]
        norm  = np.trapezoid(fl_r, np.log10(Ev_r))
        if norm > 0:
            ax.plot(np.log10(Ev_r), fl_r / norm, "r--", lw=1.5, label="spectrum (restricted)")
    ax.axvline(math.log10(Ev_min), color="gray", lw=1, linestyle=":", label=f"E_ν min")
    ax.set_xlabel("log₁₀(E_ν / eV)")
    ax.set_ylabel("density")
    ax.set_title("Neutrino energies")
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
    out = "/Users/leofederspiel/OSR_Project/cevns_mc.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out}")
    plt.show()


if __name__ == "__main__":
    source = input("Neutrino source (solar_nuclear | geo | reactor | cnb): ").strip()
    mat    = input("Target material (Ar | Xe | Ge): ").strip()
    n      = int(input("Number of MC samples [default 50000]: ").strip() or 50000)

    p = MATERIALS[mat]
    Ev_min = min_E_v_for_threshold(p["m"])
    print(f"Minimum E_ν for detectable recoil: {Ev_min:.3e} eV  (log10 = {math.log10(Ev_min):.2f})")

    print(f"Running {n:,} samples...")
    E_v_s, E_R_s, Ev_min = run_mc(source, mat, n_samples=n)
    frac = 100 * len(E_v_s) / n if n > 0 else 0
    print(f"Accepted: {len(E_v_s):,}  ({frac:.1f}%)")

    if len(E_v_s) > 0:
        print(f"Mean E_ν (sampled): {E_v_s.mean():.3e} eV")
        print(f"Mean E_R:           {E_R_s.mean():.3e} eV")
        print(f"E_R range:          [{E_R_s.min():.3e}, {E_R_s.max():.3e}] eV")
    else:
        print("No events accepted — the flux has no support above the kinematic threshold for this material.")

    plot_results(source, mat, E_v_s, E_R_s, Ev_min)