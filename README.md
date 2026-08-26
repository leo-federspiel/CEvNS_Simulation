# CEνNS Simulation

Two Python programs that model coherent elastic neutrino nucleus scattering (CEνNS). One computes the expected scattering event rate for a given neutrino source and detector material. The other runs a Monte Carlo of individual scattering events and plots the resulting energy distributions. Both cover four neutrino sources and three detector materials.

## Background

In CEνNS, a low energy neutrino scatters off an entire nucleus at once rather than off a single nucleon. Because the interaction is coherent, the cross section scales roughly with the square of the neutron number, which makes it large compared to other neutrino interactions at these energies. The tradeoff is that the only observable is a very small nuclear recoil, on the order of tens to hundreds of eV, which is why detection is hard and why a recoil threshold matters so much.

Both programs use the standard CEνNS differential cross section dσ/dE_R, built from the Fermi constant, the weak nuclear charge Q_W = N minus roughly 0.072 Z, and the Helm form factor, which accounts for the finite size of the nucleus once the momentum transfer gets large enough that the neutrino no longer sees the nucleus as a point.

## Contents

### cevns.py

Computes the expected event rate for a chosen source and material using only the Python standard library.

The cross section is evaluated from the Helm form factor (skin thickness 0.9 fm) and the weak charge for the selected nucleus. Each neutrino source is stored as a digitized flux spectrum, a set of paired log10 energy and log10 flux points, and the four available sources are `solar_nuclear`, `geo` (geoneutrinos), `reactor`, and `cnb` (the cosmic neutrino background). The three materials are argon, xenon, and germanium, each carrying its nuclear mass, neutron and proton numbers, and molar mass.

To get the rate, the program walks the flux spectrum and integrates with the trapezoidal rule. For each neutrino energy it evaluates the cross section at the midpoint of the kinematically allowed recoil window, applies a 100 eV recoil threshold, and accumulates the flux weighted contribution, then scales by the number of target nuclei in a 23 kg detector. It prints the event rate in events per second and the mean time between events.

Run it:

```
python cevns.py
```

It prompts for a source and a material, then prints the result.

### cevns_mc.py

Runs a Monte Carlo of individual scattering events and visualizes the sampled distributions. Needs NumPy and Matplotlib.

Neutrino energies are drawn by inverse CDF sampling. The program builds a piecewise linear cumulative distribution from the flux spectrum and inverts it to draw energies that follow the spectrum. Before sampling, it computes the minimum neutrino energy capable of producing a recoil above threshold for the chosen nucleus, and restricts the draws to that range, so it does not waste samples on neutrinos that could never register a signal.

Recoil energies are drawn by rejection sampling of dσ/dE_R over the interval from the threshold up to the maximum kinematic recoil. Because the cross section is not always monotone across that interval, the sampler first builds a coarse envelope on a grid to find a safe ceiling, then accepts or rejects against it. Neutrino draws whose recoil window sits entirely below threshold are discarded.

The run reports the acceptance fraction along with the mean sampled neutrino energy, the mean recoil energy, and the recoil range. It then produces three plots: the sampled neutrino energies against the restricted flux spectrum, the recoil energy distribution with the threshold marked, and a 2D histogram of the joint distribution of neutrino energy and recoil energy. The figure is saved to disk.

Run it:

```
python cevns_mc.py
```

It prompts for a source, a material, and a sample count (default 50000), then runs and displays the plots.

## Requirements

* Python 3
* NumPy and Matplotlib, for `cevns_mc.py` only (`cevns.py` uses the standard library alone)

```
pip install numpy matplotlib
```

## Quick reference

* Sources: `solar_nuclear`, `geo`, `reactor`, `cnb`
* Materials: `Ar`, `Xe`, `Ge`

## Assumptions and limitations

The detector mass (23 kg) and recoil threshold (100 eV) are fixed constants in the source, so changing them means editing the file. The flux spectra are digitized from published curves rather than computed, so their resolution is set by how finely each curve was sampled. The weak charge uses the common approximation for the neutron and proton couplings and does not include higher order electroweak corrections. The rate calculator evaluates the cross section at the midpoint of the recoil window rather than integrating fully over recoil energy, which trades some accuracy for speed and simplicity. These are all reasonable choices for a first pass estimator, and each is a natural place to extend the model.
