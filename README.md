cevns.py computes the CEνNS event rate for a given neutrino source and detector material by integrating dσ/dE_R over a digitized flux spectrum using the Helm nuclear form factor.

cevns_mc.py provides a Monte Carlo simulation of CEνNS scattering. It samples neutrino energies from the flux CDF and recoil energies from dσ/dE_R via rejection sampling, with plots of the resulting distributions.
