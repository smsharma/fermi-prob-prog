# Differentiable probabilistic programming for the GCE

- [X] Add ability to include multiple bulge models together with a simplex prior
- [X] Add ability to include multiple diffuse models together with a simplex prior
- [X] Sigmoid-bound problematic parameters to hedge against NaN? Update: Seems to work better with ELU -> Tanh nonlinearity
- [ ] See if n_exp > 1 works with more memory
- [X] Clean up pipeline for NeuTra -> MCMC
- [ ] Implement a GP template?
- [ ] Try running full pipeline (up to MCMC, potentially just SVI if including Ylm or GPs)
