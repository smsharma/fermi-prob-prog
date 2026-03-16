# Differentiable Probabilistic Programming for the Galactic Center Excess

A GPU-accelerated, fully differentiable framework for disentangling gamma-ray observations of the Galactic Center using [JAX](https://github.com/jax-ml/jax) and [NumPyro](https://num.pyro.ai). Built to flexibly explore the vast model space of the [Galactic Center Excess](https://en.wikipedia.org/wiki/Galactic_Center_GeV_excess) --- simultaneously accounting for a continuum of spatial morphologies, multiple point source populations, and diffuse emission components in a single probabilistic model.

<p align="center">
  <img src="assets/svi_posterior.gif" width="600" alt="SVI posterior optimization">
  <br>
  <em>Posterior convergence via Stochastic Variational Inference (SVI).</em>
</p>

---

## Quick Start

Fitting the fiducial model to *Fermi* data:

```python
from fpp.models.np_model import NPModel

m = NPModel()  # loads Fermi data, templates, PSF, and mask

# SVI with an inverse autoregressive flow (IAF)
m.fit_svi(data=m.data, guide='iaf', num_flows=5, hidden_dims=[128, 128],
           lr=3e-4, n_steps=10000)

# or, HMC via NUTS
m.run_nuts(data=m.data, num_chains=4, num_warmup=500, num_samples=2500)
```

### Extending the Model

`NPModel` is designed to be subclassed. Say you want to add an **isotropic point source population** on top of the fiducial model. Just inherit and override `model()` --- everything else (templates, PSF corrections, exposure regions) is reused automatically:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from fpp.models.np_model import NPModel
from fpp.models.scd import dnds
from fpp.utils.utils import jnp_trapezoid

class NPModelIso(NPModel):
    """Fiducial model + isotropic point sources."""

    def model(self, data=None, beta=1.):
        # --- call the parent to set up all fiducial components ---
        #     (diffuse, GCE, disk PS, etc. are handled by NPModel)
        #
        # Here we show only the *new* piece: an isotropic PS population
        # whose spatial template is the already-loaded self.temp_iso.

        # ... [fiducial model components go here, see examples/fit_to_fermi.ipynb] ...

        # New isotropic PS population — just need an overall flux and SCD params
        Sps_iso = numpyro.sample("Sps_iso", dist.Uniform(1e-5, 8.))

        n1  = numpyro.sample("n1_iso",  dist.Uniform(4.0, 6.0))
        n2  = numpyro.sample("n2_iso",  dist.Uniform(0.5, 1.99))
        n3  = numpyro.sample("n3_iso",  dist.Uniform(-6., -5.))
        sb1 = numpyro.sample("sb1_iso", dist.Uniform(5., 40.))
        lam = numpyro.sample("lam_iso", dist.Uniform(0.1, 0.95))

        # normalize the source-count distribution
        s_arr = jnp.logspace(-1., 2., 1000)
        theta = jnp.array([1., n1, n2, n3, sb1, lam * sb1])
        A = Sps_iso / jnp_trapezoid(s_arr * dnds(s_arr, theta), s_arr)

        # ... plug [A, n1, n2, n3, sb1, lam*sb1] into the likelihood
        #     alongside self.temp_iso as the spatial template ...

m_iso = NPModelIso()
m_iso.fit_svi(data=m_iso.data, guide='iaf', num_flows=4,
              hidden_dims=[64, 64], lr=1e-4, n_steps=5000)
```

The full working example lives in [`examples/fit_to_fermi.ipynb`](examples/fit_to_fermi.ipynb).

---

## Installation

Create a fresh environment and install:

```bash
mamba create -n fpp python=3.12
mamba activate fpp
```

Install JAX for your hardware --- see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for GPU/TPU instructions:

```bash
# CPU-only (just to get started)
pip install jax

# CUDA 12 (example)
pip install jax[cuda12]
```

Then install `fpp` in editable mode:

```bash
pip install -e .
```

---

## Results

<p align="center">
  <img src="assets/fermi_result.png" width="700" alt="Full Fermi analysis results">
  <br>
  <em>Posterior constraints from the fiducial analysis of Fermi-LAT data (2--20 GeV).</em>
</p>

<!-- TODO: replace with final figure -->

---

## Citation

If you use this code, please cite:

```bibtex
@article{fpp2026,
  title   = {Disentangling gamma-ray observations of the Galactic Center
             using differentiable probabilistic programming},
  author  = {Mishra-Sharma, Siddharth and Slatyer, Tracy R. and Sun, Yitian and Wu, Yuqing},
  year    = {2026},
  journal = {TODO}
}
```

---

## Authors

[Siddharth Mishra-Sharma](mailto:smsharma@mit.edu), [Tracy R. Slatyer](mailto:tslatyer@mit.edu), [Yitian Sun](mailto:yitians@mit.edu), and Yuqing Wu
