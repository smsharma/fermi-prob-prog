"""Regression tests for NPModel SVI and HMC fitting.

Usage:
    python tests/test_fits.py --generate              # save current outputs as ground truth
    pytest tests/test_fits.py                         # compare against saved ground truth
"""

import os
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from fpp.models.np_model import NPModel

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
FITS_FIXTURE = os.path.join(DATA_DIR, "fits_fixture.npz")

MODEL_KWARGS = dict(
    l_max=2,
    diffuse_names=["ModelO", "ModelA", "ModelF"],
    bulge_names=[
        "mcdermott2022", "mcdermott2022_bbp", "mcdermott2022_x",
        "macias2019", "coleman2019",
    ],
    ps_cat="3fgl",
    nside=128,
    n_exp=7,
    psf_tag="king",
)

SVI_KWARGS = dict(
    guide="iaf",
    num_flows=3,
    hidden_dims=[64, 64],
    n_steps=10,
    lr=5e-3,
    num_particles=4,
    renyi_alpha=1,
    lr_exp_decay=False,
    tempering_schedule="none",
)

HMC_KWARGS = dict(
    num_chains=1,
    num_warmup=5,
    num_samples=5,
    step_size=0.1,
)

RNG_SEED = 42
SVI_NUM_SAMPLES = 100

SVI_PREFIX = "svi__"
HMC_PREFIX = "hmc__"


def _run_all():
    model = NPModel(**MODEL_KWARGS)

    rng_key = jax.random.PRNGKey(RNG_SEED)
    model.fit_svi(rng_key=rng_key, data=model.data, **SVI_KWARGS)
    svi_samples = model.get_svi_samples(
        rng_key=jax.random.PRNGKey(RNG_SEED), num_samples=SVI_NUM_SAMPLES,
    )
    svi_samples = {k: np.asarray(v) for k, v in svi_samples.items()}

    rng_key = jax.random.PRNGKey(RNG_SEED)
    mcmc = model.run_nuts(rng_key=rng_key, data=model.data, **HMC_KWARGS)
    hmc_samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}

    return svi_samples, hmc_samples


def _generate():
    print("Running SVI and HMC to generate fixture ...")
    svi_samples, hmc_samples = _run_all()

    os.makedirs(DATA_DIR, exist_ok=True)
    save_dict = {}
    for k, v in svi_samples.items():
        save_dict[SVI_PREFIX + k] = v
    for k, v in hmc_samples.items():
        save_dict[HMC_PREFIX + k] = v
    np.savez(FITS_FIXTURE, **save_dict)

    print(f"Saved fixture to {FITS_FIXTURE}")
    print(f"  SVI parameters:   {list(svi_samples.keys())}")
    print(f"  HMC parameters:   {list(hmc_samples.keys())}")


def _load_fixture():
    fixture = np.load(FITS_FIXTURE)
    svi = {k[len(SVI_PREFIX):]: fixture[k] for k in fixture if k.startswith(SVI_PREFIX)}
    hmc = {k[len(HMC_PREFIX):]: fixture[k] for k in fixture if k.startswith(HMC_PREFIX)}
    return svi, hmc


# ------------------------------------------------------------------ #
#  Tests                                                              #
# ------------------------------------------------------------------ #

def test_svi_regression():
    """Compare SVI posterior samples against saved ground truth."""
    assert os.path.exists(FITS_FIXTURE), (
        f"Fixture not found at {FITS_FIXTURE}. "
        f"Run: python tests/test_fits.py --generate"
    )

    expected_svi, _ = _load_fixture()

    model = NPModel(**MODEL_KWARGS)

    rng_key = jax.random.PRNGKey(RNG_SEED)
    model.fit_svi(rng_key=rng_key, data=model.data, **SVI_KWARGS)
    svi_samples = model.get_svi_samples(
        rng_key=jax.random.PRNGKey(RNG_SEED), num_samples=SVI_NUM_SAMPLES,
    )

    assert set(svi_samples.keys()) == set(expected_svi.keys()), "SVI parameter keys mismatch"
    for key in expected_svi:
        assert np.asarray(svi_samples[key]) == pytest.approx(expected_svi[key], rel=1e-4, abs=1e-4), (
            f"SVI parameter '{key}' diverged from fixture"
        )


def test_hmc_regression():
    """Compare HMC samples against saved ground truth."""
    assert os.path.exists(FITS_FIXTURE), (
        f"Fixture not found at {FITS_FIXTURE}. "
        f"Run: python tests/test_fits.py --generate"
    )

    _, expected_hmc = _load_fixture()

    model = NPModel(**MODEL_KWARGS)

    rng_key = jax.random.PRNGKey(RNG_SEED)
    mcmc = model.run_nuts(rng_key=rng_key, data=model.data, **HMC_KWARGS)
    hmc_samples = mcmc.get_samples()

    assert set(hmc_samples.keys()) == set(expected_hmc.keys()), "HMC parameter keys mismatch"
    for key in expected_hmc:
        assert np.asarray(hmc_samples[key]) == pytest.approx(expected_hmc[key], rel=1e-4, abs=1e-4), (
            f"HMC parameter '{key}' diverged from fixture"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.generate:
        _generate()
    else:
        print("Use --generate to create fixtures, or run via pytest.")
