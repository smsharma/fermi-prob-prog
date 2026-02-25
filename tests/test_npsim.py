"""Regression test for NPModel.simulate.

Usage:
    python tests/test_npsim.py --generate              # save current outputs as ground truth
    pytest tests/test_npsim.py                         # compare against saved ground truth
"""

import os
import numpy as np
import pytest

import jax.numpy as jnp

from fpp.models.np_model import NPModel

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
SIM_FIXTURE = os.path.join(DATA_DIR, "npsim_fixture.npz")

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

# Truth parameters for simulation
TRUTH_VD = dict(
    gamma_poiss=0.9,
    gamma_ps=1.2,
    S_gce=3.0,
    S_iso=1.0,
    S_bub=2.0,
    S_psc=1.5,
    S_pib=10.0,
    S_ics=7.0,
    f_bulge_poiss=0.3,
    f_bulge_ps=0.3,
    theta_pib=np.array([0.4, 0.35, 0.25]),
    theta_ics=np.array([0.5, 0.3, 0.2]),
    theta_bulge_poiss=np.array([0.3, 0.2, 0.2, 0.15, 0.15]),
    theta_bulge_ps=np.array([0.3, 0.2, 0.2, 0.15, 0.15]),
    zs=0.3,
    C=5.0,
    Sps_gce=1.0,
    Sps_dsk=2.0,
    n1_gce=5.0,
    n2_gce=1.2,
    n3_gce=-5.5,
    sb1_gce=10.0,
    lambdas_gce=0.6,
    n1_dsk=5.5,
    n2_dsk=1.1,
    n3_dsk=-5.2,
    sb1_dsk=11.0,
    lambdas_dsk=0.4,
)

RNG_SEED = 42
MODIFIERS = []


def _run():
    model = NPModel(**MODEL_KWARGS)
    sim_map = model.simulate(TRUTH_VD, modifiers=MODIFIERS, rng_seed=RNG_SEED)
    return np.asarray(sim_map)


def _generate():
    print("Running NPModel.simulate to generate fixture ...")
    sim_map = _run()

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(SIM_FIXTURE, sim_map=sim_map)

    print(f"Saved fixture to {SIM_FIXTURE}")
    print(f"  sim_map shape: {sim_map.shape}")
    print(f"  sim_map range: [{sim_map.min():.4f}, {sim_map.max():.4f}]")
    print(f"  sim_map sum:   {sim_map.sum():.4f}")


# ------------------------------------------------------------------ #
#  Test                                                               #
# ------------------------------------------------------------------ #

def test_simulate_regression():
    """Compare NPModel.simulate output against saved ground truth."""
    assert os.path.exists(SIM_FIXTURE), (
        f"Fixture not found at {SIM_FIXTURE}. "
        f"Run: python tests/test_npsim.py --generate"
    )

    fixture = np.load(SIM_FIXTURE)
    expected = fixture["sim_map"]

    sim_map = _run()

    assert sim_map.shape == expected.shape, "Simulated map shape mismatch"
    assert sim_map == pytest.approx(expected, rel=1e-4, abs=1e-4), (
        "Simulated map diverged from fixture"
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
