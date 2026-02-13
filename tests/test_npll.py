"""Regression test for fpp.likelihoods.npll_jax.log_like_np.

Usage:
    python tests/test_npll.py --generate              # save current outputs as ground truth
    pytest tests/test_npll.py                         # compare against saved ground truth
"""

import os
import numpy as np
import pytest

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
NPLL_FIXTURE = os.path.join(DATA_DIR, "npll_fixture.npz")

N_PIX = 50
N_NPT = 2   # GCE + disk, like in NPModel
K_MAX = 100


def _build_inputs():
    """Build realistic synthetic inputs matching what NPModel.model passes to log_like_np."""
    rng = np.random.default_rng(123)

    # theta (NPT, 6): [A, n1, n2, n3, sb1, sb2]
    # A ~ O(0.01-1), n1 in [4,6], n2 in [0.5,2], n3 in [-6,-5], sb1 in [5,40], sb2 = lambda_s * sb1
    theta = jnp.array([
        [0.1, 5.0, 1.2, -5.5, 20.0, 12.0],   # GCE-like
        [0.05, 4.8, 1.0, -5.8, 15.0, 9.0],    # disk-like
    ])

    # pt_sum (P,): summed Poissonian expected counts per pixel, typically O(1-10)
    pt_sum = jnp.array(rng.uniform(1.0, 8.0, size=N_PIX))

    # npt (NPT, P): non-Poissonian template spatial weights, positive, O(0.1-5)
    npt = jnp.array(rng.uniform(0.1, 3.0, size=(N_NPT, N_PIX)))

    # data (P,): observed integer photon counts in [0, k_max]
    data = jnp.array(rng.integers(0, K_MAX + 1, size=N_PIX), dtype=jnp.int32)

    # f, rho_df: delta PSF (as used with psf_tag='delta')
    f = jnp.array([0.0, 1.0])
    rho_df = jnp.array([0.0, 1.0])

    return dict(theta=theta, pt_sum=pt_sum, npt=npt, data=data,
                f=f, rho_df=rho_df, k_max=K_MAX, n_pix=N_PIX)


def _run(inputs):
    from fpp.likelihoods.npll_jax import log_like_np
    return log_like_np(
        inputs["theta"], inputs["pt_sum"], inputs["npt"], inputs["data"],
        inputs["f"], inputs["rho_df"], inputs["k_max"], inputs["n_pix"],
    )


def _generate():
    """Generate and save test fixture."""
    inputs = _build_inputs()
    result = _run(inputs)

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(
        NPLL_FIXTURE,
        theta=np.asarray(inputs["theta"]),
        pt_sum=np.asarray(inputs["pt_sum"]),
        npt=np.asarray(inputs["npt"]),
        data=np.asarray(inputs["data"]),
        f=np.asarray(inputs["f"]),
        rho_df=np.asarray(inputs["rho_df"]),
        k_max=inputs["k_max"],
        n_pix=inputs["n_pix"],
        expected_output=np.asarray(result),
    )
    print(f"Saved fixture to {NPLL_FIXTURE}")
    print(f"  output shape: {np.asarray(result).shape}")
    print(f"  output range: [{float(jnp.min(result)):.4f}, {float(jnp.max(result)):.4f}]")


# ------------------------------------------------------------------ #
#  Test                                                               #
# ------------------------------------------------------------------ #

def test_log_like_np_regression():
    """Compare log_like_np output against saved ground truth."""
    from fpp.likelihoods.npll_jax import log_like_np

    assert os.path.exists(NPLL_FIXTURE), (
        f"Fixture not found at {NPLL_FIXTURE}. "
        f"Run: python tests/test_npll.py --generate"
    )

    fixture = np.load(NPLL_FIXTURE)

    result = log_like_np(
        jnp.array(fixture["theta"]),
        jnp.array(fixture["pt_sum"]),
        jnp.array(fixture["npt"]),
        jnp.array(fixture["data"]),
        jnp.array(fixture["f"]),
        jnp.array(fixture["rho_df"]),
        int(fixture["k_max"]),
        int(fixture["n_pix"]),
    )

    expected = fixture["expected_output"]
    result_np = np.asarray(result)
    assert result_np == pytest.approx(expected, rel=1e-10, abs=1e-12)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.generate:
        _generate()
    else:
        print("Use --generate to create fixtures, or run via pytest.")
