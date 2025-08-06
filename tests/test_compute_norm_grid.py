import numpy as np
from scipy.stats import skewnorm, multivariate_normal
from scipy.special import erf

from norm_computer import compute_norm_grid as cn


def test_importance_weighting_unbiased(monkeypatch):
    """Monte Carlo estimator remains unbiased with mismatched sampling."""
    # Stub lens model and solver so selection depends on logMstar only
    k_prime = np.sqrt(2) / 2.5

    class DummyModel:
        def __init__(self, logMstar, logMh, logRe, zl=0.3, zs=2.0):
            self.logMstar = logMstar

        def mu_from_rt(self, x):
            # Produces selection: 0.5*(1+erf(logMstar-11.4))
            return 10 ** (k_prime * (self.logMstar - 11.4))

    def dummy_solver(model, beta_unit):
        return 0.0, 0.0

    monkeypatch.setattr(cn, "LensModel", DummyModel)
    monkeypatch.setattr(cn, "solve_single_lens", dummy_solver)

    # Samples drawn from simple Gaussian distributions
    n_samples = 20000
    samples, Mh_range = cn.generate_lens_samples_no_alpha(
        n_samples=n_samples, seed=0
    )

    estimate = cn.compute_A_phys_eta(
        mu_DM_cnst=13.0,
        beta_DM=0.0,
        xi_DM=0.0,
        sigma_DM=0.2,
        samples=samples,
        Mh_range=Mh_range,
        ms=0.0,
        sigma_m=1.0,
        m_lim=0.0,
    )

    # Ground truth from target distribution
    rng = np.random.default_rng(0)
    a_skew = 10 ** cn.MODEL_P["log_s_star"]
    mu_star = cn.MODEL_P["mu_star"]
    sigma_star = cn.MODEL_P["sigma_star"]
    logMstar_target = skewnorm.rvs(
        a=a_skew, loc=mu_star, scale=sigma_star, size=200000, random_state=rng
    )
    g = 0.5 * (1 + erf(logMstar_target - 11.4))
    true_val = np.mean(g ** 2)

    assert np.isclose(estimate, true_val, rtol=0.02)


def test_ms_marginalization(monkeypatch):
    """Selection is correctly marginalized over a Gaussian prior of ms."""

    class DummyModel:
        def __init__(self, logMstar, logMh, logRe, zl=0.3, zs=2.0):
            pass

        def mu_from_rt(self, x):
            return 1.0

    def dummy_solver(model, beta_unit):
        return 0.0, 0.0

    monkeypatch.setattr(cn, "LensModel", DummyModel)
    monkeypatch.setattr(cn, "solve_single_lens", dummy_solver)

    samples, Mh_range = cn.generate_lens_samples_no_alpha(n_samples=10, seed=1)

    ms_mean = 0.0
    sigma_m = 0.1
    sigma_ms = 0.3
    m_lim = 0.2

    estimate = cn.compute_A_phys_eta(
        mu_DM_cnst=13.0,
        beta_DM=0.0,
        xi_DM=0.0,
        sigma_DM=0.2,
        samples=samples,
        Mh_range=Mh_range,
        ms=ms_mean,
        sigma_m=sigma_m,
        m_lim=m_lim,
        sigma_ms=sigma_ms,
    )

    sigma_tot = np.sqrt(sigma_m ** 2 + sigma_ms ** 2)
    z = (m_lim - ms_mean) / sigma_tot
    rho = sigma_ms ** 2 / (sigma_tot ** 2)
    mvn = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    expected = mvn.cdf([z, z])

    assert np.isclose(estimate, expected, atol=1e-6)
