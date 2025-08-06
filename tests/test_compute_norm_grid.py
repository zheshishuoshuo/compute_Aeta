import numpy as np
from scipy.stats import skewnorm
from scipy.special import erf

from norm_computer import compute_norm_grid as cn
from mock_generator.mass_sampler import sample_m_s


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
        n_samples=n_samples, seed=0, mu_DM=13.0, sigma_DM=0.2
    )
    Mh_min, Mh_max = Mh_range
    logMh_grid = np.linspace(Mh_min, Mh_max, 50)

    muA_tab, muB_tab = cn.build_physical_response_table(
        samples, logMh_grid
    )

    estimate = cn.compute_A_eta_from_table(
        muA_tab,
        muB_tab,
        samples,
        logMh_grid,
        mu_DM_cnst=13.0,
        beta_DM=0.0,
        xi_DM=0.0,
        sigma_DM=0.2,
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
    ms_target = sample_m_s(-1.3, 24.5, size=200000, rng=rng)
    mu = 10 ** (k_prime * (logMstar_target - 11.4))
    mag = ms_target - 2.5 * np.log10(mu)
    g = 0.5 * (1 + erf((0.0 - mag) / (np.sqrt(2) * 1.0)))
    true_val = np.mean(g ** 2)

    assert np.isclose(estimate, true_val, rtol=0.05)
