# === Imports ===
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.stats import norm, skewnorm
from scipy.special import erf
from ..mock_generator.lens_model import LensModel
from ..mock_generator.lens_solver import solve_single_lens
from ..mock_generator.mass_sampler import MODEL_PARAMS, sample_m_s

MODEL_P = MODEL_PARAMS["deVauc"]
# === Utils ===
# to check. use it to model the relation between logM_sps and logRe
# def logRe_of_logMsps(logMsps):
#     """根据 logM_sps 估计 logRe"""
#     a = 0.6181
#     b = -6.2756
#     return a * logMsps + b


def logRe_of_logMsps(logMsps, model='deVauc'):
    """
    使用 Sonnenfeld+2019 的 Re–M* 关系估计 logRe 的均值
    """
    p = MODEL_PARAMS[model]
    return p['mu_R0'] + p['beta_R'] * (logMsps - 11.4)


# === Sample Generation ===

def generate_lens_samples_no_alpha(
    n_samples=1000,
    seed=42,
    mu_DM=13.0,
    sigma_DM=0.2,
    n_sigma=5,
    alpha_s=-1.3,
    m_s_star=24.5,
):
    """按照先验生成透镜样本（不包含 ``alpha_sps`` 和 ``logMh``）。

    本函数只负责从真实的先验分布中抽取与透镜本身有关的
    随机量（如 ``logM_star``、``logRe``、源位置和源星等）。
    ``logMh`` 将在后续的物理响应表中离散化处理，因此不再
    在采样阶段生成。

    Parameters
    ----------
    n_samples : int
        需要生成的样本数。
    seed : int, optional
        随机数种子。
    mu_DM, sigma_DM : float
        暗物质先验的均值与标准差（仅用于确定 ``logMh`` 网格
        的范围，采样阶段并未使用）。
    n_sigma : int
        ``logMh`` 网格范围的倍数。
    alpha_s, m_s_star : float
        Schechter-like 源星等分布的参数。

    Returns
    -------
    samples : list[tuple]
        ``(logM_star, logRe, beta, m_s)`` 的列表。
    Mh_range : tuple
        ``(Mh_min, Mh_max)`` 供构建 ``logMh`` 网格时参考。
    """

    rng = np.random.default_rng(seed)

    # --- Stellar mass from skew-normal prior ---
    a_skew = 10 ** MODEL_P["log_s_star"]
    logMstar = skewnorm.rvs(
        a=a_skew, loc=MODEL_P["mu_star"], scale=MODEL_P["sigma_star"],
        size=n_samples, random_state=rng
    )

    # --- Effective radius conditional on stellar mass ---
    mu_Re = logRe_of_logMsps(logMstar)
    logRe = rng.normal(loc=mu_Re, scale=MODEL_P["sigma_R"], size=n_samples)

    # --- Range of halo mass for later grid construction ---
    Mh_min = mu_DM - n_sigma * sigma_DM
    Mh_max = mu_DM + n_sigma * sigma_DM

    # --- Source position ---
    beta = rng.uniform(0.0, 1.0, n_samples)

    # --- Source magnitude ---
    m_s = sample_m_s(alpha_s, m_s_star, size=n_samples, rng=rng)

    return list(zip(logMstar, logRe, beta, m_s)), (Mh_min, Mh_max)

# === Physical response table ===
def build_physical_response_table(samples, logMh_grid, logalpha_grid,
                                  zl=0.3, zs=2.0):
    """为给定透镜样本构建物理响应表。

    该步骤与超参数 ``η`` 无关，只计算在 ``(logMh, logalpha)`` 网格上
    每个透镜的放大率等物理量，并缓存结果以便后续重复利用。

    Parameters
    ----------
    samples : list[tuple]
        ``(logM_star, logRe, beta, m_s)`` 的样本列表。
    logMh_grid, logalpha_grid : array-like
        物理量空间的网格定义。
    zl, zs : float
        透镜和源的红移。

    Returns
    -------
    muA_table, muB_table : ndarray
        形状为 ``(N_lens, N_Mh, N_alpha)`` 的放大率表。
    """

    n_lens = len(samples)
    n_Mh = len(logMh_grid)
    n_alpha = len(logalpha_grid)
    muA = np.full((n_lens, n_Mh, n_alpha), np.nan)
    muB = np.full((n_lens, n_Mh, n_alpha), np.nan)

    for i, (logMstar, logRe, beta, _) in enumerate(samples):
        for j, logMh in enumerate(logMh_grid):

                    # 当前 LensModel 尚未使用 logalpha 参数，预留接口
                    model = LensModel(logMstar, logMh, logRe, zl=zl, zs=zs)
                    xA, xB = solve_single_lens(model, beta_unit=beta)
                    muA[i, j, k] = model.mu_from_rt(xA)
                    muB[i, j, k] = model.mu_from_rt(xB)
                except Exception:
                    continue

    return muA, muB


# === Hyper-parameter weighting ===
def compute_A_eta_from_table(muA_table, muB_table, samples, logMh_grid,
                             logalpha_grid, mu_DM_cnst, beta_DM, xi_DM,
                             sigma_DM, sigma_m=0.1, m_lim=26.5,
                             p_logalpha=None):
    """根据预计算的物理响应表和超参数 ``η`` 计算 ``A(η)``。

    Parameters
    ----------
    muA_table, muB_table : ndarray
        由 :func:`build_physical_response_table` 生成的放大率表。
    samples : list[tuple]
        ``(logM_star, logRe, beta, m_s)`` 的样本列表。
    logMh_grid, logalpha_grid : array-like
        物理网格定义。
    mu_DM_cnst, beta_DM, xi_DM, sigma_DM : float
        描述 ``P(logMh | logM_star)`` 的超参数。
    sigma_m, m_lim : float
        选择函数参数。
    p_logalpha : array-like, optional
        ``logalpha`` 的概率分布，若为 ``None`` 则视为均匀分布。

    Returns
    -------
    float
        ``A(η)`` 的估计值。
    """

    samples_array = np.asarray(samples)
    if samples_array.size == 0:
        return 0.0

    logMstar_array, logRe_array, beta_array, m_s_array = samples_array.T

    # DM conditional mean for each lens
    logRe_model_array = logRe_of_logMsps(logMstar_array)
    mu_DM_i_array = (
        mu_DM_cnst
        + beta_DM * (logMstar_array - 11.4)
        + xi_DM * (logRe_array - logRe_model_array)
    )

    # Probability over logalpha; default uniform
    if p_logalpha is None:
        p_logalpha = np.ones(len(logalpha_grid)) / len(logalpha_grid)
    else:
        p_logalpha = np.asarray(p_logalpha)

    total = 0.0
    n_lens = len(samples)

    for idx in range(n_lens):
        p_Mh = norm.pdf(logMh_grid, loc=mu_DM_i_array[idx], scale=sigma_DM)
        weight = p_Mh[:, None] * p_logalpha[None, :]

        muA = muA_table[idx]
        muB = muB_table[idx]

        magA = m_s_array[idx] - 2.5 * np.log10(muA)
        magB = m_s_array[idx] - 2.5 * np.log10(muB)
        selA = 0.5 * (1 + erf((m_lim - magA) / (np.sqrt(2) * sigma_m)))
        selB = 0.5 * (1 + erf((m_lim - magB) / (np.sqrt(2) * sigma_m)))
        sel = selA * selB

        valid = np.isfinite(sel)
        weighted = sel[valid] * weight[valid]
        norm_factor = weight[valid].sum()
        if norm_factor > 0:
            total += weighted.sum() / norm_factor

    return total / n_lens


def compute_A_phys_eta(mu_DM_cnst, beta_DM, xi_DM, sigma_DM, samples,
                       logMh_grid, logalpha_grid, zl=0.3, zs=2.0,
                       sigma_m=0.1, m_lim=26.5, p_logalpha=None):
    """Convenience wrapper performing both stages for a single ``η``.

    This function is retained for compatibility with earlier code but now
    merely orchestrates the two-step procedure:

    1. 生成物理响应表 (:func:`build_physical_response_table`)
    2. 根据超参数加权求和 (:func:`compute_A_eta_from_table`)
    """

    muA_table, muB_table = build_physical_response_table(
        samples, logMh_grid, logalpha_grid, zl=zl, zs=zs
    )
    return compute_A_eta_from_table(
        muA_table, muB_table, samples, logMh_grid, logalpha_grid,
        mu_DM_cnst, beta_DM, xi_DM, sigma_DM,
        sigma_m=sigma_m, m_lim=m_lim, p_logalpha=p_logalpha,
    )
# === 单点计算任务 ===

def single_A_eta_entry(args, seed=None, logalpha_grid=None, n_mh=100):
    """Compute a single entry of :math:`A_\text{phys}(\eta)`.

    Parameters
    ----------
    args : tuple
        ``(muDM, sigmaDM, beta_DM, xi_DM, n_samples, n_sigma)`` defining
        the physical model.
    seed : int, optional
        Base random seed for sample generation.
    logalpha_grid : array-like, optional
        ``logalpha`` 网格；若为 ``None`` 则默认单点 ``0``。
    n_mh : int
        ``logMh`` 网格的长度。
    """

    muDM, sigmaDM, beta_DM, xi_DM, n_samples, n_sigma = args
    base_seed = os.getpid() if seed is None else seed
    unique_seed = (base_seed + hash(args)) % 2**32

    samples, Mh_range = generate_lens_samples_no_alpha(
        n_samples=n_samples,
        mu_DM=muDM,
        sigma_DM=sigmaDM,
        n_sigma=n_sigma,
        seed=unique_seed,
    )
    Mh_min, Mh_max = Mh_range
    logMh_grid = np.linspace(Mh_min, Mh_max, n_mh)
    if logalpha_grid is None:
        logalpha_grid = np.array([0.0])

    muA_table, muB_table = build_physical_response_table(
        samples, logMh_grid, logalpha_grid
    )
    A_eta = compute_A_eta_from_table(
        muA_table, muB_table, samples, logMh_grid, logalpha_grid,
        mu_DM_cnst=muDM, beta_DM=beta_DM, xi_DM=xi_DM, sigma_DM=sigmaDM,
    )
    return {
        'mu_DM': muDM,
        'sigma_DM': sigmaDM,
        'beta_DM': beta_DM,
        'xi_DM': xi_DM,
        'A_phys': A_eta,
    }

# === 并行构建 A_phys 表格 ===

# def build_A_phys_table_parallel_4D(muDM_grid, sigmaDM_grid, betaDM_grid, xiDM_grid,
#                                     n_samples=1000, n_sigma=3,
#                                     filename='A_phys_table_4D.csv', nproc=None, batch_size=1000):
#     if nproc is None:
#         nproc = max(1, cpu_count() - 1)
#     filename = os.path.join(os.path.dirname(__file__), '..', 'tables', filename)
#     done_set = set()
#     if os.path.exists(filename):
#         df_done = pd.read_csv(filename)
#         done_set = set(zip(df_done['mu_DM'], df_done['sigma_DM'],
#                            df_done['beta_DM'], df_done['xi_DM']))
#         print(f"[INFO] 已完成 {len(done_set)} 个点，将跳过这些")

#     args_list = [
#         (mu, sigma, beta, xi, n_samples, n_sigma)
#         for mu in muDM_grid
#         for sigma in sigmaDM_grid
#         for beta in betaDM_grid
#         for xi in xiDM_grid
#         if (mu, sigma, beta, xi) not in done_set
#     ]
#     print(f"[INFO] 共需计算 {len(args_list)} 个 A(eta) 点")

#     with Pool(nproc) as pool:
#         buffer = []
#         with open(filename, 'a') as f:
#             if os.stat(filename).st_size == 0:
#                 f.write('mu_DM,sigma_DM,beta_DM,xi_DM,A_phys\n')

#             for result in tqdm(pool.imap_unordered(single_A_eta_entry, args_list), total=len(args_list)):
#                 buffer.append(f"{result['mu_DM']},{result['sigma_DM']},{result['beta_DM']},{result['xi_DM']},{result['A_phys']}\n")
#                 if len(buffer) >= batch_size:
#                     f.writelines(buffer)
#                     f.flush()
#                     buffer = []

#             if buffer:
#                 f.writelines(buffer)
#                 f.flush()

#     print(f"[INFO] 所有任务完成，结果已保存到 {filename}")

def _worker(arg_seed):
    params, seed = arg_seed
    return single_A_eta_entry(params, seed=seed)

def key4(mu, s, b, x, prec):
        return (round(mu, prec), round(s, prec), round(b, prec), round(x, prec))
    
def build_A_phys_table_parallel_4D(
    muDM_grid,
    sigmaDM_grid,
    betaDM_grid,
    xiDM_grid,
    n_samples=1000,
    n_sigma=3,
    filename="A_phys_table_4D.csv",
    nproc=None,
    batch_size=1000,
    prec=6,
):
    """构建 ``A_phys(eta)`` 插值表（四维），并行运行。

    使用浮点精度量化避免 ``(mu, sigma, beta, xi)`` 比较失败。

    Notes
    -----
    For reproducibility each parameter tuple is paired with a unique
    integer seed equal to its enumeration index.  ``single_A_eta_entry``
    mixes this seed with a hash of the tuple, yielding deterministic and
    independent random streams across workers.
    """
    print(f"[DEBUG] compute_A_phys_eta called with η = , n_samples = {n_samples}")

    if nproc is None:
        nproc = max(1, cpu_count() - 1)

    # 包裹标量 xiDM_grid 为列表
    if np.isscalar(xiDM_grid):
        xiDM_grid = [float(xiDM_grid)]

    # === 浮点量化 key 工具 ===


    # === 已完成点集 ===
    done_set = set()
    if os.path.exists(filename):
        df_done = pd.read_csv(filename)
        done_set = set(key4(*row, prec) for row in df_done[['mu_DM','sigma_DM','beta_DM','xi_DM']].to_numpy())
        print(f"[INFO] 已完成 {len(done_set)} 个点，将跳过这些")

    # === 待计算参数列表并分配种子 ===
    param_list = [
        (mu, sigma, beta, xi, n_samples, n_sigma)
        for mu in muDM_grid
        for sigma in sigmaDM_grid
        for beta in betaDM_grid
        for xi in xiDM_grid
        if key4(mu, sigma, beta, xi, prec) not in done_set
    ]
    args_list = [
        (params, i)  # unique seed per job
        for i, params in enumerate(param_list)
    ]
    print(f"[INFO] 共需计算 {len(args_list)} 个 A(eta) 点")



    # === 并行计算 ===
    with Pool(nproc) as pool:
        buffer = []
        with open(filename, 'a') as f:
            if os.stat(filename).st_size == 0:
                f.write('mu_DM,sigma_DM,beta_DM,xi_DM,A_phys\n')

            for result in tqdm(pool.imap_unordered(_worker, args_list), total=len(args_list)):
                buffer.append(
                    f"{result['mu_DM']},{result['sigma_DM']},{result['beta_DM']},{result['xi_DM']},{result['A_phys']}\n"
                )
                if len(buffer) >= batch_size:
                    f.writelines(buffer)
                    f.flush()
                    buffer = []

            if buffer:
                f.writelines(buffer)
                f.flush()

    print(f"[INFO] 所有任务完成，结果已保存到 {filename}")
    
# === 主程序入口 ===

if __name__ == "__main__":
    muDM_grid    = np.linspace(11.5, 13.5, 200)      # Δμ = 0.0345
    sigmaDM_grid = np.linspace(0.01, 0.5, 200)    # Δσ = 0.0138
    betaDM_grid  = np.linspace(0, 3.0, 200)    # Δβ = 0.069
    xiDM_grid    = 0                            # 固定为 0


    build_A_phys_table_parallel_4D(
        muDM_grid, sigmaDM_grid, betaDM_grid, xiDM_grid,
        n_samples=2000,
        filename="A_phys_table_4D_new_with_pMstar.csv"
    )
