import os
import numpy as np
from .compute_norm_grid import build_A_phys_table_parallel_4D


# from .norm_computer.compute_norm_grid import build_A_phys_table_parallel_4D
import cProfile, pstats





import cProfile, pstats
from norm_computer.compute_norm_grid import compute_A_phys_eta
from norm_computer.utils import default_sim_params

profiler = cProfile.Profile()
profiler.enable()

eta = (12.0, 0.2, 1.0, 0.0)
res = compute_A_phys_eta(
    eta,
    sim_params=default_sim_params(),
    n_samples=20000,
    verbose=True,
)

profiler.disable()
print("[RESULT]", res)
stats = pstats.Stats(profiler)
stats.strip_dirs().sort_stats("cumtime").print_stats(30)



# def main():
#     """Compute the A(eta) grid using default parameters."""


#     table_path = os.path.join(os.path.dirname(__file__), '..', 'tables', 'A_phys_table_4D.csv')




#     muDM_grid    = np.linspace(11.5, 13.5, 200)      # Δμ = 0.0345
#     sigmaDM_grid = np.linspace(0.01, 0.5, 200)    # Δσ = 0.0138
#     betaDM_grid  = np.linspace(0, 3.0, 200)    # Δβ = 0.069
#     xiDM_grid    = 0                            # 固定为 0


#     build_A_phys_table_parallel_4D(
#         muDM_grid,
#         sigmaDM_grid,
#         betaDM_grid,
#         xiDM_grid,
#         n_samples=2000,
#         filename=table_path,
#     )


# if __name__ == "__main__":
#     main()
