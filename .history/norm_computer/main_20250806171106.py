import os
import numpy as np
from .compute_norm_grid import build_A_phys_table_parallel_4D


# from .norm_computer.compute_norm_grid import build_A_phys_table_parallel_4D
import cProfile, pstats

def main():
    """Compute the A(eta) grid using default parameters."""
    table_path = os.path.join(os.path.dirname(__file__), '..', 'tables', 'A_phys_table_4D_test.csv')

    muDM_grid    = np.linspace(11.5, 13.5, 2)
    sigmaDM_grid = np.linspace(0.01, 0.5, 2)
    betaDM_grid  = np.linspace(0, 3.0, 2)
    xiDM_grid    = 0

    build_A_phys_table_parallel_4D(
        muDM_grid,
        sigmaDM_grid,
        betaDM_grid,
        xiDM_grid,
        n_samples=2000,
        filename=table_path,
    )

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(50)  # 打印累计时间前20名的函数
    # 如需输出到文件：
    stats.dump_stats('main.txt')
    os.remove(table)


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
