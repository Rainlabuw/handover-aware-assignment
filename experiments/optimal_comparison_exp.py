import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from algorithms.utils import *
from algorithms.solve_w_haal import solve_w_haal
from algorithms.solve_optimally import solve_optimally
from algorithms.solve_wout_handover import solve_wout_handover

"""
Compare various solutions types against the true optimal,
figure 2 in the paper.
"""

def optimal_comparison_exp():
    n = 5
    m = 5
    T = 3
    
    L = T

    init_ass = None
    
    lambda_ = 0.5

    avg_best = 0
    avg_mhal = 0
    avg_mha = 0
    avg_no_handover = 0

    num_avgs = 50
    for _ in tqdm(range(num_avgs)):
        benefit = np.random.rand(n,m,T)

        #HAAL, L=3
        _, mhal_ben, _ = solve_w_haal(benefit, init_ass, lambda_, L)
        avg_mhal += mhal_ben/num_avgs

        #HAA
        _, mha_ben, _ = solve_w_haal(benefit, init_ass, lambda_, 1)
        avg_mha += mha_ben/num_avgs

        #NHA
        _, ben, _ = solve_wout_handover(benefit, init_ass, lambda_)
        avg_no_handover += ben/num_avgs

        #Optimal
        _, ben, _ = solve_optimally(benefit, init_ass, lambda_)
        avg_best += ben/num_avgs

    fig = plt.figure()
    plt.bar([r"HAA ($\lambda = 0$)","HAA", f"HAAL (L={L})", "Optimal"], [avg_no_handover, avg_mha, avg_mhal, avg_best])
    plt.ylabel("Value")
    print(["No Handover","HAA", f"HAAL (L={L})", "Optimal"])
    print([avg_no_handover, avg_mha, avg_mhal, avg_best])
    plt.savefig("opt_comparison.png")
    plt.show()