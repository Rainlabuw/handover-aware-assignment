import matplotlib.pyplot as plt
import numpy as np

from algorithms.utils import *
from algorithms.solve_wout_handover import solve_wout_handover
from algorithms.solve_w_haal import solve_w_haald_track_iters, solve_w_haal
from algorithms.solve_w_CBBA import solve_w_CBBA_track_iters
from algorithms.solve_greedily import solve_greedily

from constellation_sim.ConstellationSim import get_constellation_bens_and_graphs_random_tasks

def midscale_constellation_exp():
    num_planes = 18
    num_sats_per_plane = 18
    m = 450
    T = 93
    altitude = 550
    fov = 60
    isl_dist = 2500

    benefits, graphs = get_constellation_bens_and_graphs_random_tasks(num_planes, num_sats_per_plane, m, T, altitude=altitude, benefit_func=calc_fov_benefits, fov=fov, isl_dist=isl_dist)

    max_L = 6
    lambda_ = 0.5
    init_assign = None

    _, no_handover_val, _ = solve_wout_handover(benefits, init_assign, lambda_)

    _, greedy_val, _ = solve_greedily(benefits, None, lambda_)

    itersd_by_lookahead = []
    valued_by_lookahead = []

    iterscbba_by_lookahead = []
    valuecbba_by_lookahead = []

    valuec_by_lookahead = []
    for L in range(1,max_L+1):
        print(f"lookahead {L}")
        _, cbba_val, _, avg_iters = solve_w_CBBA_track_iters(benefits, init_assign, lambda_, L, graphs=graphs, verbose=True)
        iterscbba_by_lookahead.append(avg_iters)
        valuecbba_by_lookahead.append(cbba_val)
        
        _, d_val, _, avg_iters = solve_w_haald_track_iters(benefits, init_assign, lambda_, L, graphs=graphs, verbose=True)
        itersd_by_lookahead.append(avg_iters)
        valued_by_lookahead.append(d_val)

        _, c_val, _ = solve_w_haal(benefits, init_assign, lambda_, L, distributed=False, verbose=True)
        valuec_by_lookahead.append(c_val)

    fig, axes = plt.subplots(2,1)
    axes[0].plot(range(1,max_L+1), valued_by_lookahead, 'g--', label="HAAL-D")
    axes[0].plot(range(1,max_L+1), valuec_by_lookahead, 'g', label="HAAL")
    axes[0].plot(range(1,max_L+1), valuecbba_by_lookahead, 'b', label="CBBA")
    axes[0].plot(range(1,max_L+1), [no_handover_val]*max_L, 'r', label="NHA")
    axes[0].plot(range(1,max_L+1), [greedy_val]*max_L, 'k', label="GA")
    axes[0].set_ylabel("Total value")
    axes[0].set_xticks(range(1,max_L+1))
    axes[0].set_ylim((0, 1.1*max(valuec_by_lookahead)))
    axes[1].set_xlabel("Lookahead window L")
    axes[0].legend(loc='lower right')

    axes[1].plot(range(1,max_L+1), itersd_by_lookahead, 'g--', label="MHAL-D")
    axes[1].plot(range(1,max_L+1), iterscbba_by_lookahead, 'b', label="CBBA")
    axes[1].set_ylim((0, 1.1*max(itersd_by_lookahead)))
    axes[0].set_xticks(range(1,max_L+1))
    axes[1].set_ylabel("Average iterations")
    axes[1].set_xlabel("Lookahead window")

    fig.set_figwidth(8)
    fig.set_figheight(5)
    plt.savefig("mhal_experiment1/paper_exp1.pdf")
    plt.show()