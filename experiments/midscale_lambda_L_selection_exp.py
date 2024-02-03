import matplotlib.pyplot as plt
import numpy as np

from algorithms.utils import *
from algorithms.solve_wout_handover import solve_wout_handover
from algorithms.solve_w_haal import solve_w_haald_track_iters, solve_w_haal
from algorithms.solve_w_CBBA import solve_w_CBBA_track_iters
from algorithms.solve_greedily import solve_greedily

from constellation_sim.ConstellationSim import get_constellation_bens_and_graphs_random_tasks

def midscale_lambda_L_selection_exp():
    num_planes = 18
    num_sats_per_plane = 18
    m = 450
    T = 93
    altitude = 550
    fov = 60
    isl_dist = 2500

    benefits, _ = get_constellation_bens_and_graphs_random_tasks(num_planes, num_sats_per_plane, m, T, altitude=altitude, benefit_func=calc_fov_benefits, fov=fov, isl_dist=isl_dist)
    avg_pass_len, avg_pass_ben = calc_pass_statistics(benefits)

    lambdas_over_bens = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    max_L = 6

    haal_ass_lengths = []
    haal_benefit_captured = []
    for L in range(1, max_L+1):
        ass_lengths = []
        ben_captured = []
        for lambda_over_ben in lambdas_over_bens:
            print(f"\nL: {L}, lambda_over_ben: {lambda_over_ben}")

            assigns, _, _ = solve_w_haal(benefits, None, lambda_over_ben/avg_pass_ben, L, verbose=True)

            _, _, avg_ass_len = calc_pass_statistics(benefits, assigns)
            ass_lengths.append(avg_ass_len)

            ben, _ = calc_value_and_num_handovers(assigns, benefits, None, 0)
            ben_captured.append(ben)
        
        haal_ass_lengths.append(ass_lengths)
        haal_benefit_captured.append(ben_captured)

    fig, axes = plt.subplots(2,1, sharex=True)
    for ax in axes:
        ax.grid(True)
    
    viridis_cmap = plt.get_cmap('viridis')
    for L_idx in range(max_L):
        color = viridis_cmap(L_idx /(max_L-1))
        axes[0].plot(lambdas_over_bens, haal_ass_lengths[L_idx], label=f"L={L_idx+1}", color=color)
        axes[1].plot(lambdas_over_bens, haal_benefit_captured[L_idx], label=f"L={L_idx+1}", color=color)
    axes[0].plot(lambdas_over_bens, [avg_pass_len]*len(lambdas_over_bens), 'k--', label="Avg. time\ntask in view")

    axes[0].set_ylabel("Avg. Length Satellite\nAssigned to Same Task")
    axes[1].set_ylabel("Total Benefit")
    axes[1].set_xlabel(r'$\frac{\lambda}{\beta_{pass, avg}}$')
    axes[1].xaxis.label.set_fontsize(16)
    axes[0].set_xlim(min(lambdas_over_bens), max(lambdas_over_bens))
    axes[1].set_xlim(min(lambdas_over_bens), max(lambdas_over_bens))

    handles, labels = [], []
    for handle, label in zip(*axes[0].get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
    
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.45,0.5), ncol=2)

    fig.set_figwidth(6)
    fig.set_figheight(6)
    plt.tight_layout()
    plt.show()