import matplotlib.pyplot as plt
import numpy as np

from algorithms.utils import *
from algorithms.solve_wout_handover import solve_wout_handover
from algorithms.solve_w_haal import solve_w_haal
from algorithms.solve_greedily import solve_greedily

from constellation_sim.ConstellationSim import get_constellation_bens_and_graphs_coverage

"""
Python code to replicate the results of Figure 5, the large-scale constellation experiment in the paper.
"""

def largescale_constellation_exp():
    num_planes = 40
    num_sats_per_plane = 25
    altitude=550 #km
    fov=60 #deg
    T = 93
    inc = 70 #deg
    isl_dist = 2500 #km

    L = 6
    init_assign = None
    lambda_ = 0.5

    benefits, graphs = get_constellation_bens_and_graphs_coverage(num_planes, num_sats_per_plane, T, inc, altitude=altitude, benefit_func=calc_fov_benefits, fov=fov, isl_dist=isl_dist)
    
    #Add phantom tasks with zero benefits so we have 1000 tasks and n<=m
    m = benefits.shape[1]
    symmetric_benefits = np.zeros((num_planes*num_sats_per_plane, num_planes*num_sats_per_plane, T))
    symmetric_benefits[:,:m,:] = benefits
    benefits = symmetric_benefits.as_type(np.float16) #reduce memory usage

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~APPLY ALGORITHMS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    haal_assigns, haal_value, haal_nh = solve_w_haal(benefits, init_assign, lambda_, L, graphs=graphs, distributed=True, verbose=True)
    haal_assigns = [ha.as_type(np.int8) for ha in haal_assigns] #reduce memory usage

    greedy_assigns, greedy_value, greedy_nh = solve_greedily(benefits, init_assign, lambda_)
    greedy_assigns = [ga.as_type(np.int8) for ga in greedy_assigns] #reduce memory usage

    nohand_assigns, nohand_value, nohand_nh = solve_wout_handover(benefits, init_assign, lambda_)
    nohand_assigns = [nha.as_type(np.int8) for nha in nohand_assigns] #reduce memory usage

    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLOT OF BENEFITS CAPTURED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, axes = plt.subplots(3,1)
    no_handover_ax = axes[0]
    greedy_ax = axes[1]
    haal_ax = axes[2]
    

    prev_no_handover = 0
    prev_haal = 0
    prev_greedy = 0

    no_handover_ben_line = []
    haal_ben_line = []
    greedy_ben_line = []
    
    agent_to_investigate = 900 #change this to change agent to investigate benefits for
    for k in range(T):
        no_handover_choice = np.argmax(nohand_assigns[k][agent_to_investigate,:])
        haal_choice = np.argmax(haal_assigns[k][agent_to_investigate,:])
        greedy_choice = np.argmax(greedy_assigns[k][agent_to_investigate,:])

        if prev_no_handover != no_handover_choice:
            no_handover_ax.axvline(k-0.5, linestyle='--')
            if k != 0: 
                if len(no_handover_ben_line) > 1:
                    no_handover_ax.plot(range(k-len(no_handover_ben_line), k), no_handover_ben_line, 'r')
                elif len(no_handover_ben_line) == 1:
                    no_handover_ax.plot(range(k-len(no_handover_ben_line), k), no_handover_ben_line, 'r.', markersize=1)
            no_handover_ben_line = [benefits[agent_to_investigate,no_handover_choice, k]]
        else:
            no_handover_ben_line.append(benefits[agent_to_investigate, no_handover_choice, k])

        if prev_haal != haal_choice:
            vline = haal_ax.axvline(k-0.5, linestyle='--')
            vline_color = vline.get_color()
            if k != 0: 
                if len(haal_ben_line) > 1:
                    haal_ax.plot(range(k-len(haal_ben_line), k), haal_ben_line,'g')
                elif len(haal_ben_line) == 1:
                    haal_ax.plot(range(k-len(haal_ben_line), k), haal_ben_line,'g.', markersize=1)
            haal_ben_line = [benefits[agent_to_investigate,haal_choice, k]]
        else:
            haal_ben_line.append(benefits[agent_to_investigate,haal_choice, k])

        if prev_greedy != greedy_choice:
            greedy_ax.axvline(k-0.5, linestyle='--')
            if k != 0: 
                if len(greedy_ben_line) > 1:
                    greedy_ax.plot(range(k-len(greedy_ben_line), k), greedy_ben_line,'b')
                elif len(greedy_ben_line) == 1:
                    greedy_ax.plot(range(k-len(greedy_ben_line), k), greedy_ben_line,'b.', markersize=1)
            greedy_ben_line = [benefits[agent_to_investigate,greedy_choice, k]]
        else:
            greedy_ben_line.append(benefits[agent_to_investigate,greedy_choice, k])

        prev_no_handover = no_handover_choice
        prev_haal = haal_choice
        prev_greedy = greedy_choice

    #plot last interval
    no_handover_ax.plot(range(k+1-len(no_handover_ben_line), k+1), no_handover_ben_line, 'r')
    haal_ax.plot(range(k+1-len(haal_ben_line), k+1), haal_ben_line,'g')
    greedy_ax.plot(range(k+1-len(greedy_ben_line), k+1), greedy_ben_line,'b')
    haal_ax.set_xlim([0, T])
    greedy_ax.set_xlim([0, T])
    no_handover_ax.set_xlim([0, T])
    haal_ax.set_xticks(range(0,T+1,10))
    greedy_ax.set_xticks(range(0,T+1,10))
    no_handover_ax.set_xticks(range(0,T+1,10))
    haal_ax.set_ylim([-0.1, 2])
    greedy_ax.set_ylim([-0.1, 2])
    no_handover_ax.set_ylim([-0.1, 2])
    haal_ax.set_ylabel("HAAL-D\nBenefit Captured")
    greedy_ax.set_ylabel("GA\nBenefit Captured")
    no_handover_ax.set_ylabel("NHA\nBenefit Captured")

    #phantom lines for legend
    haal_ax.plot([T+1], [0], color=vline_color, linestyle='--', label="Task Changes")
    haal_ax.legend(loc="upper left",bbox_to_anchor=(0,-0.15))
    haal_ax.set_xlabel("Timestep")
    plt.savefig("large_const_task_hist.pdf")
    

    #~~~~~~~~~Plot bar charts~~~~~~~~~~~~~~
    #plot value over time
    fig, axes = plt.subplots(2,1)
    labels = ("NHA", "GA", "HAAL-D")
    val_vals = (nohand_value, greedy_value, haal_value)
    nh_vals = (nohand_nh, greedy_nh, haal_nh)

    val_bars = axes[0].bar(labels, val_vals)
    axes[0].set_ylabel("Total Value")
    nh_bars = axes[1].bar(labels, nh_vals)
    axes[1].set_ylabel("Total Handovers")

    val_bars[0].set_color('r')
    nh_bars[0].set_color('r')
    val_bars[1].set_color('b')
    nh_bars[1].set_color('b')
    val_bars[2].set_color('g')
    nh_bars[2].set_color('g')
    plt.savefig("large_const_bars.pdf")
    plt.show()