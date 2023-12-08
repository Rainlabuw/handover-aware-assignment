# handover-aware-assignment
Python implementation for Handover Aware Assignment with Lookahead (HAAL), and the distributed version HAAL-D, along with related algorithms.

# Installation

Install all necessary Python depencies using `pip install -r requirements.txt`.

# Repository Structure

This repository has 3 main sections:

## Algorithms

This folder has files corresponding to each algorithm mentioned in the HAAL paper, including:
 - HAA ((`solve_w_haal.py`))
 - HAAL (`solve_w_haal.py`)
 - HAAL-D (`solve_w_haal.py`)
 - NHA (`solve_wout_handover.py`)
 - CBBA (`solve_w_CBBA.py`)
 - GA (`solve_greedily.py`)
 - Optimal (`solve_optimally.py`) (only on very small problems)

Each algorithm can be applied to a problem setting with a simple function call of the following form:

solve_w_{alg}(benefits, initial_assignment, lambda_, L (where applicable)).

`utils.py` contains a variety of useful helper functions used throughout.

## Constellation Simulation

This folder contains the necessary files to run large-scale satellite constellation simulations.

In essence, these functions provide the ability to create realistic benefit matrices on which to apply our algorithms.

## Experiments

This folder contains the code necessary to generate the figures used for the experiments in the paper.

It also provides lightweight examples of the usage of many of the functions in the repository.