from __future__ import annotations

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from src.fcnet_benchmark import FCNetBenchmark
from src.problem import Problem
from src.successive_halving import successive_halving


# Hint 1: use pseudocode from https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/18-ICML-BOHB.pdf
def hyperband(
    problem: Problem,
    min_budget_per_model: int,
    max_budget_per_model: int,
    eta: float,
    random_seed: int | None = None,
) -> list:
    """The hyperband algorithm

    Parameters
    ----------
    problem : Problem
        A problem instance to run on

    min_budget_per_model : int
        The minimum budget per model

    max_budget_per_model : int
        The maximum budget per model

    eta : float
        The eta float parameter. The budget is multiplied by eta at each iteration

    random_seed : int | None = None
        The random seed to use

    Returns
    -------
    list[dict]
        A list of dictionaries with the config information
    """
    if min_budget_per_model >= max_budget_per_model:
        raise ValueError("min_budget_per_model must be less than max_budget_per_model")
    
    if eta <= 1:
        raise ValueError("eta must be greater than 1")
    
    # Compute s_max
    s_max = int(np.log(max_budget_per_model / min_budget_per_model) / np.log(eta))
    
    configs_dicts = []
    
    # Iterate through the brackets in reverse order
    for s in reversed(range(s_max + 1)):
        # Compute the number of configurations to evaluate
        n = int(np.ceil((s_max + 1) * (eta ** s) / (s + 1)))
        
        # Compute the minimum budget for this bracket
        min_budget = int(min_budget_per_model * (eta ** s))
        
        # Run successive halving with these parameters
        configs_dict = successive_halving(
            problem=problem,
            n_models=n,
            min_budget_per_model=min_budget,
            max_budget_per_model=max_budget_per_model,
            eta=eta,
            random_seed=random_seed,
        )
        configs_dicts.append(configs_dict)
    
    return configs_dicts


# TODO: Plot Hyperband results
def plot_hyperband_results(configs_dicts):
    """Plot the results from multiple successive halving runs.
    
    Parameters
    ----------
    configs_dicts : list[dict]
        List of dictionaries containing configuration evaluations from each SH run
    """
    plt.figure(figsize=(10, 6))
    
    for i, configs_dict in enumerate(configs_dicts):
        # Collect all unique budgets across all configurations
        all_budgets = set()
        for config_info in configs_dict.values():
            _, evaluations = config_info
            all_budgets.update(evaluations.keys())
        
        budgets = sorted(all_budgets)
        best_scores = []
        
        for budget in budgets:
            scores = []
            for config_info in configs_dict.values():
                _, evaluations = config_info
                if budget in evaluations:
                    scores.append(evaluations[budget].y)
            if scores:
                best_scores.append(min(scores))
        
        if budgets and best_scores:  # Only plot if we have data
            plt.plot(budgets, best_scores, 'o-', label=f'SH Iteration {i}')
    
    plt.xlabel('Budget')
    plt.ylabel('Best Score')
    plt.title('Hyperband Results')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/hyperband_results.pdf')
    plt.close()


if __name__ == "__main__":
    try:
        problem = FCNetBenchmark(name="protein_structures")
        configs_dicts = hyperband(
            problem=problem,
            eta=2,
            random_seed=0,
            max_budget_per_model=100,
            min_budget_per_model=2,
        )
        plot_hyperband_results(configs_dicts)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
