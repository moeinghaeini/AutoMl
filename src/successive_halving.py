from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration
import matplotlib.pyplot as plt

from src.fcnet_benchmark import FCNetBenchmark
from src.problem import Evaluation, Problem


def successive_halving(
    problem: Problem,
    n_models: int,
    min_budget_per_model: int,
    max_budget_per_model: int,
    eta: float,
    random_seed: int | None = None,
) -> dict[int, tuple[Configuration, dict[int, Evaluation]]]:
    """The successive_halving routine as called by hyperband

    Parameters
    ----------
    problem : Problem
        A problem instance to evaluate on

    n_models : int
        How many models to use

    min_budget_per_model : int
        The minimum budget per model

    max_budget_per_model : int
        The maximum budget per model

    eta : float
        The eta float parameter to use. The budget is multiplied by eta at each iteration

    random_seed : int | None = None
        The random seed to use

    Returns
    -------
    dict[int, dict]
        A dictionary mapping from the model id as a integer to the config of that model
    """
    np.random.seed(random_seed)

    cs = problem.get_configuration_space()

    configs = {id: (cs.sample_configuration(), {}) for id in range(n_models)}

    # configs is a list that looks like this :
    # {
    #   1: (config, {2: Evaluation, 5: Evaluation, ..., 100: Evaluation}),
    #   ...,
    #   77: (config, {2: Evaluation, 5: Evaluation, ..., 100: Evaluation}),
    #   ...,
    #   100: (config, {2: Evaluation, 3: Evaluation, ..., 99: Evaluation}),
    # }
    # We'll call the (config, {i: Evaluation, j: Evaluation, ...}) part the `info`
    #
    # The 2, 5, 100 numbers are the `budgets` under which the evaluation was done
    # The 1, 77, 100 are the `ids` of the models.
    #
    # for id, (config, info) in configs.items():
    #   for budget, evaluation in info.items():
    #       print(f"model {id} with config {config} evaluated on budget {budget}
    #           has a runtime of {evaluation.runtime} and score of {evaluation.y}")
    #

    # This will stay as a running list of configurations we wish to keep evaluating
    configs_to_eval = list(range(n_models))

    budget = int(min_budget_per_model)

    while budget <= max_budget_per_model:

        # Evaluate the configs selected for this budget
        for id in configs_to_eval:
            config, evaluations = configs[id]
            evaluations[budget] = problem.objective_function(config, budget=budget)
            # hint : the (config, evaluations) tuple is a shallow copy of configs[id].
            # Therefore, the modifications to evaluations will be reflected in configs

        # TODO: Compute number of configs to proceed to next higher budget
        num_configs_to_proceed = max(1, int(len(configs_to_eval) / eta))

        # TODO: Select the configs which have been evaluated on the current budget
        configs_evaluated_with_budget = [id for id in configs_to_eval if budget in configs[id][1]]

        # TODO: Out of these configs select the ones to proceed to the next higher budget
        configs_to_eval = sorted(configs_evaluated_with_budget, key=lambda _id: configs[_id][1][budget].y)[:num_configs_to_proceed]

        # TODO: Increase the budget for the next SH iteration
        budget = int(budget * eta)

    return configs


def plot_successive_halving_results(configs_dict):
    plt.figure(figsize=(10, 6))
    
    # Plot each configuration's performance
    for id, (config, evaluations) in configs_dict.items():
        budgets = sorted(evaluations.keys())
        scores = [evaluations[b].y for b in budgets]
        plt.plot(budgets, scores, 'o-', alpha=0.3, label=f'Config {id}')
    
    plt.xlabel('Budget')
    plt.ylabel('Score')
    plt.title('Successive Halving Results')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('successive_halving_results.pdf')
    plt.close()


if __name__ == "__main__":
    problem = FCNetBenchmark(name="protein_structures")
    configs_dict = successive_halving(
        problem=problem,
        n_models=40,
        eta=2,
        random_seed=0,
        max_budget_per_model=100,
        min_budget_per_model=10,
    )
    plot_successive_halving_results(configs_dict)
