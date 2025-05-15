import argparse
from pathlib import Path
import os
import time
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt

import neps

from src.fcnet_benchmark import FCNetBenchmark


def run_neps(
    benchmark: FCNetBenchmark,
    fidelity_name: str = "epoch",
    optimizer: str = "hyperband"
):
    # TODO: Create Pipeline Space
    pipeline_space = {
        "lr": neps.FloatParameter(lower=1e-6, upper=1e-1, log=True),
        "batch_size": neps.IntegerParameter(lower=8, upper=128, log=True),
        "dropout": neps.FloatParameter(lower=0.0, upper=0.5),
        "n_units_1": neps.IntegerParameter(lower=16, upper=512, log=True),
        "n_units_2": neps.IntegerParameter(lower=16, upper=512, log=True),
        "n_units_3": neps.IntegerParameter(lower=16, upper=512, log=True),
        "activation": neps.CategoricalParameter(choices=["relu", "tanh"]),
        "init": neps.CategoricalParameter(choices=["uniform", "normal"]),
        fidelity_name: neps.IntegerParameter(lower=10, upper=100, is_fidelity=True)
    }

    def run_pipeline(**config):
        budget = config.pop(fidelity_name)  # Check if this would work
        evaluation = benchmark.objective_function(config, budget)

        # time.sleep(evaluation.runtime / 10)  # NOTE: uncomment only for parallel runs

        return {
            "objective_to_minimize": evaluation.y,
            "loss": -evaluation.y,
            "cost": evaluation.runtime,
            "info_dict": {
                "pid": os.getpid(),
            }
        }

    # TODO: Set the NePS pipeline to store optimizer specific results
    # You can modify the root directory to store the results in a specific folder
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory=f"./neps/{optimizer}",
        max_cost_total=3600,  # 1 hour timeout
        max_cost_per_run=600,  # 10 minutes per run
        optimizer=optimizer,
        seed=42
    )


def plot_neps(
    plt,
    root_directory="./neps/",
    task_name: str = "hyperband",
    log_x: bool = False,
    log_y: bool = True,
):
    # load the results from the task
    df = pd.read_csv(f"{root_directory}/{task_name}/summary/full.csv", index_col=0)

    # sort by time sample finished evaluation, this order is accurate even for parallel evaluations
    df = df.sort_values(by=["time_sampled", "time_end"])

    # plot the loss over time
    plt.plot(
        df["cost"].cumsum().values,
        np.minimum.accumulate(df["objective_to_minimize"]).values,
        label=task_name
    )
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.xlabel("(Simulation) Runtime Cost [in s]")
    plt.ylabel("Loss")
    plt.legend()

    return plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed for the run(s)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all", "successive_halving", "hyperband"],
        help="The optimizer to use for the search",
    )
    parser.add_argument(
        "--root_directory",
        type=str,
        default="./neps/",
        help="The directory for NePS output files",
    )
    parser.add_argument(
        "--plot_directory",
        type=str,
        default="./outputs/",
        help="The directory to save plots",
    )
    parser.add_argument(
        "--only_plot",
        action="store_true",
        help="To skip NePS run and only trigger plotting",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="To show logs in stdout at INFO level",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Run NePS
    if not args.only_plot:
        if args.run == "all" or args.run == "successive_halving":
            np.random.seed(args.seed)
            run_neps(FCNetBenchmark(name="protein_structures"), optimizer="successive_halving")

        if args.run == "all" or args.run == "hyperband":
            np.random.seed(args.seed)
            run_neps(FCNetBenchmark(name="protein_structures"), optimizer="hyperband")

    # Plotting
    plt.clf()
    if args.run == "all" or args.run == "successive_halving":
        plot_neps(
            plt,
            root_directory=args.root_directory,
            task_name=f"successive_halving",
            log_y=False,
        )
    if args.run == "all" or args.run == "hyperband":
        plot_neps(
            plt,
            root_directory=args.root_directory,
            task_name=f"hyperband",
            log_y=False,
        )
    plt.savefig(Path(args.plot_directory) / "neps_results.pdf", dpi=300)
