import argparse
import numpy as np
import torch
from .experiments import run_prune_experiment

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="l1_neuron",
                   choices=[
                            "none","random_neuron","l1_neuron","movement_neuron","imp",
                            "random","l1_unstructured","structured_out","structured_in",
                            "global_unstructured","movement","noise_synapse"
                            ]
                )
    p.add_argument("--amount", type=float, default=0.5)
    p.add_argument("--train_steps", type=int, default=600)
    p.add_argument("--ft_steps", type=int, default=200)
    p.add_argument("--last_only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["single","sweep"], default="single")
    p.add_argument("--out_csv", default="results/sweep_toy.csv")
    p.add_argument("--task", default="synthetic", help="synthetic (built-in) or ng:<NeuroGymTaskName>")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.mode == "sweep":
        from .sweeps import run_sweep
        path = run_sweep(args.out_csv, task=args.task)
        print("Wrote:", path)
    else:
        res = run_prune_experiment(strategy=args.strategy, amount=args.amount,
                               train_steps=args.train_steps, ft_steps=args.ft_steps,
                               last_only=args.last_only, seed=args.seed, task=args.task)
        print(res)

if __name__ == "__main__":
    main()
