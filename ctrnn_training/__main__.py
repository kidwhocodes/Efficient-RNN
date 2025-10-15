import argparse
import numpy as np
import torch
from .experiments import run_prune_experiment
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*migration_guide.*")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="l1_neuron",
                   choices=[
                            "none","random_neuron","l1_neuron","movement_neuron","imp",
                            "random","l1_unstructured","structured_out","structured_in",
                            "global_unstructured","movement","noise_synapse", "synflow", 
                            "fisher", "activity_neuron",
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
    p.add_argument("--no_prune", action="store_true", help="Skip pruning (control)")
    p.add_argument("--strategies", type=str,
                default="l1_neuron,movement_neuron,random_neuron,noise_synapse",
                help="Comma-separated pruning strategies for sweep")
    p.add_argument("--amounts", type=str,
                default="0.1,0.3,0.5",
                help="Comma-separated sparsity amounts for sweep")
    p.add_argument("--seeds", type=str,
                default="0,1",
                help="Comma-separated seeds for sweep")
    p.add_argument("--ng_kwargs", type=str, default=None,
               help="JSON dict passed to neurogym.make(), e.g. '{\"sigma\":1.0, \"timing\":{\"delay\":[300,600]}}'")
    p.add_argument("--ng_T", type=int, default=None, help="Override trial length T for NeuroGym tasks")
    p.add_argument("--ng_B", type=int, default=None, help="Override batch size B for NeuroGym tasks")
    p.add_argument("--score_batches", type=int, default=4,
               help="Mini-batches to estimate Fisher/activity scores")
    p.add_argument("--hidden_size", type=int, default=None, help="RNN hidden size")

    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.mode == "sweep":
        from .sweeps import run_sweep
        strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
        amounts    = tuple(float(a) for a in args.amounts.split(",") if a.strip())
        seeds      = tuple(int(s) for s in args.seeds.split(",") if s.strip())
        path = run_sweep(args.out_csv,
                        strategies=strategies,
                        amounts=amounts,
                        seeds=seeds,
                        task=args.task)
        print("Wrote:", path)
        return
    else:
        res = run_prune_experiment(strategy=args.strategy, amount=args.amount,
                               train_steps=args.train_steps, ft_steps=args.ft_steps,
                               last_only=args.last_only, seed=args.seed, task=args.task, no_prune=args.no_prune,
                               ng_kwargs=args.ng_kwargs, ng_T=args.ng_T, ng_B=args.ng_B,hidden_size=args.hidden_size,)
        print(res)

if __name__ == "__main__":
    main()
