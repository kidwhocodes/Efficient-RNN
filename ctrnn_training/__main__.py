import argparse
import numpy as np
import torch

from .experiments import run_prune_experiment

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="l1_unstructured",
                   choices=["random","l1_unstructured","structured_out","structured_in",
                            "global_unstructured","movement","imp","none"])
    p.add_argument("--amount", type=float, default=0.5)
    p.add_argument("--train_steps", type=int, default=600)
    p.add_argument("--ft_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    res = run_prune_experiment(strategy=args.strategy, amount=args.amount,
                               train_steps=args.train_steps, ft_steps=args.ft_steps)
    print(res)

if __name__ == "__main__":
    main()
