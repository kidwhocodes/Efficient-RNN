import argparse
import warnings
from typing import Tuple

from .analysis.summary import summarize_csv
from .analysis.plots import plot_metrics
from .experiments import run_prune_experiment, run_suite_from_config, run_sweep, train_baselines
from .pruning import available_pruning_strategies
from .utils import make_run_id, set_global_seed

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*migration_guide.*")


def _parse_comma_floats(src: str) -> Tuple[float, ...]:
    return tuple(float(item) for item in src.split(",") if item.strip())


def _parse_comma_ints(src: str) -> Tuple[int, ...]:
    return tuple(int(item) for item in src.split(",") if item.strip())


def _parse_comma_strs(src: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in src.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser()
    pruning_choices = ["none"] + sorted(available_pruning_strategies().keys())
    parser.add_argument("--strategy", default="l1_neuron", choices=pruning_choices)
    parser.add_argument("--amount", type=float, default=0.5)
    parser.add_argument("--train_steps", type=int, default=600)
    parser.add_argument("--ft_steps", type=int, default=200)

    last_only_group = parser.add_mutually_exclusive_group()
    last_only_group.add_argument(
        "--last_only",
        dest="last_only",
        action="store_true",
        help="Evaluate/train using only the final timestep (default).",
    )
    last_only_group.add_argument(
        "--full_sequence",
        dest="last_only",
        action="store_false",
        help="Evaluate loss/accuracy across the entire sequence.",
    )
    parser.set_defaults(last_only=True)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--movement_batches",
        type=int,
        default=20,
        help="Retained for backwards compatibility; unused by current strategies.",
    )
    parser.add_argument("--mode", choices=["single", "sweep", "suite", "summary", "plot", "baseline"], default="single")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument(
        "--task",
        default="synthetic",
        help="synthetic (built-in) or ng:<NeuroGymTaskName>",
    )
    parser.add_argument("--no_prune", action="store_true", help="Skip pruning (control)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a suite configuration file (required for mode=suite)",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Input CSV for summary/plot modes",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default=None,
        help="Optional output path (.csv or .json) for summary mode",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default="strategy,amount",
        help="Comma-separated columns to group by when summarising",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="post_acc,post_loss",
        help="Comma-separated metric columns to average",
    )
    parser.add_argument(
        "--plot_out",
        type=str,
        default="plots",
        help="Directory to store generated plots (mode=plot)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="l1_neuron,random_neuron,noise_prune",
        help="Comma-separated pruning strategies for sweep mode",
    )
    parser.add_argument(
        "--amounts",
        type=str,
        default="0.1,0.3,0.5",
        help="Comma-separated sparsity amounts for sweep mode",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1",
        help="Comma-separated seeds for sweep mode",
    )
    parser.add_argument(
        "--ng_kwargs",
        type=str,
        default=None,
        help='JSON dict passed to neurogym.make(), e.g. "{\\"sigma\\": 1.0}"',
    )
    parser.add_argument("--ng_T", type=int, default=None, help="Override trial length T for NeuroGym tasks")
    parser.add_argument("--ng_B", type=int, default=None, help="Override batch size B for NeuroGym tasks")
    parser.add_argument("--hidden_size", type=int, default=None, help="Override RNN hidden size")
    parser.add_argument("--noise_sigma", type=float, default=1.0, help="Noise prune sigma hyperparameter")
    parser.add_argument("--noise_eps", type=float, default=0.3, help="Noise prune epsilon hyperparameter")
    parser.add_argument(
        "--noise_leak_shift",
        type=float,
        default=0.0,
        help="Shift applied to the CT operator diagonal during noise pruning",
    )
    parser.add_argument(
        "--noise_matched_diagonal",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set to 0 to disable matched diagonal in noise pruning",
    )
    parser.add_argument("--noise_rng_seed", type=int, default=None, help="Optional RNG seed for noise pruning")
    parser.add_argument("--eval_seed", type=int, default=None, help="Seed for deterministic evaluation sampling")
    parser.add_argument(
        "--eval_sample_batches",
        type=int,
        default=0,
        help="Number of fixed batches to reuse during evaluation",
    )
    parser.add_argument("--eval_steps_pre0", type=int, default=50)
    parser.add_argument("--eval_steps_pre", type=int, default=100)
    parser.add_argument("--eval_steps_post0", type=int, default=100)
    parser.add_argument("--eval_steps_post", type=int, default=100)
    parser.add_argument("--skip_training", action="store_true", help="Skip the baseline training phase")
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="Optional path to save the trained model before pruning",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Optional path to load a pre-trained model before running the experiment",
    )
    parser.add_argument(
        "--baseline_config",
        type=str,
        default=None,
        help="Path to a baseline-training configuration file (mode=baseline)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional identifier for this run/sweep (defaults to timestamp)",
    )

    args = parser.parse_args()

    set_global_seed(args.seed)
    run_id = args.run_id or make_run_id()
    out_csv = args.out_csv or f"results/{run_id}.csv"

    if args.mode == "baseline":
        if args.baseline_config is None:
            raise ValueError("--baseline_config is required when mode=baseline")
        checkpoints = train_baselines(args.baseline_config, overwrite=args.skip_training)
        for path in checkpoints:
            print(path)
        return

    if args.mode == "sweep":
        strategies = _parse_comma_strs(args.strategies)
        amounts = _parse_comma_floats(args.amounts)
        seeds = _parse_comma_ints(args.seeds)
        path = run_sweep(
            out_csv,
            strategies=strategies,
            amounts=amounts,
            seeds=seeds,
            train_steps=args.train_steps,
            ft_steps=args.ft_steps,
            last_only=args.last_only,
            device=args.device,
            movement_batches=args.movement_batches,
            task=args.task,
            noise_kwargs={
                "noise_sigma": args.noise_sigma,
                "noise_eps": args.noise_eps,
                "noise_leak_shift": args.noise_leak_shift,
                "noise_matched_diagonal": bool(args.noise_matched_diagonal),
                "noise_rng_seed": args.noise_rng_seed,
            },
            run_id=run_id,
        )
        print("Wrote:", path)
        return

    if args.mode == "suite":
        if args.config is None:
            raise ValueError("--config is required when mode=suite")
        path = run_suite_from_config(args.config)
        print("Suite wrote:", path)
        return

    if args.mode == "summary":
        if args.input_csv is None:
            raise ValueError("--input_csv is required when mode=summary")
        group_fields = _parse_comma_strs(args.group_by) or ("strategy", "amount")
        metric_fields = _parse_comma_strs(args.metrics) or ("post_acc", "post_loss")
        summaries = summarize_csv(
            args.input_csv,
            group_fields=group_fields,
            metrics=metric_fields,
            output_path=args.summary_out,
        )
        for row in summaries:
            print(row)
        return

    if args.mode == "plot":
        if args.input_csv is None:
            raise ValueError("--input_csv is required when mode=plot")
        group_fields = _parse_comma_strs(args.group_by) or ("strategy", "amount")
        if len(group_fields) < 2:
            raise ValueError("--group_by must include at least a group and an amount column")
        group_field, amount_field, *rest = group_fields
        metric_fields = _parse_comma_strs(args.metrics) or ("post_acc", "post_loss")
        plot_metrics(
            args.input_csv,
            metrics=metric_fields,
            group_field=group_field,
            amount_field=amount_field,
            output_dir=args.plot_out,
        )
        print(f"Plots written to {args.plot_out}")
        return

    res = run_prune_experiment(
        strategy=args.strategy,
        amount=args.amount,
        train_steps=args.train_steps,
        ft_steps=args.ft_steps,
        last_only=args.last_only,
        seed=args.seed,
        device=args.device,
        movement_batches=args.movement_batches,
        task=args.task,
        no_prune=args.no_prune,
        ng_kwargs=args.ng_kwargs,
        ng_T=args.ng_T,
        ng_B=args.ng_B,
        hidden_size=args.hidden_size,
        noise_sigma=args.noise_sigma,
        noise_eps=args.noise_eps,
        noise_leak_shift=args.noise_leak_shift,
        noise_matched_diagonal=bool(args.noise_matched_diagonal),
        noise_rng_seed=args.noise_rng_seed,
        eval_seed=args.eval_seed,
        eval_sample_batches=args.eval_sample_batches,
        eval_steps_pre0=args.eval_steps_pre0,
        eval_steps_pre=args.eval_steps_pre,
        eval_steps_post0=args.eval_steps_post0,
        eval_steps_post=args.eval_steps_post,
        skip_training=args.skip_training,
        save_model_path=args.save_model_path,
        load_model_path=args.load_model_path,
        run_id=run_id,
    )
    print(res)


if __name__ == "__main__":
    main()
