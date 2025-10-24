import argparse
from .experiments import run_prune_experiment
import warnings
from .utils import make_run_id, set_global_seed

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*migration_guide.*")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="l1_neuron",
                   choices=[
                            "none","random_neuron","l1_neuron","movement_neuron","imp",
                            "random","l1_unstructured","structured_out","structured_in",
                            "global_unstructured","movement","noise_synapse", "synflow", 
                            "fisher", "activity_neuron", "noise_probe", "noise_combo",
                            "homeostatic_neuron","oja_synapse","variational_dropout",
                            "stdp_synapse","turnover_synapse","energy_neuron"
                            ]
                )
    p.add_argument("--amount", type=float, default=0.5)
    p.add_argument("--train_steps", type=int, default=600)
    p.add_argument("--ft_steps", type=int, default=200)
    p.add_argument("--last_only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["single","sweep"], default="single")
    p.add_argument("--out_csv", default=None)
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
    p.add_argument("--fisher_w", type=float, default=1.0)
    p.add_argument("--noiseprobe_w", type=float, default=1.0)
    p.add_argument("--activity_w", type=float, default=1.0)
    p.add_argument("--reduce", type=str, default="sumabs", choices=["sumabs", "l2"])
    p.add_argument("--debug_scores", action="store_true")
    p.add_argument("--homeo_target", type=float, default=0.05,
               help="Target mean activity for homeostatic pruning")
    p.add_argument("--homeo_mode", type=str, default="relu",
               choices=["relu", "abs", "raw"],
               help="Activity measurement for homeostatic pruning")
    p.add_argument("--homeo_var_weight", type=float, default=0.0,
               help="Variance penalty weight for homeostatic pruning")
    p.add_argument("--vd_eps", type=float, default=1e-8,
               help="Stability epsilon for variational dropout pruning")
    p.add_argument("--stdp_lag", type=int, default=1,
               help="Lag (in timesteps) for STDP synapse scoring")
    p.add_argument("--stdp_mode", type=str, default="causal",
               choices=["causal","absolute","signed"],
               help="Combination mode for STDP synapse scores")
    p.add_argument("--stdp_center", type=int, default=1,
               help="Set to 0 to disable mean-centering in STDP scoring")
    p.add_argument("--turnover_regrow", type=float, default=0.1,
               help="Fraction of pruned synapses to regrow immediately")
    p.add_argument("--turnover_scale", type=float, default=0.1,
               help="Scale factor for new synapses during turnover pruning")
    p.add_argument("--energy_beta", type=float, default=0.5,
               help="Weight on synaptic load in energy-based neuron pruning")
    p.add_argument("--energy_eps", type=float, default=1e-6,
               help="Numerical epsilon for energy-based neuron pruning")
    p.add_argument("--run_id", type=str, default=None,
               help="Optional identifier for this run/sweep (defaults to timestamp).")


    args = p.parse_args()

    set_global_seed(args.seed)
    run_id = args.run_id or make_run_id()
    out_csv = args.out_csv or f"results/{run_id}.csv"

    if args.mode == "sweep":
        from .sweeps import run_sweep
        strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
        amounts    = tuple(float(a) for a in args.amounts.split(",") if a.strip())
        seeds      = tuple(int(s) for s in args.seeds.split(",") if s.strip())
        path = run_sweep(out_csv,
                        strategies=strategies,
                        amounts=amounts,
                        seeds=seeds,
                        task=args.task,
                        run_id=run_id)
        print("Wrote:", path)
        return
    else:
        res = run_prune_experiment(strategy=args.strategy, amount=args.amount,
                               train_steps=args.train_steps, ft_steps=args.ft_steps,
                               last_only=args.last_only, seed=args.seed, task=args.task, no_prune=args.no_prune,
                               ng_kwargs=args.ng_kwargs, ng_T=args.ng_T, ng_B=args.ng_B,hidden_size=args.hidden_size,
                               fisher_w=args.fisher_w, reduce=args.reduce, debug_scores=args.debug_scores,
                               noiseprobe_w=args.noiseprobe_w, activity_w=args.activity_w,
                               homeo_target=args.homeo_target, homeo_mode=args.homeo_mode,
                               homeo_var_weight=args.homeo_var_weight,
                               vd_eps=args.vd_eps,
                               stdp_lag=args.stdp_lag, stdp_mode=args.stdp_mode,
                               stdp_center=bool(args.stdp_center),
                               turnover_regrow=args.turnover_regrow, turnover_scale=args.turnover_scale,
                               energy_beta=args.energy_beta, energy_eps=args.energy_eps,
                               run_id=run_id)
        print(res)

if __name__ == "__main__":
    main()
