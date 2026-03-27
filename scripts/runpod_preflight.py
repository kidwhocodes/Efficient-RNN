from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    import matplotlib
    import neurogym
    import torch
    from Mod_Cog import mod_cog_tasks
    from neurogym import Dataset

    from pruning_benchmark.tasks.modcog import ensure_modcog_available, resolve_modcog_callable
    from pruning_benchmark.tasks.neurogym import ModCogTrialDM

    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    print("matplotlib:", matplotlib.__version__)
    print("neurogym:", getattr(neurogym, "__version__", "unknown"))
    print("Dataset:", Dataset)
    print("mod_cog_file:", mod_cog_tasks.__file__)

    ensure_modcog_available(("modcog:ctxdlydm1", "modcog:ctxdlydm1intseq"))
    print("modcog_builders: ok")

    builder_info = resolve_modcog_callable("ctxdlydm1")
    if builder_info is None:
        raise SystemExit("Could not resolve Mod-Cog builder for ctxdlydm1")
    _, builder_fn = builder_info

    dm = ModCogTrialDM(
        builder_fn(),
        T=32,
        B=4,
        device="cpu",
        last_only=False,
        seed=0,
        mask_fixation=True,
    )
    x, y = dm.sample_batch()
    print("sample_batch:", tuple(x.shape), tuple(y.shape))

    required = [
        ROOT / "configs/cloud/modcog_ctxdlydm12_ctrnn_cloud_baselines_5seed.json",
        ROOT / "configs/cloud/modcog_ctxdlydm12_ctrnn_cloud_raw_top4_easy_nonseq_5seed.json",
        ROOT / "configs/cloud/modcog_ctxdlydm12_ctrnn_cloud_raw_top4_hard_seq_5seed.json",
        ROOT / "configs/cloud/modcog_ctxdlydm12_ctrnn_cloud_ft100_top4_easy_nonseq_5seed.json",
        ROOT / "configs/cloud/modcog_ctxdlydm12_ctrnn_cloud_ft100_top4_hard_seq_5seed.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing cloud configs: {missing}")
    print("cloud_configs: ok")

    print("PRECHECK PASSED")


if __name__ == "__main__":
    main()
