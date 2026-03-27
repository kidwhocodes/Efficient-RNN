try:
    import sb3_contrib
    import stable_baselines3

    _SB3_INSTALLED = True
except ImportError:
    _SB3_INSTALLED = False


from .envs.registration import make, register
from .utils import info, spaces
from .utils.data import Dataset
from .core import TrialEnv

__version__ = "2.2.0"

from .core import TrialEnv

