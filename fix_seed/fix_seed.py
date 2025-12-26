import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import np
import torch

def set_deterministic(seed: int = 0, verbose: bool = True):
    random.seed(seed)
    np.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if verbose:
        print("[Deterministic] seed=", seed,
              "| cudnn.deterministic=True, benchmark=False",
              "| use_deterministic_algorithms=True",
              "| CUBLAS_WORKSPACE_CONFIG=", os.environ["CUBLAS_WORKSPACE_CONFIG"])
