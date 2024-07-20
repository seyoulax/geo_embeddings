import os
import random
import torch
import numpy as np

# UTILS THAT ARE USED IN ALL NOTEBOOKS

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

    
class EarlyStoppingR2:
    """Early stops the training if validation r2 doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print, use_checkpoints=True, model_name="best_model.pt",
    ):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_r2_max = -np.Inf
        self.delta = delta
        self.path = os.path.join(path, model_name)
        self.trace_func = trace_func
        self.use_checkpoints = use_checkpoints

    def __call__(self, val_r2, model):

        score = val_r2

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_r2, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_r2, model)
            self.counter = 0

    def save_checkpoint(self, val_r2, model):
        if self.verbose:
            self.trace_func(
                f"Val R2 up from ({self.val_r2_max:.4f} to {val_r2:.4f}).  Saving model ..."
            )
        if self.use_checkpoints:
            torch.save(model.state_dict(), self.path)
        self.val_r2_max = val_r2
        

class train_CFG:
    def __init__(
        self,
    ):
        pass

    def __call__(self, attr, value):
        setattr(self, attr, value)