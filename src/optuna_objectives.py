import torch

from utils import train_CFG
from training_utils import cross_val_transductive

def objective_transductive(trial, **kwargs):
    
    params = dict(
        n_layers = trial.suggest_int("n_layers", 1, 7),
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        lr_init = trial.suggest_float("lr_init", 1e-4, 5e-3),
        use_feature_propagation = trial.suggest_categorical("use_feature_propagation", [True, False])
    )
    
    if kwargs["model_name"] == "GAT":
        params["n_heads"] = trial.suggest_int("n_heads", 2, 8, 2)
    
    if params["use_feature_propagation"]:
        dataset=kwargs["fp_ds"]
    else:
        dataset=kwargs["no_fp_ds"]
        
    cv_cfg = train_CFG()
    cv_cfg("num_epochs", 500)
    cv_cfg("verbose", 10)
    cv_cfg("scheduler", True),
    cv_cfg("stopper_patience", 300)
    cv_cfg("stopper_delta", 0.001)
    cv_cfg("started_patience", 100)
    
    model_params=dict(
        n_in = dataset.num_features,
        n_out = 1, 
        hidden_dim = params["n_layers"], 
        n_layers = params["n_layers"],
    )
    
    if kwargs["model_name"] == "GAT":
        model_params["head"] = params["n_heads"]

    val_score = cross_val_transductive(
        num_folds=5, 
        dataset=dataset, 
        model_name=kwargs["model_name"], 
        model_params=dict(
            n_in = dataset.num_features,
            n_out = 1, 
            hidden_dim = params["n_layers"], 
            n_layers = params["n_layers"],
        ),
        optimizer_params={"lr" : 0.001771619056705244}, 
        optimizer_name="AdamW",
        cv_cfg=cv_cfg, 
        checkpoints_path=None, 
        eval_test=False,
        device=kwargs["device"],
        use_image=kwargs["use_image"]
    )
    
    return val_score