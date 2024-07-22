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
    

# EarlyStopping class for ContrastiveLoss
class EarlyStoppingContrastiveLoss:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, loss, model):

        score = loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, best_score, model):
        if self.verbose:
            self.trace_func(f'Contrastive Loss decreased from ({self.loss_min:.4f} to {self.best_score:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.loss_min = best_score
        
        
# EarlyStopping class for R2 metric
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
        
    
    
def plot_embeddings(train_dataset=None, out=None, epoch=None, checkpoints_path=None):

    mapper_umap = UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=cfg.seed)

    mapped_2d_umap = mapper_umap.fit_transform(out.cpu())
    
    f, axs = plt.subplots(2, 2, figsize=(20, 16))

    axs = list(axs[0]) + list(axs[1])
    out[out <= 0] = 1

    #1
    # plotting target dist with train
    sns.scatterplot(
        x = mapped_2d_umap[train_dataset.y.cpu() == -1][:, 0],
        y = mapped_2d_umap[train_dataset.y.cpu() == -1][:, 1],
        hue=train_dataset.y.cpu()[train_dataset.y.cpu() == -1],
        palette={-1:'gray'},
        ax=axs[0],
    );
    sns.scatterplot(
        x = mapped_2d_umap[:, 0][train_dataset.train_mask.cpu()],
        y = mapped_2d_umap[:, 1][train_dataset.train_mask.cpu()],
        palette = 'winter_r',
        hue=np.log10(train_dataset.y.cpu()[train_dataset.train_mask.cpu()]),
        ax=axs[0],
    );



    #2
    # plotting target dist with val

    sns.scatterplot(
        x = mapped_2d_umap[train_dataset.y.cpu() == -1][:, 0],
        y = mapped_2d_umap[train_dataset.y.cpu() == -1][:, 1],
        hue=train_dataset.y.cpu()[train_dataset.y.cpu() == -1],
        palette={-1:'gray'},
        ax=axs[2],
    );
    sns.scatterplot(
        x = mapped_2d_umap[train_dataset.val_mask.cpu()][:, 0],
        y = mapped_2d_umap[train_dataset.val_mask.cpu()][:, 1],
        palette = 'winter_r',
        hue=np.log10(train_dataset.y.cpu()[train_dataset.val_mask.cpu()]),
        ax=axs[2],
    );



    #3
    # plotting pib3 dist with train
    sns.scatterplot(
        x = mapped_2d_umap[:, 0],
        y = mapped_2d_umap[:, 1],
        palette = 'winter_r',
        hue=np.log10(train_dataset.x[:, 97].cpu() + 1),
        ax=axs[1],
    );





    #4
    # plotting pib3 dist with val
    sns.scatterplot(
        x = mapped_2d_umap[:, 0],
        y = mapped_2d_umap[:, 1],
        palette = 'winter_r',
        hue=np.log10(train_dataset.x[:, 97].cpu() + 1),
        ax=axs[3],
    );


    for ax in axs:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.get_legend().remove()

    axs[0].set_title("target")
    axs[1].set_title("pib3")


    f.tight_layout()
    
    f.figure.savefig(f'{checkpoints_path}/embeddings_visual_{epoch}.jpeg', dpi=100)
    