import sklearn
import torch
import os
import numpy as np
import datetime
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score
import time
import wandb

from tqdm.notebook import tqdm
from torch_geometric.loader import NeighborLoader

from utils import set_seed, EarlyStoppingR2
from models import TransductiveGAT, TransductiveGCN, InductiveGCNwithIMGS, InductiveGATwithIMGS


SEED = 111



#TRANSDUCTIVE LEARNING




def train_one_epoch_transductive(dataset, model, optimizer, loss_fn, mask, use_image=False):

    model.train()
    
    model_input = [dataset] + ([] if not use_image else [dataset.image_embeds])

    out = model(*model_input)

    train_loss = loss_fn(
        out[getattr(dataset, mask)].view(-1), dataset.y[getattr(dataset, mask)]
    )

    train_r2 = sklearn.metrics.r2_score(
        dataset.y[getattr(dataset, mask)].cpu().numpy(),
        out[getattr(dataset, mask)].view(-1).detach().cpu().numpy(),
    )

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    return train_loss.item(), train_r2


def val_one_epoch_transductive(dataset, model, loss_fn, mask, use_image=False):

    model.eval()
    
    with torch.no_grad():
    
        model_input = [dataset] + ([] if not use_image else [dataset.image_embeds])
        
        out = model(*model_input)

        val_loss = loss_fn(
            out[getattr(dataset, mask)].view(-1), dataset.y[getattr(dataset, mask)]
        )
        val_r2 = sklearn.metrics.r2_score(
            dataset.y[getattr(dataset, mask)].cpu().numpy(),
            out[getattr(dataset, mask)].view(-1).detach().cpu().numpy(),
        )

    return val_loss.item(), val_r2

def train_transductive(
    dataset=None, 
    model=None, 
    optimizer=None, 
    loss_fn=None, 
    train_cfg=None, 
    scheduler=None, 
    started_patience=None, 
    earlystopper=None,
    use_image=None
):
    
    stream = tqdm(range(getattr(train_cfg, "num_epochs")), desc="training")
    for epoch in stream:

        train_loss, train_r2 = train_one_epoch_transductive(
            dataset, model, optimizer, loss_fn, train_cfg.train_mask, use_image=use_image
        )
        val_loss, val_r2 = val_one_epoch_transductive(
            dataset, model, loss_fn, train_cfg.val_mask, use_image=use_image
        )

        if epoch > started_patience:

            if earlystopper != None:
                
                earlystopper(val_r2, model)

                if earlystopper.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if scheduler != None:
                scheduler.step(val_r2)

        if getattr(train_cfg, "verbose") and epoch % getattr(train_cfg, "verbose") == 0:
            stream.set_description(f"train r2: {train_r2}, eval r2: {val_r2}")
            
            
def cross_val_transductive(
    num_folds=None, 
    dataset=None, 
    model_name=None, 
    model_params=None, 
    optimizer_params=None, 
    optimizer_name=None,
    cv_cfg=None, 
    checkpoints_path=None, 
    eval_test=False,
    device=torch.device("cpu"),
    use_image=False
):
    
    set_seed(SEED)
        
    test_preds = []
    best_scores = []
    
    for fold in range(num_folds):
        
        cur_fold_checkpoint_path = None
        scheduler_cur_fold = None
        
        assert (os.path.exists(checkpoints_path)), "path for checkoints must exists"
        
        cv_cfg.train_mask = f"train_fold_{fold}"
        cv_cfg.val_mask = f"val_fold_{fold}"
        
        if model_name == "GCN":
            cur_fold_model = TransductiveGCN(**model_params).to(device)
            
        if model_name == "GAT":
            cur_fold_model = TransductiveGAT(**model_params).to(device)
        
        loss_fn_cur_fold = torch.nn.MSELoss()
        
        earlystopper_cur_fold = EarlyStoppingR2(
            patience=cv_cfg.stopper_patience, 
            verbose=False, 
            delta=cv_cfg.stopper_delta, 
            path=checkpoints_path,
            model_name=f"best_model_fold_{fold}.pt",
            trace_func=print,
            use_checkpoints=(True if checkpoints_path is not None else False),  
        )
        
        optimizer_cur_fold = getattr(torch.optim, optimizer_name)(cur_fold_model.parameters(), **optimizer_params) 
        
        if cv_cfg.scheduler is not None:      

            scheduler_cur_fold = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer_cur_fold, 
                factor=0.9, 
                patience=30, 
                threshold=0.01,
                min_lr=1e-5 / 5
            )
               
        
        train_transductive(
            dataset=dataset,
            model=cur_fold_model, 
            optimizer=optimizer_cur_fold, 
            loss_fn=loss_fn_cur_fold, 
            train_cfg=cv_cfg,
            started_patience=cv_cfg.started_patience, 
            scheduler=scheduler_cur_fold,
            earlystopper=earlystopper_cur_fold,
            use_image=use_image
        )
        
        best_scores.append(earlystopper_cur_fold.best_score)
        
        if eval_test:
        
            if model_name == "GCN":
                best_cur_fold_model = TransductiveGCN(**model_params).to(device)

            if model_name == "GAT":
                best_cur_fold_model = TransductiveGAT(**model_params).to(device)
            
            best_cur_fold_model.load_state_dict(torch.load(cur_fold_checkpoint_path))

            best_cur_fold_model.eval()

            with torch.no_grad():

                out = best_cur_fold_model(dataset)
                
                cur_fold_test_preds = out[dataset.test_mask].view(-1).cpu().numpy()
                
                test_preds.append(cur_fold_test_preds)
                
        print(f"fold {fold} finished with best_val_score = {best_scores[-1]}")
        
    if eval_test:
        
        r2_final_test = sklearn.metrics.r2_score(
            dataset.y[dataset.test_mask].cpu().numpy(), 
            np.array(test_preds).mean(axis=0)
        )
        
        print(f"test r2 = {r2_final_test}")
        
    print(f"mean best_val_score = {np.mean(best_scores):4f}")
    return np.mean(best_scores).round(4)








#INDUCTIVE LEARNING

def train_one_epoch_inductive(train_loader, model, optimizer, loss_fn, use_pretrained=False, pretrained_model=None):
        
    model.train()
    
    loss_i = []
    r2_i = []
    
    for batch in train_loader:
        
        if use_pretrained:
            out = model(batch, use_pretrained=True, pretrained_model=pretrained_model)[:batch.batch_size]
        else:
            out = model(batch)[:batch.batch_size]
            
        y = batch.y[:batch.batch_size].squeeze()

        train_loss = loss_fn(out.view(-1), y)
        
        train_r2 = sklearn.metrics.r2_score(
            y.cpu(), 
            out.view(-1).detach().cpu().numpy()
        ) 

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        loss_i.append(train_loss.item() ** 0.5)
        r2_i.append(train_r2)
    
    return np.mean(loss_i), np.mean(r2_i)

@torch.no_grad()
def val_one_epoch_inductive(val_loader, model, loss_fn, use_pretrained=False, pretrained_model=None):
    
    model.eval()
    
    loss_i = []
    r2_i = []
    
    for batch in val_loader:
        
        if use_pretrained:
            out = model(batch, use_pretrained=True, pretrained_model=pretrained_model)[:batch.batch_size]
        else:
            out = model(batch)[:batch.batch_size]
            
        y = batch.y[:batch.batch_size].squeeze()

        val_loss = loss_fn(out.view(-1), y)

        val_r2 = sklearn.metrics.r2_score(
            y.cpu(), 
            out.view(-1).cpu().numpy()
        ) 

        loss_i.append(val_loss.item() ** 0.5)
        r2_i.append(val_r2)

    return np.mean(loss_i), np.mean(r2_i)


def inference_inductive(dataset, model, loader, use_pretrained=False, pretrained_model=None):
    
    assert ((use_pretrained == True and pretrained_model is not None) or (not use_pretrained)), "you must provide timm pretrained model if are`re using pretrained model pipeline"
    
    ys = model.inference(dataset.x, loader, dataset.imgs, use_pretrained=use_pretrained, pretrained_model=pretrained_model).view(-1).cpu().numpy()
    
    train_preds = ys[dataset.train_mask.cpu()]
    val_preds = ys[dataset.val_mask.cpu()]
    test_preds = ys[dataset.test_mask.cpu()]
    
    train_metric = sklearn.metrics.r2_score(
        dataset.y[dataset.train_mask].cpu(),
        train_preds
    )
    
    val_metric = sklearn.metrics.r2_score(
        dataset.y[dataset.val_mask].cpu(),
        val_preds
    )
    
    test_metric = sklearn.metrics.r2_score(
        dataset.y[dataset.test_mask].cpu(),
        test_preds
    )
    
    return train_metric, val_metric, test_metric


def train_inductive(
    train_loader=None, 
    val_loader=None,
    model=None, 
    optimizer=None, 
    loss_fn=None, 
    train_cfg=None, 
    scheduler=None, 
    started_patience=None, 
    earlystopper=None,
    use_pretrained=False, 
    pretrained_model=None
):
    assert ((use_pretrained == True and pretrained_model is not None) or (not use_pretrained)), "you must provide timm pretrained model if are`re using pretrained model pipeline"
    
    stream = tqdm(range(getattr(train_cfg, "num_epochs")), desc="training")
    for epoch in stream:
        
        train_loss, train_r2 = train_one_epoch_inductive(
            train_loader=train_loader, 
            model=model, 
            optimizer=optimizer, 
            loss_fn=loss_fn,  
            use_pretrained=use_pretrained,
            pretrained_model=pretrained_model
        )
        val_loss, val_r2 = val_one_epoch_inductive(
            val_loader=val_loader, 
            model=model, 
            loss_fn=loss_fn, 
            use_pretrained=use_pretrained,
            pretrained_model=pretrained_model
        )

        if epoch > started_patience:

            if earlystopper != None:
                
                earlystopper(val_r2, model)

                if earlystopper.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if scheduler != None:
                scheduler.step(val_r2)

        if getattr(train_cfg, "verbose") and epoch % getattr(train_cfg, "verbose") == 0:
            stream.set_description(f"train r2: {train_r2}, eval r2: {val_r2}")
            
def cross_val_inductive(
    num_folds=None, 
    dataset=None, 
    model_name=None, 
    model_params=None, 
    optimizer_params=None, 
    optimizer_name=None,
    cv_cfg=None, 
    checkpoints_path=None, 
    eval_test=False,
    device=torch.device("cpu"),
    use_pretrained=False,
    pretrained_model=None,
    inference_loader=None
):
    
    assert ((not eval_test) or (eval_test and inference_loader is not None)), "if you want to evaluate on test please pass cross_val_inductive(inference_loader=inference_loader)"
    
    set_seed(SEED)
        
    test_preds = []
    best_scores = []
    
    for fold in range(num_folds):
        
        cur_fold_checkpoint_path = None
        scheduler_cur_fold = None
        
        assert (os.path.exists(checkpoints_path)), "path for checkoints must exists"
        
        cv_cfg.train_mask = f"train_fold_{fold}"
        cv_cfg.val_mask = f"val_fold_{fold}"
        
        #train loader
        train_loader = NeighborLoader(
            dataset,
            input_nodes=getattr(dataset, cv_cfg.train_mask),
            num_neighbors=[-1, -1, -1, -1],
            batch_size=256,
            shuffle=True,
        )

        #val loader
        val_loader = NeighborLoader(
            dataset,
            input_nodes=getattr(dataset, cv_cfg.val_mask),
            num_neighbors=[-1, -1, -1, -1],
            batch_size=256,
            shuffle=False,
        )
        
        if model_name == "GCN":
            cur_fold_model = InductiveGCNwithIMGS(**model_params).to(device)
            
        if model_name == "GAT":
            cur_fold_model = InductiveGATwithIMGS(**model_params).to(device)
        
        loss_fn_cur_fold = torch.nn.MSELoss()
        
        earlystopper_cur_fold = EarlyStoppingR2(
            patience=cv_cfg.stopper_patience, 
            verbose=False, 
            delta=cv_cfg.stopper_delta, 
            path=checkpoints_path,
            model_name=f"best_model_fold_{fold}.pt",
            trace_func=print,
            use_checkpoints=(True if checkpoints_path is not None else False),
        )
        
        optimizer_cur_fold = getattr(torch.optim, optimizer_name)(cur_fold_model.parameters(), **optimizer_params) 
        
        if cv_cfg.scheduler is not None:      

            scheduler_cur_fold = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer_cur_fold, 
                factor=0.9, 
                patience=30, 
                threshold=0.01,
                min_lr=1e-5 / 5
            )
               
        
        train_inductive(
            train_loader=train_loader,
            val_loader=val_loader,
            model=cur_fold_model, 
            optimizer=optimizer_cur_fold, 
            loss_fn=loss_fn_cur_fold, 
            train_cfg=cv_cfg,
            scheduler=scheduler_cur_fold,
            started_patience=cv_cfg.started_patience,
            earlystopper=earlystopper_cur_fold,
            pretrained_model=pretrained_model
        )
        
        best_scores.append(earlystopper_cur_fold.best_score)
                
        print(f"fold {fold} finished with best_val_score = {best_scores[-1]}")
        
    print(f"mean best_val_score = {np.mean(best_scores):4f}")
    return np.mean(best_scores).round(4)










# UNSUPERVISED LEARNINGS GRAPHSAGE

def train_one_epoch_GraphSage(loader, model, optimizer, reduction="mean"):
    
    model.train()
    
    total_loss = 0.0
        
    for i, batch in enumerate(loader):
        
        batch = batch.to(cfg.device)
        h = model(batch.x, batch.edge_index)
        
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * pred.size(0)
        
    return total_loss / cfg.graph_num_nodes

def eval_one_epoch_GraphSage_catboost(model, dataset, return_hidden=False):
    
    model.eval()
    
    with torch.no_grad():
        out = model(dataset.x, dataset.edge_index).cpu()
    
    all_folds_val_r2 = []
    
    for fold in range(5):

        clf = CatBoostRegressor(
            iterations=500,
            task_type="GPU",
            border_count=254,
            random_state=cfg.seed,
            eval_metric="RMSE",
            use_best_model=True
        )

        train_pool = Pool(data=out[getattr(dataset, f"train_fold_{fold}")].numpy(), label=dataset.y[getattr(dataset, f"train_fold_{fold}")].numpy())
        eval_pool = Pool(data=out[getattr(dataset, f"val_fold_{fold}")].numpy(), label=dataset.y[getattr(dataset, f"val_fold_{fold}")].numpy())

        clf.fit(
            train_pool, 
            verbose=0,
            eval_set=[eval_pool],
        )
        
        val_preds = clf.predict(eval_pool)
        
        val_score = r2_score(eval_pool.get_label(), val_preds.reshape(-1))
        
        all_folds_val_r2.append(val_score)
        
    all_folds_val_r2 = np.mean(all_folds_val_r2)

    if return_hidden: 
        return all_folds_val_r2, out
    else: 
        return all_folds_val_r2
    

def train_embedder(
    model=None, 
    loader=None, 
    optimizer=None, 
    train_cfg=None,
    full_dataset=None,
    earlystopper_loss=None,
    earlystopper_r2=None

):

    stream = tqdm(range(train_cfg.num_epochs))
    
    val_r2 = 0

    for epoch in stream:

        #training
        loss = train_one_epoch(         
            loader=loader,
            model=model,
            optimizer=optimizer, 
            reduction="mean"
        )
        
        if epoch % 25 == 0:
            
            val_r2, h = eval_one_epoch_catboost(
                model,
                full_dataset,
                return_hidden=True
            )
            
            plot_embeddings(
                train_dataset=full_dataset,
                out=h,
                epoch=epoch
            )

        stream.set_description(f"train loss: {round(loss, 5)}, val r2: {val_r2}")

        #saving model if its best till now
        earlystopper_loss(loss, model) 
        earlystopper_r2(val_r2, model)
        
        # wandb.log(
        #     {
        #         "loss" : round(loss, 5), 
        #         "val_r2" : round(val_r2, 3)
        #     }
        # )

        
        
        
        
        
        
        
        
        
# UNSUPERVISED LEARNINGS INFOMAX

def train_one_epoch_infomax(train_loader, model, epoch, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch:02d}'):
        
        batch = batch.to(device)
        
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(batch.x, batch.edge_index,
                                      batch.batch_size)
        
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pos_z.size(0)
        total_examples += pos_z.size(0)

    return total_loss / total_examples


@torch.no_grad()
def val_one_epoch_infomax(test_loader, model, data, epoch, device):
    
    model.eval()
    zs = []
    for batch in test_loader:
        batch = batch.to(device)
        pos_z, _, _ = model(batch.x, batch.edge_index, batch.batch_size)
        zs.append(pos_z.cpu())
    out = torch.cat(zs, dim=0).to(device)
    
    all_folds_val_r2 = []
    all_folds_train_r2 = []
    
    for fold in tqdm(range(5), desc='Cross val'):

        clf = CatBoostRegressor(
            iterations=500,
            task_type="GPU",
            border_count=128
        )   
        
        train_data = torch.cat((out[getattr(data, f"train_fold_{fold}")], data.x[getattr(data, f"train_fold_{fold}")]), -1)
        val_data = torch.cat((out[getattr(data, f"val_fold_{fold}")], data.x[getattr(data, f"val_fold_{fold}")]), -1) 
        
        train_pool = Pool(data=train_data.cpu().numpy(), label=data.y[getattr(data, f"train_fold_{fold}")].cpu().numpy())
        eval_pool = Pool(data=val_data.cpu().numpy(), label=data.y[getattr(data, f"val_fold_{fold}")].cpu().numpy())

        clf.fit(
            train_pool, 
            eval_set=[eval_pool],
            use_best_model=True,
            logging_level="Silent"
        )
        
        train_preds = clf.predict(train_pool)
        val_preds = clf.predict(eval_pool)
        
        train_r2 = r2_score(train_pool.get_label(), train_preds)
        val_r2 = r2_score(eval_pool.get_label(), val_preds)
        
        all_folds_train_r2.append(train_r2)
        all_folds_val_r2.append(val_r2)
        
    all_folds_train_r2 = np.mean(all_folds_train_r2) 
    all_folds_val_r2 = np.mean(all_folds_val_r2)

    return (all_folds_train_r2, all_folds_val_r2)
    

def train_infomax(dataset, train_loader, test_loader, model, epochs, optimizer, model_save_path, device, use_wndb=False):
    min_loss = np.inf
    max_r2 = -np.inf
    
    for epoch in range(1, epochs+1):
        start = time.time()

        train_loss = train_one_epoch_infomax(train_loader, model, epoch, optimizer, device)

        time_left = (time.time() - start) * (epochs - epoch)
        time_left = str(datetime.timedelta(seconds=time_left))

        print(f'Train loss: {np.round(train_loss, 5):.4f}, ',
              f'Time left: {time_left}')

        if epoch % 25 != 0:
            if train_loss < min_loss:
                print('Save new model')
                min_loss = train_loss
                torch.save(model.state_dict(), f'{model_save_path}/best_loss_models/{epoch}_{np.round(train_loss, 5)}.pth')
            if use_wndb: wandb.log({"Train loss": train_loss})
        else:
            all_folds_train_r2, all_folds_val_r2 = val_one_epoch_infomax(test_loader, model, dataset, epoch, device)
            print(f'Train R2: {np.round(all_folds_train_r2, 5):.4f}, ',
                  f'Val R2: {np.round(all_folds_val_r2, 5):.4f}, ')
            if all_folds_val_r2 > max_r2:
                print('Save new model')
                max_r2 = all_folds_val_r2
                torch.save(model.state_dict(), f'{model_save_path}/best_r2_models/{epoch}_{np.round(train_loss, 5)}_{np.round(all_folds_val_r2, 5)}.pth')
            if use_wndb: wandb.log({"Train R2": all_folds_train_r2, "Val R2": all_folds_val_r2})
            
              
def test_infomax(dataset, model):
    all_pos_z = model(dataset.x, dataset.edge_index, len(dataset.x))[0]
    all_folds_val_r2, all_folds_train_r2 =[], []

    for fold in range(5):
        clf = CatBoostRegressor(
                iterations=500,
                task_type="GPU",
                border_count=128
            )
        train_data = torch.cat((all_pos_z[getattr(dataset, f"train_fold_{fold}")], dataset.x[getattr(dataset, f"train_fold_{fold}")]), -1)
        val_data = torch.cat((all_pos_z[getattr(dataset, f"val_fold_{fold}")], dataset.x[getattr(dataset, f"val_fold_{fold}")]), -1)

        train_pool = Pool(data=train_data.cpu().detach().numpy(), label=dataset.y[getattr(dataset, f"train_fold_{fold}")].cpu().detach().numpy())
        eval_pool = Pool(data=val_data.cpu().detach().numpy(), label=dataset.y[getattr(dataset, f"val_fold_{fold}")].cpu().detach().numpy())

        clf.fit(
                train_pool, 
                eval_set=[eval_pool],
                use_best_model=True,
                verbose=10
            )

        val_preds = clf.predict(eval_pool)
        train_preds = clf.predict(train_pool)

        val_r2 = r2_score(eval_pool.get_label(), val_preds)
        train_r2 = r2_score(train_pool.get_label(), train_preds)

        all_folds_val_r2.append(val_r2)
        all_folds_train_r2.append(train_r2)


    all_folds_val_r2 = np.mean(all_folds_val_r2)
    all_folds_train_r2 = np.mean(all_folds_train_r2)
    
    return all_folds_train_r2, all_folds_val_r2