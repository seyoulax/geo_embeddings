import pandas as pd, numpy as np, matplotlib.pyplot as plt, time
from tqdm.notebook import tqdm

import os
import sys
import timm
import torch
import argparse 
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import r2_score
import torchvision.transforms as transforms

sys.path.append("../src")

#here you can find utils that are used in each notebook
from utils import set_seed, EarlyStoppingR2, train_CFG
from datasets import RasterDataset, TimmRasterDataset
from models import RasterCNN

def get_images_features(dataset):
    a = []
    data = {}
    for i in dataset.imgs:
        a.append(i)
    m1 = np.zeros_like(dataset.imgs[199][0]) 
    m1[26:, 26:] = 1      

    m2 = np.zeros_like(dataset.imgs[199][0]) 
    m2[26:, :27] = 1    

    m3 = np.zeros_like(dataset.imgs[199][0])  
    m3[:27:, :27] = 1   

    m4 = np.zeros_like(dataset.imgs[199][0]) #1
    m4[:27:, 26:] = 1 
    
    
    for i in range(len(a)):
        for j in range(12):
            if i not in data:
                data[i] = [(dataset.imgs[i][j] * m1 > 0).sum().item()]
            else:
                data[i] += [(dataset.imgs[i][j] * m1 > 0).sum().item()]
            data[i] += [(dataset.imgs[i][j] * m2 > 0).sum().item()]
            data[i] += [(dataset.imgs[i][j] * m3 > 0).sum().item()]
            data[i] += [(dataset.imgs[i][j] * m4 > 0).sum().item()]
    
    for i in range(len(a)):
        for j in range(12):
            if i not in data:
                data[i] = [(dataset.imgs[i][j] > 0).sum().item()]
            else:
                data[i] += [(dataset.imgs[i][j] > 0).sum().item()]
    
    df = pd.DataFrame.from_dict(data, orient="index")
    res = df.apply(pd.Series.explode)
    
    res.columns = ['num', 'm10','m20','m3','m40','m11', 'm21', 'm31', 'm41',
                   'm12','m22','m32','m42','m13','m23','m33','m43','m14','m24',
                   'm34','m44','m15','m25','m35', 'm16','m26','m36','m46','m17',
                   'm27','m37','m47','m18','m28','m38','m48','m19','m29','m39',
                   'm49','m110','m210','m310','m410','m111','m211','m311','m411',
                   'summ0', 'summ1', 'summ2','summ3','summ4','summ5','summ6',
                   'summ7','summ8','summ9','summ10','summ11']
    return torch.FloatTensor(res.values) / (52 * 52)

class cfg:
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = cfg()

parser=argparse.ArgumentParser(description="pipeline_args")

#features args
parser.add_argument("--pipeline", type=str, default="efficient_all_channels")
parser.add_argument("--region", type=int, default=777)
parser.add_argument(
    "--path_to_graph", 
    type=str, 
    default=f"/home/jupyter/datasphere/s3/s3sirius/sirius_2024_participants/twwist/graph_with_cv_full_and_images/"
)
parser.add_argument("--path_to_save", type=str, default="../data/image_embeddings/")
parser.add_argument("--path_to_chkps", type=str, default="../chkps/image_embeddings/")

args=parser.parse_args()

path_to_graph = os.path.join(args.path_to_graph, f"images_graph_{args.region}.pickle")
path_to_save = os.path.join(args.path_to_save, f"image_embeddings_{args.pipeline}_{args.region}.pickle")

#loading with torch
print("Loading graph ")
graph = torch.load(path_to_graph)
xs = graph.imgs[(graph.y != -1)]
ys = graph.y[(graph.y != -1)].tolist()

if not os.path.exists(f"{args.path_to_save}/image_features_{args.region}.pickle"):
    
    print("Starting getting graph features")
    image_features = get_images_features(graph)
    
    torch.save(image_features, f"{args.path_to_save}/image_features_{args.region}.pickle")

if args.pipeline == "basic":
    full_dataset = RasterDataset(xs, ys)
else:
    
    if args.pipeline == "efficient_delete_channels":
        # throwing out some trash channels
        graph.imgs = graph.imgs[:, [0, 3, 4, 5, 6, 7, 8, 9, 10]]
    
    full_dataset = TimmRasterDataset(graph.imgs, graph.y, transforms)



if "efficient" in args.pipeline:
    
    print("Loading timm models")
    feature_extractor = timm.create_model('efficientnet_b0.ra_in1k',
        pretrained=True,
        features_only=True,
        in_chans=1,
        out_indices=[2]
    ).to(cfg.device)

    model = timm.create_model('efficientnet_b0.ra_in1k',
            pretrained=True,
            features_only=False,
            in_chans=12
    ).to(cfg.device)

    dataloader_all_images = torch.utils.data.DataLoader(
        full_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    use_feature_exctractor = (False if args.pipeline == "efficient_all_channels" else True)

    model.eval()

    t = nn.AvgPool2d(kernel_size=(28, 28))

    #resulted image embeddins
    image_embeds = []

    print("Starting getting embeddings")
    with torch.no_grad():
        for images in tqdm(dataloader_all_images):
            if use_feature_exctractor:
                images = images.flatten(end_dim=1).unsqueeze(1)
                output = feature_extractor(images.to(cfg.device))[0]
                output = t(output).flatten(start_dim=1)
                output = torch.cat([x for x in output], dim=0)
            else:
                output = model.forward_features(images.to(cfg.device))
                output = model.forward_head(output, pre_logits=True)
            image_embeds.append(output)
            
    image_embeds = torch.cat(image_embeds, dim=0).view(len(image_embeds), -1)
    torch.save(image_embeds.cpu(), path_to_save)

else:
    
    train_set, val_set = torch.utils.data.random_split(full_dataset, [0.9, 0.1])

    batch_size = 64

    dataloader_train = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
    )
    
    print("Starting training basic CNN")
    model = RasterCNN(12).to(cfg.device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=7e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=4e-5, 
        max_lr=2e-4, 
        cycle_momentum = False, 
        step_size_up=20,
    )
    
    set_seed(111)

    best_metric = float('inf')
    best_epoch = -1

    for epoch in range(cfg.num_epochs):

        loss_i_train = []
        r2_i_train = []

        model.train()

        for images, targets in tqdm(dataloader_train, desc='Training'):

            images = images.to(cfg.device)
            targets = targets.to(cfg.device)

            optimizer.zero_grad()

            outputs = model(images.float()).squeeze()
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()

            loss_i_train.append(loss.item() ** 0.5)
            r2_i_train.append(
                r2_score(targets.cpu(), outputs.detach().cpu())
            )

        print(f"Epoch: {epoch}, Train rmse: {np.mean(loss_i_train)} r2: {np.mean(r2_i_train)}")

        loss_i_val = []
        r2_i_val = []

        model.eval()

        with torch.no_grad():
            for images, targets in dataloader_val:

                images = images.to(cfg.device)
                targets = targets.to(cfg.device)

                outputs = model(images.float()).squeeze()

                loss_i_val.append(loss.item() ** 0.5)
                r2_i_val.append(
                    r2_score(targets.cpu(), outputs.detach().cpu())
                )

        print(f"Validation rmse: [{np.mean(loss_i_val)}], r2: [{np.mean(r2_i_val)}]")

        if np.mean(loss_i_val) - best_metric < -1e6 / 2:
            best_metric = np.mean(loss_i_val)
            best_epoch = epoch

            if epoch != 0:

                torch.save(model, f'{args.path_to_chkps}/weights_{epoch}_{round(best_metric, 5)}.pt')


    print('Finished Training')
    
    all_images_dataset = RasterDataset(graph.imgs, graph.y)

    dataloader_all_images = torch.utils.data.DataLoader(
        all_images_dataset, 
        batch_size=1,
        shuffle=False
    )

    hidden_states = []

    best_model = torch.load(f'{args.path_to_chkps}/weights_{best_epoch}_{round(best_metric, 5)}.pt')
    
    print("Getting embeddings")
    best_model.eval()
    with torch.no_grad():
        for image, _ in tqdm(dataloader_all_images):
            _, h = best_model(image.to(cfg.device).float(), return_hidden=True)
            hidden_states.append(h.squeeze(0))
            
    hidden_states = torch.cat(hidden_states, dim=0).view(-1, hidden_states[0].shape[0])
    hidden_states = hidden_states.to(torch.device('cpu'))
    torch.save(graph, path_to_save)
    print("Embeddings saved succesfully")