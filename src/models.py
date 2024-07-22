from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GATConv

import torch.nn.functional as F
import torch.nn as nn
import torch

# Graph CONV BLOCKS

class GCNBlock(nn.Module):

    def __init__(self, h_in, h_out):

        super(GCNBlock, self).__init__()

        self.skip = nn.Linear(h_in, h_out)
        self.conv = GCNConv(h_in, h_out, improved=True)
        self.act = nn.ELU(alpha=0.1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index) + self.skip(x)
        return self.act(self.dropout(out))
    
    
class GATBlock(nn.Module):

    def __init__(self, h_in, h_out, n_heads):

        super(GATBlock, self).__init__()

        self.skip = nn.Linear(h_in, h_out * n_heads)
        self.conv = GATConv(h_in, h_out, heads=n_heads, dropout=0.0)
        self.act = nn.ELU(alpha=0.1)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index) + self.skip(x)
        return self.act(self.dropout(out))

    
# TRANSDACIVE MODELS

#GCN
class TransductiveGCN(nn.Module):

    def __init__(
        self,
        n_in=128,
        n_out=1,
        hidden_dim=128,
        n_layers=1,
        use_image=False,
        image_size=None
    ):
        assert (
            not use_image and image_size == None or use_image and image_size != None
        ), "image_size must be None if you`re not using image or not None if you`re using image"
        
        
        super(TransductiveGCN, self).__init__()
        
        self.use_image = use_image
        
        hidden_dims = []
        factor = 1
        for i in range(1, n_layers + 1):
            hidden_dims.append(int(hidden_dim * factor))
            if i <= n_layers // 2: factor *= 2
            else: factor /= 2
            

        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
        )
        
        if use_image:
            hidden_dims[0] += image_size

        self.conv_layers = nn.ModuleList(
            GCNBlock(h_in, h_out)
            for h_in, h_out in zip(
                hidden_dims, hidden_dims[1:]
            )
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dims[-1], n_out),
        )

    def forward(self, data, images=None):

        x, edge_index = data.x, data.edge_index

        x = self.encoder(x)
        
        if self.use_image:
            x = torch.cat([x, images], dim=-1)

        for layer in self.conv_layers:
            x = layer(x, edge_index)

        return self.decoder(x)

#GAT
class TransductiveGAT(nn.Module):
    def __init__(
            self, 
            n_in=128, 
            n_out=128, 
            hidden_dim=128, 
            head=2, 
            n_layers=1,
            use_image=False,
            image_size=None
    ):
        
        assert (
            not use_image and image_size == None or use_image and image_size != None
        ), "image_size must be None if you`re not using image or not None if you`re using image"
        
        super(TransductiveGAT, self).__init__()
        
        self.use_image = use_image
        
        hidden_dims = []
        factor = 1
        for i in range(1, n_layers + 1):
            hidden_dims.append(int(hidden_dim * factor))
            if i <= n_layers // 2: factor *= 2
            else: factor /= 2
        
        heads = [1] + [head] * n_layers
        
        
        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        if use_image:
            hidden_dims[0] += image_size
                    
        self.conv_layers = nn.ModuleList(
            GATBlock(h_in * n_head_in, h_out, n_head_out)
            for h_in, h_out, n_head_in, n_head_out in zip(hidden_dims, hidden_dims[1:], heads, heads[1:])
        )
        
        
        self.decoder = nn.Sequential(
            nn.Linear(heads[-1] * hidden_dims[-1], hidden_dims[-1]),
            nn.LeakyReLU(negative_slope=0.1),
            # nn.Dropout(p=0.),
            nn.Linear(hidden_dims[-1], n_out)
        )

    def forward(self, dataset, images=None):
        
        x, edge_index = dataset.x, dataset.edge_index
        
        x = self.encoder(x)
        
        if self.use_image:
            x = torch.cat([x, images], dim=-1)
        
        for layer in self.conv_layers:
            x = layer(x, edge_index)

        return self.decoder(x)
    
# INDUCTIVE MODELS WITH IMGS

# GAT
class InductiveGATwithIMGS(nn.Module):
    def __init__(self, n_in, n_out, hidden_dim, head, cnn_in_channels, cnn_out_channels, n_layers=1):
        
        super(InductiveGATwithIMGS, self).__init__()
        
        hidden_dims = [hidden_dim] * (n_layers + 1)
        heads = [head] * n_layers
        
        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        heads = [1] + heads
                
        self.gconv_layers = nn.ModuleList(
            GATBlock(h_in * n_head_in, h_out, n_head_out)
            for h_in, h_out, n_head_in, n_head_out in zip(hidden_dims, hidden_dims[1:], heads, heads[1:])
        )
        
        self.cnn = CNNBlock(cnn_in_channels, cnn_out_channels)
        self.pre_gnn = nn.Linear(hidden_dim, hidden_dims[0])
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1] * head + cnn_out_channels, hidden_dims[-1]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dims[-1], n_out)
        )

    def forward(self, data, use_pretrained=False, pretrained_model=None):
        
        x, edge_index, x_images = data.x, data.edge_index, data.imgs
        
        x = self.encoder(x)
        
        if use_pretrained:
            x_cnn = pretrained_model(x_images.float())
        else:
            x_cnn = self.cnn(x_images)
        
        x_gnn = self.pre_gnn(x)
        
        for layer in self.gconv_layers:
            x_gnn = layer(x_gnn, edge_index)
            
        x_cat = torch.cat((x_gnn, x_cnn), dim=1)  
            
        return self.decoder(x_cat)
    
    @torch.no_grad()
    def inference(self, x_all, loader, x_images_all, use_pretrained=False, pretrained_model=None):
        self.eval()
        
        x_all = self.encoder(x_all)
        
        if use_pretrained:
            x_cnn_all = pretrained_model(x_images_all.float())
        else:
            x_cnn_all = self.cnn(x_images_all)
        
        x_gnn_all = self.pre_gnn(x_all)
        
        for layer in self.gconv_layers:
            xs = []
            for batch in loader:
                x = x_gnn_all[batch.n_id]
                edge_index = batch.edge_index
                out = layer(x, edge_index)[:batch.batch_size]
                xs.append(out)
            x_gnn_all = torch.cat(xs, dim=0)
        
        x_cat = torch.cat([x_gnn_all, x_cnn_all], dim=1)
        
        return self.decoder(x_gnn_all)
    
# GCN
class InductiveGCNwithIMGS(nn.Module):

    def __init__(
        self,
        n_in=128,
        n_out=1,
        hidden_dim=128,
        n_layers=1,
        cnn_in_channels=12,
        cnn_out_channels=128,
        image_size=None,
    ):
        
        super(InductiveGCNwithIMGS, self).__init__()
        
        hidden_dims = []
        factor = 1
        for i in range(1, n_layers + 1):
            hidden_dims.append(int(hidden_dim * factor))
            if i <= n_layers // 2: factor *= 2
            else: factor /= 2
        
        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
        )

        self.cnn = CNNBlock(cnn_in_channels, cnn_out_channels)


        self.gconv_layers = nn.ModuleList(
            GCNBlock(h_in, h_out)
            for h_in, h_out in zip(
                hidden_dims, hidden_dims[1:]
            )
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(cnn_out_channels+hidden_dims[-1], hidden_dims[-1]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dims[-1], n_out),
        )

    def forward(self, data, use_pretrained=False, pretrained_model=None):

        x, edge_index, imgs = data.x, data.edge_index, data.imgs

        x = self.encoder(x)

        for layer in self.gconv_layers:
            x = layer(x, edge_index)
            
        if use_pretrained:
            x_cnn = pretrained_model(imgs.float())
        else:
            x_cnn = self.cnn(imgs.float())
        
        x_cat = torch.cat((x, x_cnn), dim=1)

        return self.decoder(x_cat)

    @torch.no_grad()
    def inference(self, x_all, loader, images, use_pretrained=False, pretrained_model=None):
        
        self.eval()
        
        x_all = self.encoder(x_all)
                
        for layer in self.gconv_layers:
            
            xs = []
            
            for batch in loader:
                
                x = x_all[batch.n_id]
                edge_index = batch.edge_index
                out = layer(x, edge_index)[:batch.batch_size]
                xs.append(out)               
            
            x_all = torch.cat(xs, dim=0)
            
        if use_pretrained:
            x_cnn = pretrained_model(images.float())
        else:
            x_cnn = self.cnn(images.float())
            
        x = torch.cat((x_all, x_cnn), dim=1)
        
        return self.decoder(x)
    
    
    
# CNNs
    

# CUSTOM CNN FOR EXTRACTING EMBEDDINGS
class RasterCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, 5, padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=(0, 0))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=(0, 0))
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.AvgPool2d(kernel_size=40, stride=1, padding=0)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x, return_hidden=False):
        
        x = F.relu(self.conv1(x))
        x = self.bn1(x)


        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)


        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        if return_hidden: hidden_state = x

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x if not return_hidden else (x, hidden_state)
    
# DEFAULT CNN FOR INDUCTIVE LEARNING
class CNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):   
        super(CNNBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1)  
        self.conv3 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(36, 256)
        self.fc2 = nn.Linear(256, out_channels) 
    
    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    
# UNSUPERVISED LEARNING

# Infomax
class DefaultEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_channels):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_channels)
        )
        
        self.convs = nn.ModuleList([
            SAGEConv(hidden_channels, 256),
            SAGEConv(256, 512),
            SAGEConv(512, hidden_channels)
        ])

        self.activations = nn.ModuleList()
        self.activations.extend([
            nn.PReLU(256),
            nn.PReLU(512),
            nn.PReLU(hidden_channels)
        ])

    def forward(self, x, edge_index, batch_size):
        x = self.encoder(x)
        
        for conv, act in zip(self.convs, self.activations):
            x = conv(x, edge_index)
            x = act(x)
            
        return x[:batch_size]
    
class GATEncoder(nn.Module):
    def __init__(
        self,
        n_in=128,
        n_out=1,
        hidden_dims=[256, 256],
        heads=[2],
    ):

        super(GATEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        
        heads = [1] + heads

        self.conv_layers = nn.ModuleList(
            GATBlock(h_in * n_head_in, h_out, n_head_out)
            for h_in, h_out, n_head_in, n_head_out in zip(
                hidden_dims, hidden_dims[1:], heads, heads[1:]
            )
        )

        self.decoder = nn.Sequential(
            nn.Linear(heads[-1] * hidden_dims[-1], hidden_dims[-1] * 4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dims[-1] * 4, n_out),
        )

    def forward(self, x, edge_index, batch_size):
        x = self.encoder(x)

        for layer in self.conv_layers:
            x = layer(x, edge_index)
        
        x = self.decoder(x)[:batch_size]
        
        return x