import torch
import torchvision.transforms as transforms

class RasterDataset(torch.utils.data.Dataset):
    def __init__(self, images, y):

        self.images = images
        self.y = y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img = self.images[idx]
        y = self.y[idx]
        
        return img, y
    

class TimmRasterDataset(torch.utils.data.Dataset):
    def __init__(self, images, y, timm_transform):

        self.images = images.to(torch.float32)
        self.y = y
        self.transform = transforms.Resize((224, 224), antialias=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.transform(self.images[idx])
        
        return image