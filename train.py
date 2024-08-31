import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image

from model import Sample_UNet,UNet
from utils import GaussianDiffusion

import os


def save_checkpoint(model, optimizer, epoch, loss, file_path):
    # 检查并创建目录
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "hyperparameters": {
            "learning_rate": optimizer.param_groups[0]["lr"],
            # 添加其他超参数
        },
        "rng_state": torch.get_rng_state(),
    }
    torch.save(checkpoint, os.path.join(file_path, f"checkpoint_{epoch}.pth"))


# train
def train(
    model,
    dataloader,
    optimizer,
    epochs,
    utils,
    device,
    save_interval=10,
):
    for epoch in range(epochs):
        for batch in dataloader:
            x_0 = batch.to(device)
            t = torch.randint(0, utils.n_step, (x_0.shape[0],), device=device).long()
            noise = torch.randn_like(x_0)
            x_t = utils.q_sample(x_0, t, noise)  # 添加噪声
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        if epoch % save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=loss.item(),
                file_path="./checkpoints",
            )
    return model


# dataset
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx+1}.png")
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image

"""
## Hyperparameters
"""

batch_size = 32
num_epochs = 100  # Just for the sake of demonstration
total_timesteps = 300
norm_groups = 8  # Number of groups used in GroupNormalization layer
learning_rate = 2e-4

img_size = 64
img_channels = 3
clip_min = -1.0
clip_max = 1.0

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # Number of residual blocks


transform = transforms.Compose([
    transforms.Resize((img_size,img_size)), # resize to 64x64
    transforms.Lambda(lambda x: x.to(torch.float32)),
    transforms.Lambda(lambda x: x * 2 - 1), # normalize to [-1, 1]
  ])
dataset = CustomImageDataset(img_dir="./dataset/64x64/train/nolabel",transform=transform)
dataloader = DataLoader(dataset,batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Sample_UNet()
model = UNet(in_channels=img_channels, widths=widths, has_attention=has_attention, num_res_blocks=1, norm_groups=norm_groups)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

utils = GaussianDiffusion(0.0001, 0.02, 1000)

train(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    epochs=300,
    utils=utils,
    device=device,
)
