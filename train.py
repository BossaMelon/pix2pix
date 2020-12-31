import torch
import torch.nn as nn
from tqdm.auto import tqdm

from dataloader import get_dataloader
from models.unet import UNet
from utils.util import device, save_tensor_images, crop, image_cat

criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
input_dim = 1
label_dim = 1
display_step = 20
batch_size = 4
lr = 0.0002
initial_shape = 512
target_shape = 373


def train():
    dataloader = get_dataloader()
    unet = UNet(input_dim, label_dim).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)

    unet_loss = 0.

    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader):
            # Flatten the image
            real = real.to(device)
            labels = labels.to(device)

            ### Update U-Net ###
            unet_opt.zero_grad()
            pred = unet(real)
            unet_loss = criterion(pred, labels)
            unet_loss.backward()
            unet_opt.step()

        print(f"Epoch {epoch}: U-Net loss: {unet_loss.item():.4f}")

        image_cat(
            crop(real, torch.Size([len(real), 1, target_shape, target_shape])),
            labels,
            torch.sigmoid(pred),
            size=(input_dim, target_shape, target_shape),
            file_name=f'unet-rlp-{epoch}'
        )

if __name__ == '__main__':
    train()
