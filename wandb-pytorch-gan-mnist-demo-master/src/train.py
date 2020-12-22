import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml

from dataset import get_mnist
from networks import Discriminator, Generator

# # Number of workers for dataloader
# workers = 2

# # Batch size during training
# batch_size = 128

# # Spatial size of training images. All images will be resized to this
# #   size using a transformer.
# image_size = 32

# # Number of channels in the training images. For color images this is 3
# nc = 3

# # Size of z latent vector (i.e. size of generator input)
# nz = 100

# # Size of feature maps in generator
# ngf = 64

# # Size of feature maps in discriminator
# ndf = 64

# # Number of training epochs
# num_epochs = 30

# # Learning rate for optimizers
# lr = 0.0002

# # Beta1 hyperparam for Adam optimizers
# beta1 = 0.5

# # Number of GPUs available. Use 0 for CPU mode.
# ngpu = 1

# parser = argparse.ArgumentParser()
# parser.add_argument('--workers', type=int, default=2)
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--nc', type=int, default=3)
# parser.add_argument('--nz', type=int, default=100)
# parser.add_argument('--ngf', type=int, default=64)
# parser.add_argument('--ndf', type=int, default=64)
# parser.add_argument('--num_epochs', type=int, default=30)
# parser.add_argument('--lr', type=float, default=0.0002)
# parser.add_argument('--beta1', type=float, default=0.5)
# parser.add_argument('--ngpu', type=int, default=1)
# parser.add_argument('--log_dir', type=str, default="log")
# opt = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--params', type=str, required=True)
opt = parser.parse_args()
params_path = Path.cwd() / Path(opt.params)
params = yaml.safe_load(params_path.read_text())

params['cuda'] = params['cuda'] and torch.cuda.is_available()

wandb.login()
wandb.init(project='wandb-pytorch-gan-mnist-demo', config=params)
run_id = wandb.run.id
run_log_dir = Path(params['log_dir']) / run_id
run_log_dir.mkdir(exist_ok=True, parents=True)

dataset = get_mnist()
dataloader = DataLoader(
    dataset,
    batch_size=params['batch_size'],
    shuffle=True,
    num_workers=4,
    drop_last=True
)

device = torch.device('cuda:0' if params['cuda'] else 'cpu')

d_net = Discriminator().to(device)
g_net = Generator(nz=params['nz']).to(device)

wandb.watch(d_net)
wandb.watch(g_net)

bce = nn.BCELoss()

d_opt = torch.optim.Adam(d_net.parameters(), lr=params['d_lr'], betas=(0.5, 0.999))
g_opt = torch.optim.Adam(g_net.parameters(), lr=params['g_lr'], betas=(0.5, 0.999))

fixed_noise = torch.randn(64, params['nz']).to(device)

for epoch in tqdm(range(1, params['num_epochs'] + 1), desc='Training'):

    d_net.train()
    g_net.train()

    total_g_loss_train = 0
    total_d_real_loss_train = 0
    total_d_fake_loss_train = 0
    total_d_loss_train = 0

    for i, (real_image, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch :4d}', leave=False), 1):

        noise = torch.randn(params['batch_size'], params['nz']).to(device)

        real_image = real_image.to(device)
        fake_image = g_net(noise)

        real_label = torch.ones(params['batch_size'], 1).to(device)
        fake_label = torch.zeros(params['batch_size'], 1).to(device)

        g_net.zero_grad()
        g_loss = bce(d_net(fake_image), real_label)
        total_g_loss_train += g_loss.item()
        g_loss.backward()
        g_opt.step()

        d_net.zero_grad()
        d_real_loss = bce(d_net(real_image), real_label)
        d_fake_loss = bce(d_net(fake_image.detach()), fake_label)
        d_loss = d_real_loss + d_fake_loss
        total_d_real_loss_train += d_real_loss.item()
        total_d_fake_loss_train += d_fake_loss.item()
        total_d_loss_train += d_loss.item()
        d_loss.backward()
        d_opt.step()
    
    g_loss_train = total_g_loss_train / len(dataloader)
    d_real_loss_train = total_d_real_loss_train / len(dataloader)
    d_fake_loss_train = total_d_fake_loss_train / len(dataloader)
    d_loss_train = total_d_loss_train / len(dataloader)

    g_net.eval()
    with torch.no_grad():
        fixed_fake_image = g_net(fixed_noise).detach()

    wandb.log({
        'g_loss_train': g_loss_train,
        'd_real_loss_train': d_real_loss_train,
        'd_fake_loss_train': d_fake_loss_train,
        'd_loss_train': d_loss_train,
        'examples': [wandb.Image(i) for i in fixed_fake_image]
    })

    torch.save(g_net.state_dict(), run_log_dir / f'generator-{epoch :04d}.pth')
    torch.save(d_net.state_dict(), run_log_dir / f'discriminator-{epoch :04d}.pth')