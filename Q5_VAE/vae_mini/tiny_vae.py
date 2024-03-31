#!/usr/bin/env python

# @XREMOTE_HOST: elk.fleuret.org
# @XREMOTE_EXEC: python
# @XREMOTE_PRE: source ${HOME}/misc/venv/pytorch/bin/activate
# @XREMOTE_PRE: ln -sf ${HOME}/data/pytorch ./data
# @XREMOTE_GET: *.png

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import sys, os, argparse, time, math

import torch, torchvision

from torch import optim, nn
from torch.nn import functional as F

######################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################

parser = argparse.ArgumentParser(
    description="Very simple implementation of a VAE for teaching."
)

parser.add_argument("--nb_epochs", type=int, default=25)

parser.add_argument("--learning_rate", type=float, default=1e-3)

parser.add_argument("--batch_size", type=int, default=100)

parser.add_argument("--data_dir", type=str, default="./data/")

parser.add_argument("--log_filename", type=str, default="train.log")

parser.add_argument("--latent_dim", type=int, default=32)

parser.add_argument("--nb_channels", type=int, default=32)

parser.add_argument("--no_dkl", action="store_true")

parser.add_argument("--beta", type=float, default=1.0)

args = parser.parse_args()

log_file = open(args.log_filename, "w")

######################################################################


def log_string(s):
    t = time.strftime("%Y-%m-%d_%H:%M:%S ", time.localtime())

    if log_file is not None:
        log_file.write(t + s + "\n")
        log_file.flush()

    print(t + s)
    sys.stdout.flush()


######################################################################


def sample_categorical(param):
    dist = torch.distributions.Categorical(logits=param)
    return (dist.sample().unsqueeze(1).float() - train_mu) / train_std


def log_p_categorical(x, param):
    x = (x.squeeze(1) * train_std + train_mu).long().clamp(min=0, max=255)
    param = param.permute(0, 3, 1, 2)
    return -F.cross_entropy(param, x, reduction="none").flatten(1).sum(dim=1)


def sample_gaussian(param):
    mean, log_var = param
    std = log_var.mul(0.5).exp()
    return torch.randn(mean.size(), device=mean.device) * std + mean


def log_p_gaussian(x, param):
    mean, log_var, x = param[0].flatten(1), param[1].flatten(1), x.flatten(1)
    var = log_var.exp()
    return -0.5 * (((x - mean).pow(2) / var) + log_var + math.log(2 * math.pi)).sum(1)


def dkl_gaussians(param_a, param_b):
    mean_a, log_var_a = param_a[0].flatten(1), param_a[1].flatten(1)
    mean_b, log_var_b = param_b[0].flatten(1), param_b[1].flatten(1)
    var_a = log_var_a.exp()
    var_b = log_var_b.exp()
    return 0.5 * (
        log_var_b - log_var_a - 1 + (mean_a - mean_b).pow(2) / var_b + var_a / var_b
    ).sum(1)


def dup_param(param, nb):
    mean, log_var = param
    s = (nb,) + (-1,) * (mean.dim() - 1)
    return (mean.expand(s), log_var.expand(s))


######################################################################


class VariationalAutoEncoder(nn.Module):
    def __init__(self, nb_channels, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, nb_channels, kernel_size=1),  # to 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=5),  # to 24x24
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=5),  # to 20x20
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=4, stride=2),  # to 9x9
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=2),  # to 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels, 2 * latent_dim, kernel_size=4),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, nb_channels, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                nb_channels, nb_channels, kernel_size=3, stride=2
            ),  # from 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                nb_channels, nb_channels, kernel_size=4, stride=2
            ),  # from 9x9
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nb_channels, nb_channels, kernel_size=5),  # from 20x20
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nb_channels, 2, kernel_size=5),  # from 24x24
        )

    def encode(self, x):
        output = self.encoder(x).view(x.size(0), 2, -1)
        mu, log_var = output[:, 0], output[:, 1]
        return mu, log_var

    def decode(self, z):
        # return self.decoder(z.view(z.size(0), -1, 1, 1)).permute(0, 2, 3, 1)
        output = self.decoder(z.view(z.size(0), -1, 1, 1))
        mu, log_var = output[:, 0:1], output[:, 1:2]
        log_var.flatten(1)[...] = 1  # math.log(1e-1)
        # log_var.flatten(1)[...] = log_var.flatten(1)[:, :1]
        # log_var = log_var.clamp(min=2*math.log(1/256))
        return mu, log_var


######################################################################

data_dir = os.path.join(args.data_dir, "mnist")

train_set = torchvision.datasets.MNIST(data_dir, train=True, download=True)
train_input = train_set.data.view(-1, 1, 28, 28).float()

test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True)
test_input = test_set.data.view(-1, 1, 28, 28).float()

######################################################################


def save_images(model, prefix=""):
    def save_image(x, filename):
        x = x * train_std + train_mu
        x = x.clamp(min=0, max=255) / 255
        torchvision.utils.save_image(1 - x, filename, nrow=12, pad_value=1.0)
        log_string(f"wrote {filename}")

    # Save a bunch of train images

    x = train_input[:36]
    save_image(x, f"{prefix}train_input.png")

    # Save the same images after encoding / decoding

    param_q_Z_given_x = model.encode(x)
    z = sample_gaussian(param_q_Z_given_x)
    param_p_X_given_z = model.decode(z)
    x = sample_gaussian(param_p_X_given_z)
    save_image(x, f"{prefix}train_output.png")
    save_image(param_p_X_given_z[0], f"{prefix}train_output_mean.png")

    # Save a bunch of test images

    x = test_input[:36]
    save_image(x, f"{prefix}input.png")

    # Save the same images after encoding / decoding

    param_q_Z_given_x = model.encode(x)
    z = sample_gaussian(param_q_Z_given_x)
    param_p_X_given_z = model.decode(z)
    x = sample_gaussian(param_p_X_given_z)
    save_image(x, f"{prefix}output.png")
    save_image(param_p_X_given_z[0], f"{prefix}output_mean.png")

    # Generate a bunch of images

    z = sample_gaussian(dup_param(param_p_Z, x.size(0)))
    param_p_X_given_z = model.decode(z)
    x = sample_gaussian(param_p_X_given_z)
    save_image(x, f"{prefix}synth.png")
    save_image(param_p_X_given_z[0], f"{prefix}synth_mean.png")


######################################################################

model = VariationalAutoEncoder(nb_channels=args.nb_channels, latent_dim=args.latent_dim)

model.to(device)

######################################################################

train_input, test_input = train_input.to(device), test_input.to(device)

train_mu, train_std = train_input.mean(), train_input.std()
train_input.sub_(train_mu).div_(train_std)
test_input.sub_(train_mu).div_(train_std)

######################################################################

zeros = train_input.new_zeros(1, args.latent_dim)

param_p_Z = zeros, zeros

for n_epoch in range(args.nb_epochs):
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    acc_loss = 0

    for x in train_input.split(args.batch_size):
        param_q_Z_given_x = model.encode(x)
        z = sample_gaussian(param_q_Z_given_x)
        param_p_X_given_z = model.decode(z)
        log_p_x_given_z = log_p_gaussian(x, param_p_X_given_z)

        if args.no_dkl:
            log_q_z_given_x = log_p_gaussian(z, param_q_Z_given_x)
            log_p_z = log_p_gaussian(z, param_p_Z)
            log_p_x_z = log_p_x_given_z + log_p_z
            loss = -(log_p_x_z - log_q_z_given_x).mean()
        else:
            dkl_q_Z_given_x_from_p_Z = dkl_gaussians(param_q_Z_given_x, param_p_Z)
            loss = -(log_p_x_given_z - args.beta * dkl_q_Z_given_x_from_p_Z).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_loss += loss.item() * x.size(0)

    log_string(f"acc_loss {n_epoch} {acc_loss/train_input.size(0)}")

    if (n_epoch + 1) % 25 == 0:
        save_images(model, f"epoch_{n_epoch+1:04d}_")

######################################################################