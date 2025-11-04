# import torch
# import os
# from tqdm import trange
# import argparse
# from torchvision import datasets, transforms
# import torch.nn as nn
# import torch.optim as optim
# from model import Generator, Discriminator
# from utils import D_train, G_train, save_models
# from tqdm import tqdm
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
#     parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
#     parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
#     parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD.")
#     parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
#     args = parser.parse_args()

#     to_download=False
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         device_type = "cuda"
#         print(f"Using device: CUDA")
#         # Use all available GPUs if args.gpus is -1
#         if args.gpus == -1:
#             args.gpus = torch.cuda.device_count()
#             print(f"Using {args.gpus} GPUs.")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#         device_type = "mps"
#         print(f"Using device: MPS (Apple Metal)")
#     else:
#         device = torch.device("cpu")
#         device_type = "cpu"
#         print(f"Using device: CPU")
        

    

#     # Create directories
#     os.makedirs('checkpoints', exist_ok=True)
#     data_path = os.getenv('DATA')
#     if data_path is None:
#         data_path = "data"
#         to_download = True
#     # Data Pipeline
#     print('Dataset loading...')
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5,), std=(0.5,))
#     ])
#     train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=to_download)
#     test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=to_download)

#     train_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=4,  # Use multiple workers for data loading
#         pin_memory=True  # Faster data transfer to GPU
#     )
#     test_loader = torch.utils.data.DataLoader(
#         dataset=test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
#     print('Dataset loaded.')

#     # Model setup
#     print('Model loading...')
#     mnist_dim = 784
#     G = Generator(g_output_dim=mnist_dim).to(device)
#     D = Discriminator(mnist_dim).to(device)

#     # Wrap models in DataParallel if multiple GPUs are available
#     if args.gpus > 1:
#         G = torch.nn.DataParallel(G)
#         D = torch.nn.DataParallel(D)
#     print('Model loaded.')

#     # Loss and optimizers
#     criterion = nn.BCELoss()
#     G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
#     D_optimizer = optim.Adam(D.parameters(), lr=args.lr)


#     print('Start training:')
#     n_epoch = args.epochs
#     for epoch in tqdm(range(1, n_epoch + 1)):
#         for batch_idx, (x, _) in enumerate(train_loader):
#             x = x.view(-1, mnist_dim).to(device)
#             D_train(x, G, D, D_optimizer, criterion, device)
#             G_train(x, G, D, G_optimizer, criterion, device)

#         if epoch % 10 == 0:
#             save_models(G, D, 'checkpoints')

#     print('Training done.')


import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from model import Generator, Discriminator
from utils import save_models


N_CRITIC = 2
BETA1 = 0.5


def calculate_gradient_penalty(
    D: torch.nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
    create_graph: bool = True,
) -> torch.Tensor:
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    disc_interpolates = D(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=create_graph,
        retain_graph=create_graph,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


def compute_d_loss(
    real_data: torch.Tensor,
    G: torch.nn.Module,
    D: torch.nn.Module,
    device: torch.device,
    *,
    create_graph: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = real_data.size(0)

    real_logits = D(real_data)
    d_loss_real = -real_logits.mean()

    z = torch.randn(batch_size, 100, device=device)
    fake_data = G(z).detach()
    fake_logits = D(fake_data)
    d_loss_fake = fake_logits.mean()

    gradient_penalty = calculate_gradient_penalty(
        D, real_data, fake_data, device, create_graph=create_graph
    )

    d_loss = d_loss_real + d_loss_fake + gradient_penalty
    wasserstein_distance = real_logits.mean() - fake_logits.mean()
    return d_loss, wasserstein_distance


def compute_g_loss(batch_size: int, G: torch.nn.Module, D: torch.nn.Module, device: torch.device) -> torch.Tensor:
    z = torch.randn(batch_size, 100, device=device)
    fake_data = G(z)
    fake_logits = D(fake_data)
    return -fake_logits.mean()


def average(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def evaluate_losses(
    loader: torch.utils.data.DataLoader,
    G: torch.nn.Module,
    D: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    was_training_g = G.training
    was_training_d = D.training
    G.eval()
    D.eval()

    critic_losses: List[float] = []
    generator_losses: List[float] = []

    for real_images, _ in loader:
        real_images = real_images.to(device)
        d_loss, _ = compute_d_loss(real_images, G, D, device, create_graph=False)
        critic_losses.append(float(d_loss.detach().cpu().item()))

        with torch.no_grad():
            g_loss = compute_g_loss(real_images.size(0), G, D, device)
        generator_losses.append(float(g_loss.cpu().item()))

    if was_training_g:
        G.train()
    if was_training_d:
        D.train()

    G.zero_grad(set_to_none=True)
    D.zero_grad(set_to_none=True)
    return average(critic_losses), average(generator_losses)


def save_loss_history(history: Dict[str, List[float]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    args = parser.parse_args()


    to_download=False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
        if args.gpus == -1:
            args.gpus = torch.cuda.device_count()
            print(f"Using {args.gpus} GPUs.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
        
    os.makedirs('checkpoints', exist_ok=True)
    data_path = os.getenv('DATA')
    if data_path is None:
        data_path = "data"
        to_download = True
    

    print('Dataset loading...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=to_download)
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=to_download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print('Dataset loaded.')


    print('Model loading...')
    G = Generator().to(device)
    D = Discriminator().to(device)

    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
    print('Model loaded.')


    # WGAN-GP #####################
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(BETA1, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(BETA1, 0.999))


    print('Start training (WGAN-GP Mode):')
    history: Dict[str, List[float]] = {
        "epochs": [],
        "train_d_loss": [],
        "train_g_loss": [],
        "train_total_loss": [],
        "train_w_distance": [],
        "val_d_loss": [],
        "val_g_loss": [],
        "val_total_loss": [],
    }

    epoch_iterator = tqdm(range(1, args.epochs + 1), desc="baseline-epoch", leave=False)
    for epoch in epoch_iterator:
        train_d_losses: List[float] = []
        train_g_losses: List[float] = []
        train_wd: List[float] = []

        batch_iterator = tqdm(train_loader, desc=f"train-{epoch}", leave=False)
        for batch_idx, (real_images, _) in enumerate(batch_iterator):
            real_images = real_images.to(device)

            D.zero_grad(set_to_none=True)
            d_loss, w_dist = compute_d_loss(real_images, G, D, device)
            d_loss.backward()
            D_optimizer.step()

            train_d_losses.append(float(d_loss.detach().cpu().item()))
            train_wd.append(float(w_dist.detach().cpu().item()))

            if batch_idx % N_CRITIC == 0:
                G.zero_grad(set_to_none=True)
                g_loss = compute_g_loss(real_images.size(0), G, D, device)
                g_loss.backward()
                G_optimizer.step()
                train_g_losses.append(float(g_loss.detach().cpu().item()))

            postfix = {"d_loss": train_d_losses[-1], "w_dist": train_wd[-1]}
            if train_g_losses:
                postfix["g_loss"] = train_g_losses[-1]
            batch_iterator.set_postfix(postfix)

        avg_train_d = average(train_d_losses)
        avg_train_g = average(train_g_losses)
        avg_train_total = avg_train_d + avg_train_g
        avg_train_wd = average(train_wd)

        val_d, val_g = evaluate_losses(test_loader, G, D, device)
        val_total = val_d + val_g

        history["epochs"].append(epoch)
        history["train_d_loss"].append(avg_train_d)
        history["train_g_loss"].append(avg_train_g)
        history["train_total_loss"].append(avg_train_total)
        history["train_w_distance"].append(avg_train_wd)
        history["val_d_loss"].append(val_d)
        history["val_g_loss"].append(val_g)
        history["val_total_loss"].append(val_total)

        epoch_iterator.set_postfix(
            train_total=f"{avg_train_total:.4f}", val_total=f"{val_total:.4f}", w_dist=f"{avg_train_wd:.4f}"
        )

        if epoch % 10 == 0:
            save_models(G, D, "checkpoints")

    save_models(G, D, "checkpoints")
    save_loss_history(history, Path("results/training_logs/baseline.json"))
    print("Training done. Loss history saved to results/training_logs/baseline.json")
