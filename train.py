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


import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator

from utils import  save_models 
from tqdm import tqdm


N_CRITIC = 2 
BETA1 = 0.5   
# -------------------------------
import torch
import torch.nn as nn


def calculate_gradient_penalty(D, real_data, fake_data, device, lambda_gp=10):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    disc_interpolates = D(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


def D_train_wgangp(real_data, G, D, D_optimizer, device):
    D.zero_grad()
    batch_size = real_data.size(0)
    

    D_real = D(real_data)
    D_loss_real = -torch.mean(D_real)
    

    z = torch.randn(batch_size, 100).to(device)
    fake_data = G(z).detach()
    D_fake = D(fake_data)
    D_loss_fake = torch.mean(D_fake)
    

    gradient_penalty = calculate_gradient_penalty(D, real_data.data, fake_data.data, device)

   
    D_loss = D_loss_real + D_loss_fake + gradient_penalty
    
  
    D_loss.backward()
    D_optimizer.step()
    
    
    w_distance = D_real.mean() - D_fake.mean()
    return D_loss.data.item(), w_distance.data.item()


def G_train_wgangp(real_data, G, D, G_optimizer, device):
    G.zero_grad()
    batch_size = real_data.size(0)
    
 
    z = torch.randn(batch_size, 100).to(device)
    fake_data = G(z)
    D_fake = D(fake_data)
    
   
    G_loss = -torch.mean(D_fake)
    
   
    G_loss.backward()
    G_optimizer.step()
    return G_loss.data.item()


def D_train(x, G, D, D_optimizer, criterion, device):

    pass 
def G_train(x, G, D, G_optimizer, criterion, device):

    pass
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
        device_type = "cuda"
        print(f"Using device: CUDA")
        if args.gpus == -1:
            args.gpus = torch.cuda.device_count()
            print(f"Using {args.gpus} GPUs.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print(f"Using device: CPU")
        
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
    n_epoch = args.epochs
    for epoch in tqdm(range(1, n_epoch + 1)):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            

            D_loss, W_dist = D_train_wgangp(x, G, D, D_optimizer, device)
            

            if batch_idx % N_CRITIC == 0:
                G_loss = G_train_wgangp(x, G, D, G_optimizer, device)

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

    print('Training done.')
