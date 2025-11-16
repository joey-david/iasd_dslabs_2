import torch 
import torchvision
import os
import argparse
import numpy as np

from model import Generator, Discriminator
from utils import load_model

def rejection_sampling(G, D, batch_size, device, threshold=0.8, max_attempts=10):
    """
    Rejection sampling: only keep samples that D scores above threshold
    """
    accepted_samples = []
    attempts = 0
    
    while len(accepted_samples) < batch_size and attempts < max_attempts:
        # Generate a larger batch to improve efficiency
        z = torch.randn(batch_size * 2, 100).to(device)
        x = G(z)
        
        # Get discriminator scores
        with torch.no_grad():
            scores = D(x).squeeze()
        
        # Accept samples above threshold
        mask = scores > threshold
        accepted = x[mask]
        
        if len(accepted) > 0:
            accepted_samples.append(accepted)
        
        attempts += 1
    
    if len(accepted_samples) == 0:
        # Fallback: return best samples from last attempt
        print(f"Warning: No samples met threshold {threshold}, returning best samples")
        z = torch.randn(batch_size, 100).to(device)
        x = G(z)
        return x
    
    accepted_samples = torch.cat(accepted_samples, dim=0)
    return accepted_samples[:batch_size]


def metropolis_hastings_sampling(G, D, batch_size, device, n_steps=5):
    """
    Metropolis-Hastings sampling in latent space
    """
    # Initialize with random samples
    z_current = torch.randn(batch_size, 100).to(device)
    x_current = G(z_current)
    
    with torch.no_grad():
        score_current = D(x_current).squeeze()
    
    for step in range(n_steps):
        # Propose new samples
        z_proposal = z_current + torch.randn_like(z_current) * 0.1
        x_proposal = G(z_proposal)
        
        with torch.no_grad():
            score_proposal = D(x_proposal).squeeze()
        
        # Acceptance probability
        accept_prob = torch.min(torch.ones_like(score_proposal), 
                                score_proposal / (score_current + 1e-8))
        
        # Accept or reject
        rand = torch.rand_like(accept_prob)
        accept_mask = rand < accept_prob
        
        # Update accepted samples
        z_current[accept_mask] = z_proposal[accept_mask]
        score_current[accept_mask] = score_proposal[accept_mask]
    
    x_final = G(z_current)
    return x_final


def langevin_sampling(G, D, batch_size, device, n_steps=10, step_size=0.01):
    """
    Langevin dynamics sampling in latent space
    """
    z = torch.randn(batch_size, 100, device=device, requires_grad=True)
    
    for step in range(n_steps):
        x = G(z)
        score = D(x)
        
        # Compute gradient of discriminator score w.r.t. z
        grad = torch.autograd.grad(score.sum(), z, create_graph=False)[0]
        
        # Langevin update
        noise = torch.randn_like(z) * np.sqrt(2 * step_size)
        z = z.detach() + step_size * grad + noise
        z.requires_grad = True
    
    z = z.detach()
    x_final = G(z)
    return x_final


def top_k_sampling(G, D, batch_size, device, k_multiplier=5):
    """
    Generate k times more samples and keep the top k based on D scores
    """
    k = batch_size * k_multiplier
    z = torch.randn(k, 100).to(device)
    x = G(z)
    
    with torch.no_grad():
        scores = D(x).squeeze()
    
    # Get top-k samples
    _, top_indices = torch.topk(scores, batch_size)
    return x[top_indices]





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples with advanced sampling methods.')
    parser.add_argument("--batch_size", type=int, default=64,
                      help="The batch size to use for generation.")
    parser.add_argument("--method", type=str, default="topk",
                      choices=["standard", "rejection", "metropolis",  "topk"],
                      help="Sampling method to use.")
    parser.add_argument("--threshold", type=float, default=0.8,
                      help="Threshold for rejection sampling.")
    parser.add_argument("--mh_steps", type=int, default=5,
                      help="Number of Metropolis-Hastings steps.")
    parser.add_argument("--langevin_steps", type=int, default=10,
                      help="Number of Langevin dynamics steps.")
    parser.add_argument("--k_multiplier", type=int, default=5,
                      help="Multiplier for top-k sampling.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    print('Model Loading...')
    mnist_dim = 784

    G = Generator(g_output_dim=mnist_dim).to(device)
    G_path = os.path.join('checkpoints', 'pr_200_G.pth')
    if os.path.exists(G_path):
        G_ckpt = torch.load(G_path, map_location=device)
        G.load_state_dict({k.replace('module.', ''): v for k, v in G_ckpt.items()})
        print('Generater loaded.')
    
    # Load discriminator for sampling methods that need it
    D = None
    if args.method != "standard":
        D = Discriminator(mnist_dim).to(device)
        D_path = os.path.join('checkpoints', 'pr_200_D.pth')
        if os.path.exists(D_path):
            D_ckpt = torch.load(D_path, map_location=device)
            D.load_state_dict({k.replace('module.', ''): v for k, v in D_ckpt.items()})
            print('Discriminator loaded.')
        else:
            print('Warning: Discriminator not found, falling back to standard sampling.')
            args.method = "standard"
    
    if torch.cuda.device_count() > 1:
        G = torch.nn.DataParallel(G)
        if D is not None:
            D = torch.nn.DataParallel(D)
    
    G.eval()
    if D is not None:
        D.eval()

    print('Model loaded.')
    print(f'Sampling method: {args.method}')

    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples < 10000:
            # Choose sampling method
            if args.method == "standard":
                z = torch.randn(args.batch_size, 100).to(device)
                x = G(z)
            elif args.method == "rejection":
                x = rejection_sampling(G, D, args.batch_size, device, 
                                      threshold=args.threshold)
            elif args.method == "metropolis":
                x = metropolis_hastings_sampling(G, D, args.batch_size, device, 
                                                n_steps=args.mh_steps)
            elif args.method == "topk":
                x = top_k_sampling(G, D, args.batch_size, device, 
                                  k_multiplier=args.k_multiplier)
            
            x = x.reshape(-1, 28, 28)
            
            for k in range(min(x.shape[0], 10000 - n_samples)):
                torchvision.utils.save_image(x[k:k+1], 
                                           os.path.join('samples', f'{n_samples}.png'))
                n_samples += 1
                
            print(f'Generated {n_samples}/10000 samples', end='\r')
    
    print(f'\nGeneration complete! {n_samples} samples saved.')


    #reject 25
    #topk 22.9
    #metropolis 32

    # diversity 27

