import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from prdc import compute_prdc
except ImportError:
    raise ImportError("Please install the prdc package (pip install prdc)")


def load_images(folder, limit=None):
    files = sorted(Path(folder).glob('*.png'))
    if limit:
        files = files[:limit]
    imgs = []
    for fp in tqdm(files, desc=f"Loading {folder}"):
        img = Image.open(fp).convert('RGB')
        arr = np.asarray(img, dtype=np.float32) / 255.0
        imgs.append(arr.transpose(2, 0, 1))
    return np.stack(imgs)


def main():
    parser = argparse.ArgumentParser(description="Compute PRDC metrics for GAN samples.")
    parser.add_argument('--real_dir', required=True, help='Directory of real reference images')
    parser.add_argument('--fake_dir', required=True, help='Directory of generated images')
    parser.add_argument('--sample_limit', type=int, default=None, help='Optional number of images to use')
    parser.add_argument('--nearest_k', type=int, default=5, help='k nearest neighbors for PRDC')
    args = parser.parse_args()

    real = load_images(args.real_dir, args.sample_limit)
    fake = load_images(args.fake_dir, args.sample_limit)

    real_features = real.reshape(real.shape[0], -1)
    fake_features = fake.reshape(fake.shape[0], -1)

    metrics = compute_prdc(real_features=real_features,
                           fake_features=fake_features,
                           nearest_k=args.nearest_k)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()
