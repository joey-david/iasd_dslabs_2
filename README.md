[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SQRTf064)
# DataLabAssignement2

## Using GPUs with MesoNet

### 1. Access MesoNet
To use the provided GPUs, you must first:
- Create a MesoNet account and request access to the project.
- Once accepted, add your SSH key to MesoNet for secure access. Follow the instructions [here](https://www.mesonet.fr/documentation/user-documentation/code_form/juliet/connexion).

### 2. Set Up Your Environment
On Juliet (MesoNet's cluster), you need to:

1. Create a virtual environment for Python:
   ```bash
   python -m venv venv
   ```

2. Activate the environment:
   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Run Training or Generation
- Use the provided shell scripts (`scripts/train.sh` or `scripts/generate.sh`) to launch jobs.
- These scripts will automatically activate the virtual environment (`venv`).
- If you don't use Juliet cluster, please provide a data_path argument for the training scripts. Otherwise, everything should be automated.

### 4. Hardware Compatibility
The code is compatible with:
- **NVIDIA GPUs (CUDA)**
- **Apple Silicon (MPS, for M1/M2/M3/M4 chips)**
- **CPU-only mode** (not recommended for performance)

We recommend using CUDA (for NVIDIA GPUs) or MPS (for Apple Silicon).

### 5. Slurm Resources
- If you need help with Slurm (MesoNet's job scheduler), there will be a Slurm lecture on 29/10/2025 or refer to the official documentation: [Slurm Documentation](https://slurm.schedmd.com/documentation.html).

### 6. Example Commands
To submit a job to MesoNet:
```bash
# Make the script executable
chmod +x scripts/train.sh

# Submit the job to Slurm
scripts/train.sh
```

## Running with Docker

Build the GPU-enabled image once (the build pre-downloads the Inception weights used for the metrics):

```bash
docker build -t mnist-gan-eval .
```

On the remote GPU node, mount the checkpoints, evaluation outputs, MNIST data cache, and (optionally) any figures you want to persist. Then launch the metrics script inside the container:

```bash
docker run --rm \
  --gpus all \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v "$PWD/checkpoints":/app/checkpoints \
  -v "$PWD/checkpoints_diffaug":/app/checkpoints_diffaug \
  -v "$PWD/results":/app/results \
  -v "$PWD/data":/app/data \
  -v "$HOME/.cache/torch":/app/.cache/torch \
  mnist-gan-eval \
  evaluate_metrics.py \
    --checkpoint baseline=checkpoints \
    --checkpoint diffaug=checkpoints_diffaug \
    --num-samples 10000 \
    --batch-size 512
```

The script writes metric reports to `results/docker_eval/<label>/metrics.json` and saves a sample grid for each evaluated checkpoint. If `checkpoints_diffaug` is unavailable you can omit that `--checkpoint` argument. Increase `--num-samples` to the desired evaluation size (10 000 matches the testing platform) and reuse the mounted caches to avoid repeated downloads on subsequent runs.
## train.py
This script performs adversarial training for a GAN. Before running it, ensure the `DATA` variable in `scripts/train.sh` points to your dataset directory. If left unchanged, the script will create a default `data` folder in the repository root. If you use Juliet, no changes are required—the default path is already configured. You can adjust the learning rate, number of epochs, and batch size directly in `scripts/train.sh`.

## generate.py
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples. You can launch this script on juliet with `scripts/generate.sh`

Example:
  > python3 generate.py --bacth_size 64

or 

```bash
# Make the script executable
chmod +x scripts/generate.sh

# Submit the job to Slurm
scripts/generate.sh
```

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt


## Checkpoints
Push the minimal amount of models in the folder *checkpoints*.
