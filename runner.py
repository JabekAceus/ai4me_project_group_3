import argparse
import json
import random
import subprocess
from pathlib import Path
from pprint import pprint
from shutil import rmtree

import numpy as np
import torch
import torch.nn.functional as F
import torch_optimizer as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


from architectures import ENet, ENet2_5D
from dataset import SliceDataset
from losses import (BoundaryLoss, CompoundLoss, CrossEntropy, DiceLoss,
                    FocalLoss, TverskyLoss)
from utils import (dice_coef, probs2class, probs2one_hot, save_images,
                   tqdm_)


CONFIGURATIONS = {
    
    "baseline": {"--loss_function": "cross_entropy", "--optimizer": "adam", "--online_augmentations": "none", "--architecture": "enet"},
    
    "dice_loss": {"--loss_function": "dice"},
    "focal_loss": {"--loss_function": "focal"},
    "tversky_loss": {"--loss_function": "tversky"},
    "compound_loss": {"--loss_function": "compound_dice_focal"},
    "boundary_loss": {
        "--loss_function": "boundary",
        "--dataset": "SEGTHOR_DIST",
        "_prepare_data": True,
        "_prepare_script": "prepare_data.py",
        "_prepare_args": "--preprocessing distance_maps"
    },
    "ce_dice_loss": {
        "--loss_function": "compound_ce_dice"},
    
    "adamw": {"--optimizer": "adamw"},
    "ranger": {"--optimizer": "ranger"},
    "cosine_scheduler": {"--scheduler": "cosine", "--optimizer": "adamw"},
    
    "enet_se": {"--architecture": "enet_se"},
    "enet_attention": {"--architecture": "enet_attention"},
    "true_2.5d": {
        "--architecture": "enet_2.5d",
        "--dataset": "SEGTHOR_2.5D",
        "--in_channels": 3,
        "_prepare_data": True,
        "_prepare_script": "prepare_data.py",
        "_prepare_args": "--slicing_mode 2.5d --num_slices 3"
    },
    "improved_upsampling": {
        "--architecture": "improved_upsampling",
    },
    
    "heavy_aug": {"--online_augmentations": "heavy"},
    "2.5d": {
        "--dataset": "SEGTHOR_2.5D",
        "--in_channels": 3,
        "_prepare_data": True,
        "_prepare_script": "prepare_data.py",
        "_prepare_args": "--slicing_mode 2.5d --num_slices 3"
    },
    "ct_windowing": {
        "--dataset": "SEGTHOR_WINDOWED",
        "_prepare_data": True,
        "_prepare_script": "prepare_data.py",
        "_prepare_args": "--preprocessing ct_windowing"
    },
    "elastic_augmentations": {
        "--online_augmentations": "elastic",
    },
}





def train_single_run(args: argparse.Namespace):
    """
    Encapsulates the entire training and validation process for a single run.
    """
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.dest.mkdir(parents=True, exist_ok=True)
    gpu = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> RUN: Using device: {device}")
    K = 5  
    
    if args.architecture == "enet_2.5d":
        net = ENet2_5D(in_dim=args.in_channels, out_dim=K)
    else:
        net = ENet(in_dim=args.in_channels, out_dim=K,
                   use_se=('se' in args.architecture),
                   use_attention=('attention' in args.architecture),
                   improved_upsampling=('improved_upsampling' in args.architecture))
    net.to(device)
    optimizer_map = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW, 'ranger': optim.Ranger}
    optimizer = optimizer_map[args.optimizer](net.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) if args.scheduler == 'cosine' else None
    loss_map = {
        'cross_entropy': CrossEntropy(),
        'dice': DiceLoss(),
        'focal': FocalLoss(),
        'tversky': TverskyLoss(),
        'compound_dice_focal': CompoundLoss(losses=[DiceLoss(), FocalLoss()], weights=[0.5, 0.5]),
        'boundary': BoundaryLoss(),
        'compound_ce_dice': CompoundLoss(losses=[CrossEntropy(), DiceLoss()], weights=[0.5, 0.5])
    }
    loss_fn = loss_map[args.loss_function]
    data_format = 'npy' if '2.5D' in args.dataset or '2.5d' in args.dataset else 'png'
    train_set = SliceDataset('train', Path("data") / args.dataset, K=K, data_format=data_format, augmentations=args.online_augmentations, debug=args.debug)
    val_set = SliceDataset('val', Path("data") / args.dataset, K=K, data_format=data_format, augmentations='none', debug=args.debug)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    
    log_loss_tra_epoch = []
    log_loss_val_epoch = []
    log_dice_val_all_epochs = torch.zeros((args.epochs, len(val_set), K))
    best_dice = 0.0
    best_weights_path = args.dest / "best_weights.pt"
    print(f">> RUN: Starting training for {args.epochs} epochs...")
    for e in range(args.epochs):
        net.train()
        epoch_train_losses = []
        tq_iter = tqdm_(enumerate(train_loader), total=len(train_loader), desc=f">> Training ({e:2d})")
        for i, data in tq_iter:
            img, gt, dist = data['images'].to(device), data['gts'].to(device), data['dist_maps'].to(device)
            optimizer.zero_grad()
            pred_logits = net(img)
            pred_probs = F.softmax(pred_logits, dim=1)
            loss = loss_fn(pred_probs, gt, dist_maps=dist)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
            tq_iter.set_postfix({"Loss": f"{np.mean(epoch_train_losses):.4f}"})
        log_loss_tra_epoch.append(np.mean(epoch_train_losses))
        if scheduler:
            scheduler.step()
        net.eval()
        epoch_val_losses = []
        epoch_dice_scores_gpu = torch.zeros((len(val_set), K), device=device)
        with torch.no_grad():
            j = 0
            val_iter = tqdm_(enumerate(val_loader), total=len(val_loader), desc=f">> Validation ({e:2d})")
            for i, data in val_iter:
                img, gt, dist = data['images'].to(device), data['gts'].to(device), data['dist_maps'].to(device)
                pred_logits = net(img)
                pred_probs = F.softmax(pred_logits, dim=1)
                loss_val = loss_fn(pred_probs, gt, dist_maps=dist)
                epoch_val_losses.append(loss_val.item())
                pred_seg = probs2one_hot(pred_probs)
                b_size = img.shape[0]
                epoch_dice_scores_gpu[j:j+b_size, :] = dice_coef(pred_seg, gt)
                j += b_size
        current_dice = epoch_dice_scores_gpu[:, 1:].mean().item()
        log_loss_val_epoch.append(np.mean(epoch_val_losses))
        log_dice_val_all_epochs[e] = epoch_dice_scores_gpu.cpu()
        print(f"Epoch {e:2d} | Val Loss: {log_loss_val_epoch[-1]:.4f} | Val Dice: {current_dice:.4f}")
        if current_dice > best_dice:
            best_dice = current_dice
            print(f"  >>> New best model saved with Dice: {best_dice:.4f}")
            torch.save(net.state_dict(), best_weights_path)
    
    if args.save_images and best_weights_path.exists():
        print("\n" + "="*20 + " GENERATING BEST EPOCH IMAGES " + "="*20)
        best_epoch_dir = args.dest / "best_epoch"
        if best_epoch_dir.exists():
            rmtree(best_epoch_dir)
        best_epoch_dir.mkdir()
        net.load_state_dict(torch.load(best_weights_path))
        net.eval()
        with torch.no_grad():
            for data in tqdm_(val_loader, desc=">> Generating images"):
                img = data['images'].to(device)
                pred_logits = net(img)
                pred_probs = F.softmax(pred_logits, dim=1)
                save_images(probs2class(pred_probs) * 63, data['stems'], best_epoch_dir)
        print("Best epoch images generated successfully.")
    
    np.save(args.dest / "loss_tra.npy", np.array(log_loss_tra_epoch))
    np.save(args.dest / "loss_val.npy", np.array(log_loss_val_epoch))
    np.save(args.dest / "dice_val.npy", log_dice_val_all_epochs.numpy())
    print(f">> RUN: Training complete. Results saved to '{args.dest}'.")





def is_run_complete(run_dir: Path, required_epochs: int) -> bool:
    """Checks if a run has completed at least the required number of epochs."""
    metric_file = run_dir / "loss_tra.npy"
    if not metric_file.exists(): return False
    try:
        completed_epochs = np.load(metric_file).shape[0]
        return completed_epochs >= required_epochs
    except Exception:
        return False

def run_command(command: str):
    """Helper to run shell commands for data preparation."""
    print(f"\nExecuting: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        raise





def main(args):
    """
    Main orchestrator for a single experiment. Sets up directories, handles
    data preparation, loops through training runs, and exits.
    """
    config_params = CONFIGURATIONS[args.config]
    exp_name = args.config
    exp_dir = Path(args.base_dest_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    if config_params.get("_prepare_data"):
        data_path = Path("data") / config_params["--dataset"]
        if not data_path.exists() or args.force_data_prep:
            prep_cmd = (f"python -O {config_params['_prepare_script']} "
                        f"--source_dir {args.source_data} --dest_dir {data_path} "
                        f"{config_params['_prepare_args']}")
            run_command(prep_cmd)
        else:
            print(f"Pre-processed data for '{exp_name}' already exists.")
    
    completed_runs_count = sum(1 for run in exp_dir.glob(f"{exp_name}_run_*") if is_run_complete(run, args.epochs))
    if completed_runs_count >= args.runs:
        print(f"Experiment '{exp_name}' already has {completed_runs_count} completed runs (requested {args.runs}). Skipping training.")
        return
    start_run_idx = completed_runs_count + 1
    end_run_idx = start_run_idx + (args.runs - completed_runs_count)
    if completed_runs_count > 0:
        print(f"Found {completed_runs_count} completed runs for '{exp_name}'. Resuming to complete a total of {args.runs} runs.")
        print(f"   Starting new runs from index {start_run_idx}.")
    
    for i in range(start_run_idx, end_run_idx):
        run_dest = exp_dir / f"{exp_name}_run_{i}"
        print(f"\n{'='*20} STARTING RUN {i}/{args.runs}: {run_dest.name} {'='*20}")
        
        run_args_dict = {
            "dest": run_dest, "epochs": args.epochs, "seed": args.base_seed + i,
            "gpu": args.gpu, "debug": args.debug, "lr": args.lr, "batch_size": args.batch_size,
            "num_workers": args.num_workers, "save_images": args.save_images
        }
        
        temp_config = {"dataset": "SEGTHOR_PROCESSED", "in_channels": 1, "scheduler": "none"}
        for k, v in CONFIGURATIONS['baseline'].items():
            temp_config[k.strip('--')] = v
        for k, v in config_params.items():
            if not k.startswith('_'): temp_config[k.strip('--')] = v
        run_args_dict.update(temp_config)
        run_args = argparse.Namespace(**run_args_dict)
        print(">> RUN ARGS:")
        pprint(vars(run_args))
        train_single_run(run_args)
    print("\nExperiment pipeline complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runner for a single segmentation experiment.")
    
    parser.add_argument('--config', type=str, required=True, choices=CONFIGURATIONS.keys(), help="The name of the experiment configuration to run.")
    parser.add_argument('--runs', type=int, default=3, help="Number of times to run the experiment (with different seeds).")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs for each run.")
    
    parser.add_argument('--base_dest_dir', type=Path, default=Path("results"), help="The base directory where experiment folders will be created.")
    parser.add_argument('--source_data', type=Path, default=Path("data/segthor_train"), help="Path to the raw training data (e.g., SegTHOR).")
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    
    parser.add_argument('--base_seed', type=int, default=42, help="Base seed for reproducibility. Each run will use base_seed + run_number.")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available.")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode (e.g., smaller dataset).")
    parser.add_argument('--force_data_prep', action='store_true', help="Force regeneration of pre-processed data even if it exists.")
    parser.add_argument('--save_images', action='store_true', help="Save validation prediction images for the best model.")
    args = parser.parse_args()
    main(args)
