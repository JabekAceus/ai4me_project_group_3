import argparse
import csv
import json
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from evaluate import main as run_3d_evaluation
from runner import CONFIGURATIONS


def aggregate_experiment_results(exp_dir: Path):
    """
    Aggregates results from all completed runs within an experiment directory.
    Calculates mean/std dev for the best validation Dice and its corresponding loss.
    """
    run_dirs = sorted([d for d in exp_dir.glob(f"{exp_dir.name}_run_*") if d.is_dir()])
    if not run_dirs:
        print(f"  [!] No run folders found in '{exp_dir}' to aggregate.")
        return
    best_dice_scores, best_epoch_losses = [], []
    for run_dir in run_dirs:
        try:
            loss_val = np.load(run_dir / "loss_val.npy")
            dice_val = np.load(run_dir / "dice_val.npy")  
            
            mean_dice_per_epoch = dice_val[:, :, 1:].mean(axis=(1, 2))
            best_epoch_idx = np.argmax(mean_dice_per_epoch)
            best_dice_scores.append(mean_dice_per_epoch[best_epoch_idx])
            best_epoch_losses.append(loss_val[best_epoch_idx]) 
        except FileNotFoundError:
            print(f"  [!] Metric files not found in {run_dir}. Skipping for aggregation.")
            continue
    if not best_dice_scores:
        print(f"  [!] No valid metric files found in {exp_dir} to aggregate.")
        return
    metrics = {
        "experiment_name": exp_dir.name,
        "num_successful_runs": len(best_dice_scores),
        "mean_best_val_dice": f"{np.mean(best_dice_scores):.4f}",
        "std_dev_best_val_dice": f"{np.std(best_dice_scores):.4f}",
        "mean_best_epoch_val_loss": f"{np.mean(best_epoch_losses):.4f}",
        "std_dev_best_epoch_val_loss": f"{np.std(best_epoch_losses):.4f}",
        "individual_best_dice_scores": [f"{s:.4f}" for s in best_dice_scores]
    }
    metrics_path = exp_dir / "aggregated_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"  [+] Aggregated metrics saved to '{metrics_path}'")





def generate_plots(results_dir: Path):
    """
    Scans the results directory, finds all experiment folders, and generates
    learning curve plots with mean and standard deviation bands.
    """
    print("\n" + "="*20 + " GENERATING LEARNING CURVE PLOTS " + "="*20)
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    for exp_dir in exp_dirs:
        print(f"--> Plotting for experiment: {exp_dir.name}")
        run_dirs = sorted([d for d in exp_dir.glob(f"{exp_dir.name}_run_*")])
        if not run_dirs:
            continue
        all_loss_tra, all_loss_val, all_dice_val = [], [], []
        min_epochs = float('inf')
        for run_dir in run_dirs:
             try:
                loss_tra = np.load(run_dir / "loss_tra.npy")
                min_epochs = min(min_epochs, len(loss_tra))
             except (FileNotFoundError, ValueError):
                continue
        if min_epochs == float('inf') or min_epochs == 0:
            print(f"  [!] No valid metric files found to plot for {exp_dir.name}.")
            continue
        for run_dir in run_dirs:
            try:
                loss_tra = np.load(run_dir / "loss_tra.npy")
                loss_val = np.load(run_dir / "loss_val.npy")
                dice_val = np.load(run_dir / "dice_val.npy")
                mean_dice_per_epoch = dice_val[:, :, 1:].mean(axis=(1, 2))
                all_loss_tra.append(loss_tra[:min_epochs])
                all_loss_val.append(loss_val[:min_epochs])
                all_dice_val.append(mean_dice_per_epoch[:min_epochs])
            except (FileNotFoundError, ValueError):
                continue
        if not all_dice_val:
            continue
        loss_tra_arr = np.array(all_loss_tra)
        loss_val_arr = np.array(all_loss_val)
        dice_val_arr = np.array(all_dice_val)
        mean_loss_tra, std_loss_tra = loss_tra_arr.mean(axis=0), loss_tra_arr.std(axis=0)
        mean_loss_val, std_loss_val = loss_val_arr.mean(axis=0), loss_val_arr.std(axis=0)
        mean_dice_val, std_dice_val = dice_val_arr.mean(axis=0), dice_val_arr.std(axis=0)
        epochs = np.arange(1, min_epochs + 1)
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Learning Curves for "{exp_dir.name}" ({len(run_dirs)} runs)', fontsize=16)
        ax1.plot(epochs, mean_loss_tra, 'b-', label='Mean Training Loss')
        ax1.fill_between(epochs, mean_loss_tra - std_loss_tra, mean_loss_tra + std_loss_tra, color='b', alpha=0.2)
        ax1.plot(epochs, mean_loss_val, 'r-', label='Mean Validation Loss')
        ax1.fill_between(epochs, mean_loss_val - std_loss_val, mean_loss_val + std_loss_val, color='r', alpha=0.2)
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax2.plot(epochs, mean_dice_val, 'g-', label='Mean Validation Dice')
        ax2.fill_between(epochs, mean_dice_val - std_dice_val, mean_dice_val + std_dice_val, color='g', alpha=0.2)
        best_epoch = np.argmax(mean_dice_val)
        best_score = np.max(mean_dice_val)
        ax2.axvline(x=best_epoch + 1, color='k', linestyle='--', label=f'Best Epoch: {best_epoch + 1} ({best_score:.4f})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Coefficient')
        ax2.set_title('Validation Dice Coefficient')
        ax2.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = exp_dir / "learning_curves.png"
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"  [+] Plot saved to {plot_path}")





def compile_csv_report(results_dir: Path):
    """
    Gathers all aggregated_metrics.json files, computes improvement over
    baseline, ranks them, and saves a final summary CSV file.
    """
    print("\n" + "="*20 + " COMPILING FINAL REPORT " + "="*20)
    metric_files = list(results_dir.glob("**/aggregated_metrics.json"))
    if not metric_files:
        print("Error: No 'aggregated_metrics.json' files found. Cannot compile report.")
        return
    all_results = []
    baseline_dice = None
    for metric_file in metric_files:
        with open(metric_file, 'r') as f:
            data = json.load(f)
            if data['experiment_name'] == 'baseline':
                baseline_dice = float(data['mean_best_val_dice'])
                break
    if baseline_dice is None:
        print("Warning: Baseline experiment not found. Cannot calculate improvement.")
    for metric_file in metric_files:
        with open(metric_file, 'r') as f:
            data = json.load(f)
        dice_score_2d = float(data['mean_best_val_dice'])
        improvement = "N/A"
        if baseline_dice is not None and data['experiment_name'] != 'baseline':
            diff = dice_score_2d - baseline_dice
            improvement = f"{diff:+.4f}"
        
        mean_3d_dice = data.get('mean_3d_dice', 'N/A')
        std_3d_dice = data.get('std_3d_dice', 'N/A')
        mean_3d_hd95 = data.get('mean_3d_hd95', 'N/A')
        std_3d_hd95 = data.get('std_3d_hd95', 'N/A')
        all_results.append({
            'Experiment Name': data['experiment_name'],
            'Mean Val Dice (2D)': dice_score_2d,
            'Std Dev Dice (2D)': float(data['std_dev_best_val_dice']),
            'Mean 3D Dice': mean_3d_dice,
            'Std Dev 3D Dice': std_3d_dice,
            'Mean 3D HD95': mean_3d_hd95,
            'Std Dev 3D HD95': std_3d_hd95,
            'Improvement vs Baseline (2D)': improvement,
            'Successful Runs': int(data['num_successful_runs'])
        })
    
    all_results.sort(key=lambda x: x['Mean Val Dice (2D)'], reverse=True)
    save_path = results_dir / "final_experiment_summary.csv"
    try:
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"✅ Successfully compiled and ranked results into '{save_path}'")
        print("\n--- Experiment Summary (Ranked by Mean 2D Val Dice) ---")
        header = f"{'Rank':<5} | {'Experiment':<25} | {'Mean 2D Dice':<22} | {'Mean 3D Dice':<22} | {'Mean 3D HD95':<22}"
        print(header)
        print("-" * len(header))
        for i, res in enumerate(all_results):
            dice2d_str = f"{res['Mean Val Dice (2D)']:.4f} ± {res['Std Dev Dice (2D)']:.4f}"
            dice3d_str = f"{res['Mean 3D Dice']} ± {res['Std Dev 3D Dice']}" if res['Mean 3D Dice'] != 'N/A' else "N/A"
            hd95_str = f"{res['Mean 3D HD95']} ± {res['Std Dev 3D HD95']}" if res['Mean 3D HD95'] != 'N/A' else "N/A"
            print(f"{i+1:<5} | {res['Experiment Name']:<25} | {dice2d_str:<22} | {dice3d_str:<22} | {hd95_str:<22}")
    except (IOError, IndexError) as e:
        print(f"Error writing CSV file: {e}")





def main():
    parser = argparse.ArgumentParser(description="Master orchestrator for all segmentation experiments.")
    parser.add_argument('--configs', type=str, nargs='*', help="Optional: list of specific configs to run. If not provided, all are run.")
    parser.add_argument('--runs', type=int, default=3, help="Number of runs for each experiment.")
    parser.add_argument('--epochs', type=int, default=40, help="Number of epochs for each run.")
    parser.add_argument('--base_dest_dir', type=Path, default=Path("results"), help="Base directory for all results.")
    parser.add_argument('--source_data', type=Path, default=Path("data/segthor_train"), help="Path to the raw training data.")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available.")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_images', action='store_true', help="Save validation prediction images for the best model.")
    parser.add_argument('--eval_only', action='store_true', help="Skip training and run only aggregation, evaluation, and reporting.")
    args = parser.parse_args()
    if not args.eval_only:
        
        configs_to_run = args.configs if args.configs else CONFIGURATIONS.keys()
        for config_name in configs_to_run:
            if config_name not in CONFIGURATIONS:
                print(f"Warning: Config '{config_name}' not found. Skipping.")
                continue
            print("\n" + "#"*30 + f" STARTING EXPERIMENT: {config_name} " + "#"*30)
            try:
                command = [
                    "python", "-O", "runner.py", "--config", config_name,
                    "--runs", str(args.runs), "--epochs", str(args.epochs),
                    "--base_dest_dir", str(args.base_dest_dir),
                    "--source_data", str(args.source_data), "--lr", str(args.lr),
                    "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers)
                ]
                if args.gpu: command.append("--gpu")
                if args.debug: command.append("--debug")
                if args.save_images: command.append("--save_images")
                print(f"Executing command: {' '.join(command)}")
                subprocess.run(command, check=True)
            except Exception as e:
                print(f"An error occurred during experiment '{config_name}': {e}")
                print("   Continuing to the next experiment.")
                continue
    else:
        print("\n" + "="*20 + " --eval_only flag detected, SKIPPING TRAINING. " + "="*20)
        
        configs_to_run = args.configs if args.configs else CONFIGURATIONS.keys()
    
    print("\n" + "="*20 + " AGGREGATING ALL EXPERIMENT RESULTS " + "="*20)
    for config_name in configs_to_run:
        print(f"--> Aggregating for experiment: {config_name}")
        exp_dir = args.base_dest_dir / config_name
        if exp_dir.is_dir():
            aggregate_experiment_results(exp_dir)
        else:
            print(f"  [!] Directory not found for {config_name}. Skipping aggregation.")
    
    print("\n" + "="*20 + " RUNNING ALL 3D EVALUATIONS " + "="*20)
    for config_name in configs_to_run:
        print("\n" + "#"*15 + f" AUTO 3D EVALUATION FOR: {config_name} " + "#"*15)
        exp_dir = args.base_dest_dir / config_name
        run_dirs = sorted([d for d in exp_dir.glob(f"{config_name}_run_*") if d.is_dir()])
        if not run_dirs:
            print(f"  [!] No run folders found for {config_name}. Skipping 3D evaluation.")
            continue
        all_runs_3d_dice = []
        all_runs_3d_hd95 = []
        for i, run_dir in enumerate(run_dirs):
            print(f"--> Evaluating run {i+1}/{len(run_dirs)} for {config_name}...")
            pred_slice_dir = run_dir / "best_epoch"
            if not pred_slice_dir.exists():
                print(f"  [!] 'best_epoch' folder not found for run {run_dir.name}. Skipping this run.")
                continue
            dataset_name = CONFIGURATIONS[config_name].get('--dataset', 'SEGTHOR_PROCESSED')
            eval_args = argparse.Namespace(
                pred_slice_dir=pred_slice_dir,
                gt_dir=Path("data") / dataset_name / "val_gt_nifti",
                original_scans_dir=Path("data") / dataset_name / "val_scans_nifti",
                output_dir=run_dir / "final_3d_metrics",
                post_process=True, num_classes=5,
                grp_regex=r"^(Patient(?:_CT)?_\d+)_\d{4}\.png$",
                metrics=['3d_dice', '3d_hd95']
            )
            try:
                run_3d_evaluation(eval_args)
                
                metrics_dir = run_dir / "final_3d_metrics" / "3_metrics"
                if metrics_dir.exists():
                    dice_path = metrics_dir / "3d_dice_all.npy"
                    if dice_path.exists():
                        dice_data = np.load(dice_path)
                        mean_fg_dice = np.nanmean(dice_data, axis=0)[1:].mean()
                        all_runs_3d_dice.append(mean_fg_dice)
                    hd95_path = metrics_dir / "3d_hd95_all.npy"
                    if hd95_path.exists():
                        hd95_data = np.load(hd95_path)
                        mean_fg_hd95 = np.nanmean(hd95_data, axis=0)[1:].mean()
                        all_runs_3d_hd95.append(mean_fg_hd95)
            except Exception as e:
                print(f"  [!] An error occurred during 3D evaluation for {run_dir.name}: {e}")
        
        agg_metrics_path = exp_dir / "aggregated_metrics.json"
        if not agg_metrics_path.exists():
            print(f"  [!] Cannot find '{agg_metrics_path.name}' to update with 3D metrics.")
            continue
        try:
            with open(agg_metrics_path, 'r+') as f:
                metrics_data = json.load(f)
                if all_runs_3d_dice:
                    metrics_data['mean_3d_dice'] = f"{np.mean(all_runs_3d_dice):.4f}"
                    metrics_data['std_3d_dice'] = f"{np.std(all_runs_3d_dice):.4f}"
                else:
                    metrics_data['mean_3d_dice'] = "N/A"
                    metrics_data['std_3d_dice'] = "N/A"
                if all_runs_3d_hd95:
                    metrics_data['mean_3d_hd95'] = f"{np.mean(all_runs_3d_hd95):.4f}"
                    metrics_data['std_3d_hd95'] = f"{np.std(all_runs_3d_hd95):.4f}"
                else:
                    metrics_data['mean_3d_hd95'] = "N/A"
                    metrics_data['std_3d_hd95'] = "N/A"
                f.seek(0)
                json.dump(metrics_data, f, indent=4)
                f.truncate()
            print(f"  [+] Updated '{agg_metrics_path.name}' with aggregated 3D metrics from {len(all_runs_3d_dice)} evaluated runs.")
        except Exception as json_e:
            print(f"  [!] Could not update JSON with aggregated 3D metrics for {config_name}: {json_e}")
    
    generate_plots(args.base_dest_dir)
    compile_csv_report(args.base_dest_dir)
    print("\nAll experiments and reporting are complete!")


if __name__ == '__main__':
    main()
