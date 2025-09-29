import argparse
import json
import csv
from pathlib import Path

def main(args):
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"❌ Error: Results directory not found at '{results_dir}'")
        return
    all_results = []
    baseline_dice = None
    
    metric_files = list(results_dir.glob("**/aggregated_metrics.json"))
    if not metric_files:
        print("❌ Error: No 'aggregated_metrics.json' files found. Make sure experiments have finished.")
        return
    print(f"Found {len(metric_files)} experiment summaries to compile.")
    
    for metric_file in metric_files:
        with open(metric_file, 'r') as f:
            data = json.load(f)
            if data['experiment_name'] == 'baseline':
                baseline_dice = float(data['mean_best_val_dice'])
                break 
    
    if baseline_dice is None:
        print("⚠️ Warning: Baseline experiment not found. Cannot determine improvement.")
    
    for metric_file in metric_files:
        with open(metric_file, 'r') as f:
            data = json.load(f)
        dice_score = float(data['mean_best_val_dice'])
        
        improved = "N/A" 
        if baseline_dice is not None and data['experiment_name'] != 'baseline':
            improved = "Yes" if dice_score > baseline_dice else "No"
        all_results.append({
            'name': data['experiment_name'],
            'loss': float(data['mean_best_epoch_val_loss']),
            'loss_std': float(data['std_dev_best_epoch_val_loss']),
            'dice_score': dice_score,
            'dice_score_std': float(data['std_dev_best_val_dice']),
            'improved_over_baseline': improved
        })
    
    all_results.sort(key=lambda x: x['dice_score'], reverse=True)
    
    save_path = results_dir / "final_results_summary.csv"
    try:
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✅ Successfully compiled results into '{save_path}'")
        
        print("\n--- Experiment Summary ---")
        print(f"{'Experiment':<25} | {'Dice Score':<15} | {'Loss':<15} | {'Improved':<10}")
        print("-" * 75)
        for res in all_results:
            dice_str = f"{res['dice_score']:.4f} ± {res['dice_score_std']:.4f}"
            loss_str = f"{res['loss']:.4f} ± {res['loss_std']:.4f}"
            print(f"{res['name']:<25} | {dice_str:<15} | {loss_str:<15} | {res['improved_over_baseline']:<10}")
    except (IOError, IndexError) as e:
        print(f"❌ Error writing CSV file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compile all experiment results into a single CSV.")
    parser.add_argument('--results_dir', type=str, default="results", help="Path to the main results directory.")
    args = parser.parse_args()
    main(args)
    
