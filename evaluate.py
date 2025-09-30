import argparse
import re
import os
from pathlib import Path
from collections import defaultdict
from functools import partial

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from skimage.io import imread
from skimage.transform import resize
from skimage.measure import label


from utils import tqdm_
from distorch.metrics import boundary_metrics, overlap_metrics
from compute_metrics import VolumeDataset 
import os
from multiprocessing import Pool
from functools import partial




def stitch_volumes(slice_dir: Path, dest_dir: Path, source_scan_dir: Path, num_classes: int, grp_regex: str):
    """
    Stitches 2D image slices (.png) back into 3D NIfTI volumes.
    """
    print("\n" + "="*20 + " STEP 1: STITCHING VOLUMES " + "="*20)
    dest_dir.mkdir(parents=True, exist_ok=True)
    patient_slices = defaultdict(list)
    regex = re.compile(grp_regex)
    print(f"Scanning for slices in '{slice_dir}'...")
    all_files = list(slice_dir.glob("*.png"))
    if not all_files:
        print(f"Error: No .png slices found in '{slice_dir}'. Cannot proceed.")
        return False
    for slice_file in all_files:
        match = regex.match(slice_file.name)
        if match:
            patient_id = match.groups()[0]
            patient_slices[patient_id].append(slice_file)
    if not patient_slices:
        print(f"Error: No files matched the regex '{grp_regex}'. Check the pattern.")
        return False
        
    print(f"Found slices for {len(patient_slices)} patients.")
    for patient_id, slice_files in tqdm_(patient_slices.items(), desc="Stitching Patients"):
        slice_files.sort() 
        
        original_scan_path = source_scan_dir / f"{patient_id}.nii.gz"
        if not original_scan_path.exists():
            print(f"\nWarning: Source scan not found for patient {patient_id} at '{original_scan_path}'. Skipping.")
            continue
        
        try:
            original_nii = nib.load(original_scan_path)
            original_shape = original_nii.shape
            original_affine = original_nii.affine
            original_header = original_nii.header
        except Exception as e:
            print(f"\nError loading NIfTI file {original_scan_path}: {e}")
            continue
        
        first_slice_img = imread(slice_files[0])
        slice_h, slice_w = first_slice_img.shape
        stacked_volume = np.zeros((slice_h, slice_w, len(slice_files)), dtype=np.uint8)
        
        multiplier = 63 if num_classes == 5 else (255 / (num_classes - 1) if num_classes > 1 else 1)
        for i, slice_file in enumerate(slice_files):
            slice_img = imread(slice_file)
            class_labels = np.round(slice_img / multiplier).astype(np.uint8)
            stacked_volume[:, :, i] = class_labels
        resized_volume = resize(stacked_volume,
                                output_shape=original_shape,
                                order=0, 
                                preserve_range=True,
                                anti_aliasing=False).astype(np.uint8)
        stitched_nii = nib.Nifti1Image(resized_volume, original_affine, original_header)
        stitched_nii.set_data_dtype(np.uint8)
        save_path = dest_dir / f"{patient_id}.nii.gz"
        nib.save(stitched_nii, str(save_path))
    print(f"Stitching complete. Volumes saved to '{dest_dir}'")
    return True




def clean_single_volume(nii_path: Path, *, dest_path: Path, min_size_fraction=0.05):
    """
    Removes small, disconnected components from a single 3D NIfTI segmentation file.
    """
    try:
        nii = nib.load(nii_path)
        volume = np.asarray(nii.dataobj)
        cleaned_volume = np.zeros_like(volume)
        for class_label in np.unique(volume):
            if class_label == 0: continue 
            class_mask = (volume == class_label)
            labeled_mask, num_components = label(class_mask, return_num=True, connectivity=3)
            
            if num_components > 0:
                component_sizes = np.bincount(labeled_mask.ravel())[1:] 
                min_size = component_sizes.max() * min_size_fraction
                
                for i, size in enumerate(component_sizes):
                    if size > min_size:
                        cleaned_volume[labeled_mask == (i + 1)] = class_label
        
        cleaned_nii = nib.Nifti1Image(cleaned_volume.astype(np.uint8), nii.affine, nii.header)
        nib.save(cleaned_nii, dest_path / nii_path.name)
    except Exception as e:
        print(f"Could not process {nii_path.name}: {e}")

def clean_volumes(source_dir: Path, dest_dir: Path):
    """
    Applies the cleaning function to all NIfTI files in a directory using multiprocessing.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    nii_files = list(source_dir.glob("*.nii.gz"))
    if not nii_files:
        print(f"Error: No stitched volumes (.nii.gz) found in '{source_dir}' to clean.")
        return False
    print(f"Found {len(nii_files)} NIfTI files to clean in '{source_dir}'...")
    
    p_clean_single_volume = partial(clean_single_volume, dest_path=dest_dir)
    
    num_workers = os.cpu_count() // 2  
    with Pool(processes=num_workers) as pool:
        
        list(tqdm_(pool.imap_unordered(p_clean_single_volume, nii_files), 
                  total=len(nii_files), 
                  desc="Cleaning Volumes"))
    print(f"Cleaning complete. Cleaned volumes saved to '{dest_dir}'")
    return True




def compute_final_metrics(pred_dir: Path, ref_dir: Path, output_dir: Path, metrics: list[str], K: int):
    """
    Computes 3D metrics between predicted segmentations and a reference (ground truth).
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted([p.stem.replace('.nii', '') for p in ref_dir.glob("*.nii.gz")])
    pred_stems = {p.stem.replace('.nii', '') for p in pred_dir.glob("*.nii.gz")}
    
    
    valid_stems = [s for s in stems if s in pred_stems]
    if not valid_stems:
        print(f"Error: No matching predictions found in '{pred_dir}' for ground truths in '{ref_dir}'.")
        return False
    print(f"Found {len(valid_stems)} matching cases for metric computation.")
    dataset = VolumeDataset(valid_stems, ref_dir, pred_dir, ".nii.gz", ".nii.gz", K)
    loader = DataLoader(dataset, batch_size=1, num_workers=os.cpu_count() // 2, shuffle=False)
    
    
    cmp_metrics = {m: defaultdict(lambda: torch.zeros((K,), dtype=torch.float32, device=device) * torch.nan) for m in metrics}
    
    for data in tqdm_(loader, desc="Calculating Metrics"):
        ref, pred = data['ref'].to(device), data['pred'].to(device)
        stem, voxelspacing = data['stem'][0], data['voxelspacing']
        if set(metrics).intersection(['3d_dice', '3d_jaccard']):
            s = overlap_metrics(pred.long(), ref.long(), K)
            if '3d_dice' in metrics: cmp_metrics['3d_dice'][stem] = s.Dice[0, :]
            if '3d_jaccard' in metrics: cmp_metrics['3d_jaccard'][stem] = s.Jaccard[0, :]
        for k in range(K):
            if k == 0: continue 
            pred_k, ref_k = (pred == k)[:, None, ...], (ref == k)[:, None, ...]
            if set(metrics).intersection({'3d_hd', '3d_hd95', '3d_assd'}):
                h = boundary_metrics(pred_k, ref_k, element_size=tuple(float(e) for e in voxelspacing))
                if '3d_hd' in metrics: cmp_metrics['3d_hd'][stem][k] = h.Hausdorff
                if '3d_hd95' in metrics: cmp_metrics['3d_hd95'][stem][k] = (h.Hausdorff95_1_to_2 + h.Hausdorff95_2_to_1) / 2
                if '3d_assd' in metrics: cmp_metrics['3d_assd'][stem][k] = h.AverageSymmetricSurfaceDistance
    
    print("\n--- Final 3D Metrics (Mean over cases) ---")
    header = f"{'Metric':<12} | " + " | ".join([f'Class {i}' for i in range(K)])
    print(header)
    print("-" * len(header))
    for key, values in cmp_metrics.items():
        np_dict = {k: t.cpu().numpy() for k, t in values.items()}
        stacked = np.stack(list(np_dict.values()))
        mean_scores = np.nanmean(stacked, axis=0)
        
        
        np.savez(output_dir / f'{key}_per_case.npz', **np_dict)
        
        np.save(output_dir / f'{key}_all.npy', stacked)
        mean_scores_str = " | ".join([f'{s:.4f}' for s in mean_scores])
        print(f'{key:<12} | {mean_scores_str}')
    
    print("-" * len(header))
    print(f"Metrics computation complete. Results saved in '{output_dir}'")
    return True




def main(args):
    """
    Main function to orchestrate the evaluation pipeline.
    """
    
    stitched_dir = args.output_dir / "1_stitched"
    cleaned_dir = args.output_dir / "2_cleaned"
    metrics_dir = args.output_dir / "3_metrics"
    
    
    success = stitch_volumes(
        slice_dir=args.pred_slice_dir,
        dest_dir=stitched_dir,
        source_scan_dir=args.original_scans_dir,
        num_classes=args.num_classes,
        grp_regex=args.grp_regex
    )
    if not success:
        return 
    
    final_pred_dir = stitched_dir
    if args.post_process:
        print("\n" + "="*20 + " STEP 2: CLEANING VOLUMES " + "="*20)
        success_clean = clean_volumes(source_dir=stitched_dir, dest_dir=cleaned_dir)
        if not success_clean:
            print("Warning: Cleaning step failed. Computing metrics on uncleaned volumes.")
        else:
            final_pred_dir = cleaned_dir 
    
    print("\n" + "="*20 + " STEP 3: COMPUTING METRICS " + "="*20)
    compute_final_metrics(
        pred_dir=final_pred_dir,
        ref_dir=args.gt_dir,
        output_dir=metrics_dir,
        metrics=args.metrics,
        K=args.num_classes
    )
    
    print("\nFull evaluation pipeline complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full 3D evaluation pipeline: Stitch -> Clean -> Compute Metrics.")
    
    
    parser.add_argument('--pred_slice_dir', type=Path, required=True, help="Directory with the 2D predicted slices (.png) from the model.")
    parser.add_argument('--gt_dir', type=Path, required=True, help="Directory with the 3D ground truth NIfTI files.")
    parser.add_argument('--original_scans_dir', type=Path, required=True, help="Directory with original 3D scans, used to get header/affine info for stitching.")
    parser.add_argument('--output_dir', type=Path, required=True, help="Main directory to save all evaluation outputs (stitched, cleaned, metrics).")
    
    parser.add_argument('--post_process', action='store_true', help="Enable the post-processing (cleaning) step after stitching.")
    parser.add_argument('--num_classes', '-K', type=int, default=5, help="Number of classes in the segmentation task.")
    
    
    parser.add_argument('--grp_regex', type=str, default=r"^(Patient_CT_(\d+))_\d{4}\.png$", help="Regex to extract patient ID from slice filenames.")
    
    
    parser.add_argument('--metrics', type=str, nargs='+', default=['3d_dice', '3d_hd95'],
                        choices=['3d_hd', '3d_hd95', '3d_assd', '3d_dice', '3d_jaccard'],
                        help="The 3D metrics to compute.")
    
    args = parser.parse_args()
    main(args)