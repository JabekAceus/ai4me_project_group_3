import argparse
import random
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import shutil

import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt, zoom


from utils import tqdm_
import os

def norm_arr(img: np.ndarray) -> np.ndarray:
    """Standard min-max normalization to fit [0, 255] range."""
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    if shifted.max() > 0:
        norm = shifted / shifted.max()
    else:
        norm = shifted
    return (255 * norm).astype(np.uint8)

def ct_windowing(img: np.ndarray, window_center: int = 50, window_width: int = 400) -> np.ndarray:
    """Applies a standard soft-tissue window to a CT scan, then normalizes."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_img = np.clip(img, img_min, img_max)
    return norm_arr(windowed_img)

def slice_patient(
    id_: str, 
    dest_path: Path, 
    source_path: Path, 
    shape: tuple, 
    slicing_mode: str, 
    num_slices: int, 
    preprocessing: str, 
    test_mode: bool = False
):
    """
    Loads a patient's 3D scan and GT, preprocesses, and saves it as 2D/2.5D slices.
    """
    if not test_mode:
        ct_path = source_path / "train" / id_ / f"{id_}.nii.gz"
        gt_path = source_path / "train" / id_ / "GT.nii.gz"
    else:
        ct_path = source_path / "test" / f"{id_}.nii.gz"
        gt_path = None 
    try:
        nib_obj = nib.load(str(ct_path))
        ct: np.ndarray = np.asarray(nib_obj.dataobj)
        gt = np.asarray(nib.load(str(gt_path)).dataobj) if gt_path and gt_path.exists() else np.zeros_like(ct, dtype=np.uint8)
    except FileNotFoundError:
        print(f"⚠️ Warning: Could not find data for ID {id_}. Skipping.")
        return
    processed_ct = ct_windowing(ct) if preprocessing == 'ct_windowing' else norm_arr(ct)
    img_dest, gt_dest = dest_path / 'img', dest_path / 'gt'
    img_dest.mkdir(parents=True, exist_ok=True)
    gt_dest.mkdir(parents=True, exist_ok=True)
    dist_map_norm = None
    if preprocessing == 'distance_maps':
        dist_dest = dest_path / 'dist'
        dist_dest.mkdir(parents=True, exist_ok=True)
        
        binary_gt = gt > 0
        inside_dist = distance_transform_edt(binary_gt)
        outside_dist = distance_transform_edt(~binary_gt)
        dist_map = inside_dist - outside_dist 
        
        
        if dist_map.max() > dist_map.min():
            dist_map_norm = (dist_map.astype(np.float32) - dist_map.min()) / (dist_map.max() - dist_map.min())
        else:
            dist_map_norm = np.zeros_like(dist_map, dtype=np.float32)
        
    for idz in range(ct.shape[2]):
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            
            gt_slice = resize(gt[:, :, idz], shape, order=0, preserve_range=True).astype(np.uint8)
            imsave(gt_dest / f"{id_}_{idz:04d}.png", gt_slice * 63) 
            
            if slicing_mode == '2.5d':
                offset = num_slices // 2
                
                padded_ct = np.pad(processed_ct, ((0,0), (0,0), (offset, offset)), mode='constant')
                
                stacked_slices = np.stack([padded_ct[:, :, idz + i] for i in range(num_slices)], axis=-1)
                img_slice_resized = resize(stacked_slices, (*shape, num_slices), anti_aliasing=True, preserve_range=True).astype(np.uint8)
                np.save(img_dest / f"{id_}_{idz:04d}.npy", img_slice_resized) 
            else: 
                img_slice = resize(processed_ct[:, :, idz], shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)
                imsave(img_dest / f"{id_}_{idz:04d}.png", img_slice)
        
        if dist_map_norm is not None:
            dist_slice = resize(dist_map_norm[:, :, idz], shape, preserve_range=True).astype(np.float32)
            np.save(dist_dest / f"{id_}_{idz:04d}.npy", dist_slice)





def get_splits(src_path: Path, val_patients: int = 10, fold: int = 0):
    """
    Creates train/validation/test splits from the source directory.
    """
    train_path = src_path / 'train'
    ids = sorted([p.name for p in train_path.iterdir() if p.is_dir()])
    random.shuffle(ids) 
    val_start_idx = fold * val_patients
    val_end_idx = (fold + 1) * val_patients
    if val_end_idx > len(ids):
        raise ValueError(f"Fold {fold} is out of bounds with {val_patients} validation patients.")
    val_ids = ids[val_start_idx:val_end_idx]
    train_ids = [i for i in ids if i not in val_ids]
    test_ids = sorted([p.stem.replace('.nii', '') for p in (src_path / 'test').glob("*.nii.gz")])
    print(f"Total training patients: {len(ids)}. Split: {len(train_ids)} train, {len(val_ids)} val.")
    print(f"Total test patients: {len(test_ids)}.")
    return train_ids, val_ids, test_ids





def main(args):
    """
    Main function to run the data preparation pipeline.
    """
    src_path, dest_path = Path(args.source_dir), Path(args.dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed) 
    training_ids, validation_ids, test_ids = get_splits(src_path, args.val_patients, args.fold)
    if validation_ids:
        print("\nCopying validation NIfTI files for 3D evaluation...")
        val_gt_nifti_dir = dest_path / "val_gt_nifti"
        val_gt_nifti_dir.mkdir(exist_ok=True, parents=True)
        val_scans_nifti_dir = dest_path / "val_scans_nifti"
        val_scans_nifti_dir.mkdir(exist_ok=True, parents=True)
        for patient_id in tqdm_(validation_ids, desc="Copying val files"):
            
            gt_src_path = src_path / "train" / patient_id / "GT.nii.gz"
            if gt_src_path.exists():
                shutil.copy(gt_src_path, val_gt_nifti_dir / f"{patient_id}.nii.gz")
            
            scan_src_path = src_path / "train" / patient_id / f"{patient_id}.nii.gz"
            if scan_src_path.exists():
                shutil.copy(scan_src_path, val_scans_nifti_dir / f"{patient_id}.nii.gz")
    for mode, split_ids in zip(["train", "val", "test"], [training_ids, validation_ids, test_ids]):
        if not split_ids:
            print(f"No IDs found for {mode} set. Skipping.")
            continue
            
        dest_mode = dest_path / mode
        print(f"\nSlicing {len(split_ids)} patients for '{mode}' set...")
        
        
        pfun = partial(slice_patient, 
                       dest_path=dest_mode, 
                       source_path=src_path,
                       shape=tuple(args.shape), 
                       slicing_mode=args.slicing_mode,
                       num_slices=args.num_slices, 
                       preprocessing=args.preprocessing, 
                       test_mode=(mode == 'test'))
        
        
        with Pool(1) as p:
            list(tqdm_(p.imap(pfun, split_ids), total=len(split_ids), desc=f"Slicing {mode} set"))
    print(f"\n🎉 Data preparation complete. Processed data is in '{dest_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation pipeline for 3D medical images.")
    parser.add_argument('--source_dir', type=Path, required=True, help="Directory containing the raw NIfTI data (e.g., 'segthor_train').")
    parser.add_argument('--dest_dir', type=Path, required=True, help="Directory where the processed slices will be saved.")
    parser.add_argument('--slicing_mode', type=str, default='2d', choices=['2d', '2.5d'], help="Slicing mode: 2D saves .png, 2.5D saves .npy stacks.")
    parser.add_argument('--num_slices', type=int, default=3, help="Number of slices for 2.5D mode (must be odd).")
    parser.add_argument('--preprocessing', type=str, default='norm', choices=['norm', 'ct_windowing', 'distance_maps'], help="Preprocessing method to apply.")
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256], help="The target 2D shape (H, W) for each slice.")
    parser.add_argument('--val_patients', type=int, default=10, help="Number of patients to hold out for the validation set.")
    parser.add_argument('--fold', type=int, default=0, help="The validation fold to use for splitting.")
    parser.add_argument('--seed', type=int, default=42, help="Seed for the random train/validation split.")
    parser.add_argument('--num_workers', type=int, default=max(1, os.cpu_count() // 2), help="Number of CPU cores to use for processing.")
    args = parser.parse_args()
    if args.slicing_mode == '2.5d' and args.num_slices % 2 == 0:
        raise ValueError("Number of slices for 2.5D mode must be an odd number.")
        
    main(args)
