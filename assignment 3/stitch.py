import argparse
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import nibabel as nib
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Stitch 2D slices back into 3D volumes.")
    parser.add_argument('--data_folder', type=Path, required=True)
    parser.add_argument('--dest_folder', type=Path, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--grp_regex', type=str, required=True)
    parser.add_argument('--source_scan_pattern', type=str, required=True)
    args = parser.parse_args()

    args.dest_folder.mkdir(parents=True, exist_ok=True)
    patient_slices = defaultdict(list)
    regex = re.compile(args.grp_regex)

    print(f"Scanning for slices in '{args.data_folder}'...")
    all_files = list(args.data_folder.glob("*.png"))
    for slice_file in all_files:
        match = regex.match(slice_file.name)
        if match:
            patient_id = match.groups()[0]
            patient_slices[patient_id].append(slice_file)

    if not patient_slices:
        print(f"Error: No files")
        return
    print(f"{len(patient_slices)} patients.")

    for patient_id, slice_files in tqdm(patient_slices.items(), desc="Stitching Patients"):
        slice_files.sort() 

        original_scan_path = Path(args.source_scan_pattern.format(id_=patient_id))
        if not original_scan_path.exists():
            print(f"\nSource scan not found {patient_id} at '{original_scan_path}'")
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

        K = args.num_classes
        if K <= 1: multiplier = 1
        elif K == 5: multiplier = 63  
        else: multiplier = 255 / (K - 1)

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
        stitched_nii.set_data_dtype(np.uint8) # Ensure output dtype is uint8

        save_path = args.dest_folder / f"{patient_id}.nii.gz"
        nib.save(stitched_nii, str(save_path))

if __name__ == '__main__':
    main()