import argparse
import os
from functools import partial
from pathlib import Path
from pprint import pprint
from functools import partial
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from distorch.metrics import boundary_metrics, overlap_metrics

class termcolors:
    RESET = '\033[39m\033[49m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    BRED = '\033[91m'
    BGREEN = '\033[92m'
    BYELLOW = '\033[93m'
    BBLUE = '\033[94m'
    BMAGENTA = '\033[95m'
    BCYAN = '\033[96m'


tc = termcolors


class TQDM(tqdm):
    @property
    def rate_(self) -> float:
        return self._ema_dn() / self._ema_dt() if self._ema_dt() else 0.


tqdm_ = partial(TQDM, dynamic_ncols=True,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')


class VolumeDataset(Dataset):
    def __init__(self, stems: list[str], ref_folder: Path, pred_folder: Path,
                 ref_extension: str, pred_extension: str, K: int,
                 /, quiet: bool = False, crop: bool = False, bg_k: int = 0) -> None:
        self.stems: list[str] = stems
        self.ref_folder: Path = ref_folder
        self.pred_folder: Path = pred_folder
        self.ref_extension: str = ref_extension
        self.pred_extension: str = pred_extension
        self.K: int = K
        self.crop: bool = crop
        self.bg_k: int = bg_k
        if not quiet:
            print(f'{tc.BLUE}>>> Initializing Volume dataset with {len(self.stems)} volumes{tc.RESET}')
            print(f'>>  {self.ref_folder}, {self.pred_folder}')
            
        assert all((self.ref_folder / (s + self.ref_extension)).exists()
                   for s in self.stems), self.ref_folder
        assert all((self.pred_folder / (s + self.pred_extension)).exists()
                   for s in self.stems), self.pred_folder
        if not quiet:
            print(f'{tc.GREEN}> All stems found in both folders{tc.RESET}')
    def __len__(self):
        return len(self.stems)
    def __getitem__(self, index: int) -> dict[str, Tensor | tuple[float, ...] | str]:
        stem: str = self.stems[index]
        ref: np.ndarray
        pred: np.ndarray
        spacing: tuple[float, ...]
        match (self.ref_extension, self.pred_extension):
            case ('.png', '.png'):
                ref = np.asarray(Image.open(self.ref_folder / (stem + self.ref_extension)))
                pred = np.asarray(Image.open(self.pred_folder / (stem + self.pred_extension)))
                spacing = (1, 1)
            case ('.nii.gz', '.nii.gz') | ('.nii', '.nii.gz') | ('.nii.gz', '.nii'):
                ref_obj = nib.load(self.ref_folder / (stem + self.ref_extension))
                spacing = ref_obj.header.get_zooms()  
                ref = np.asarray(ref_obj.dataobj, dtype=int)  
                pred_obj = nib.load(self.pred_folder / (stem + self.pred_extension))
                assert spacing == pred_obj.header.get_zooms()  
                pred = np.asarray(pred_obj.dataobj, dtype=int)  
            case _:
                raise NotImplementedError(self.ref_extension, self.pred_extension)
        if self.crop:
            r_blob = (ref != self.bg_k)
            r_coords = np.argwhere(r_blob)
            r_xyz1 = r_coords.min(axis=0)
            r_xyz2 = r_coords.max(axis=0)
            p_blob = (pred != self.bg_k)
            p_coords = np.argwhere(p_blob)
            p_xyz1 = p_coords.min(axis=0)
            p_xyz2 = p_coords.max(axis=0)
            
            xyz1 = np.minimum(r_xyz1, p_xyz1)
            xyz2 = np.maximum(r_xyz2, p_xyz2) + 1
            slices = [slice(*pair) for pair in zip(xyz1.tolist(), xyz2.tolist())]
            
            ref = ref[*slices]
            pred = pred[*slices]
            assert ref.shape == pred.shape
            assert (ref != self.bg_k).sum() == r_blob.sum()
            assert (pred != self.bg_k).sum() == p_blob.sum()

        return {'ref': torch.as_tensor(ref, dtype=torch.uint8),
                'pred': torch.as_tensor(pred, dtype=torch.uint8),
                'voxelspacing': spacing,
                'stem': stem}


def compute_metrics(loader, metrics: list[str], device, K: int,
                    ignore: list[int] | None = None) -> dict[str, dict[str, Tensor]]:
    cmp_metrics: dict[str, dict[str, Tensor]]
    
    cmp_metrics = {m: defaultdict(lambda: torch.zeros((K,),
                                                      dtype=torch.float32,
                                                      device=device) * torch.nan)
                   for m in metrics}
    desc = '>> Computing'
    tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
    with torch.no_grad():
        for j, data in tq_iter:
            ref: Tensor = data['ref'].to(device)
            pred: Tensor = data['pred'].to(device)
            voxelspacing: tuple[float, ...] = data['voxelspacing']
            stem: str = data['stem'][0]
            assert pred.shape == ref.shape
            B, *scan_shape = ref.shape
            assert B == 1, (B, ref.shape)
            if set(metrics).intersection(['3d_dice', '3d_jaccard',
                                          'pixel_accuracy', 'confusion_matrix']):
                s = overlap_metrics(pred.type(torch.int64),
                                    ref.type(torch.int64),
                                    K)
                if '3d_dice' in metrics:
                    cmp_metrics['3d_dice'][stem] = s.Dice[0, :]
                if '3d_jaccard' in metrics:
                    cmp_metrics['3d_jaccard'][stem] = s.Jaccard[0, :]
                if 'pixel_accuracy' in metrics:
                    cmp_metrics['pixel_accuracy'][stem] = s.PixelAccuracy
                if 'confusion_matrix' in metrics:
                    cmp_metrics['confusion_matrix'][stem] = s.ConfusionMatrix

            for k in range(K):
                if ignore is not None and k in ignore:
                    continue
                pred_k = (pred == k)[:, None, ...]  
                ref_k = (ref == k)[:, None, ...]  
                if set(metrics).intersection({'3d_hd', '3d_hd95', '3d_assd'}):
                    h = boundary_metrics(pred_k,
                                         ref_k,
                                         weight_by_size=False,
                                         element_size=tuple(float(e) for e in voxelspacing))
                    if '3d_hd' in metrics:
                        cmp_metrics['3d_hd'][stem][k] = h.Hausdorff
                    if '3d_hd95' in metrics:
                        cmp_metrics['3d_hd95'][stem][k] = (h.Hausdorff95_1_to_2 + h.Hausdorff95_2_to_1) / 2
                    if '3d_assd' in metrics:
                        cmp_metrics['3d_assd'][stem][k] = h.AverageSymmetricSurfaceDistance
                    if '3d_nsd' in metrics:
                        cmp_metrics['3d_nsd'][stem][k] = h.NormalizedSymmetricSurfaceDistance
            tq_iter.set_postfix({'batch_shape': list(pred.shape),
                                 'voxelspacing': [f'{float(e):.3f}' for e in voxelspacing]})
    return cmp_metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute metrics for a list of images')
    parser.add_argument('--ref_folder', type=Path, required=True)
    parser.add_argument('--pred_folder', type=Path, required=True)
    extension_choices = ['.nii.gz', '.png', '.npy', '.nii']
    parser.add_argument('--ref_extension', type=str, required=True, choices=extension_choices)
    parser.add_argument('--pred_extension', type=str, choices=extension_choices)
    parser.add_argument('--num_classes', '-K', '-C', type=int, required=True)
    parser.add_argument('--ignored_classes', type=int, nargs='*',
                        help="Classes to skip (for instance background, or any other non-predicted class).")
    parser.add_argument('--metrics', type=str, nargs='+', choices=['3d_hd', '3d_hd95', '3d_assd',
                                                                   '3d_nsd',
                                                                   '3d_dice', '3d_jaccard',
                                                                   'pixel_accuracy', 'confusion_matrix'],
                        help="The metrics to compute.")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--num_workers', type=int, default=len(os.sched_getaffinity(0)) // 2,
                        help="Number of workers for the dataloader. "
                             "Higher is not always better, depending on hardware (CPU notably).")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing metrics output, without prompt.")
    parser.add_argument('--crop', action='store_true',
                        help="Crop the scans around the objects, discarding parts that contain only"
                             " the background class.")
    parser.add_argument('--background_class', type=int, default=0)
    parser.add_argument('--chill', action='store_true',
                        help="Does not enforce that both folders have exactly the same scans inside,"
                             " keep the intersection of the two.")
    parser.add_argument('--save_folder', type=Path, default=None, help='The folder where to save the metrics')
    args = parser.parse_args()
    if not args.pred_extension:
        args.pred_extension = args.ref_extension
    pprint(args.__dict__)
    return args


def main() -> None:
    args = get_args()
    device = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
    K: int = args.num_classes
    
    stems: list[str] = sorted(map(lambda p: str(p.name).removesuffix(args.ref_extension),
                                  args.ref_folder.glob(f'*{args.ref_extension}')))
    if args.chill:
        stems = [s for s in stems if (args.pred_folder / (s + args.pred_extension)).exists()]
    dt_set = VolumeDataset(stems, args.ref_folder, args.pred_folder,
                           args.ref_extension, args.pred_extension, args.num_classes,
                           quiet=False,
                           crop=args.crop,
                           bg_k=args.background_class)
    loader = DataLoader(dt_set,
                        batch_size=1,
                        num_workers=args.num_workers,
                        shuffle=False,
                        drop_last=False)
    computed_metrics: dict[str, dict[str, Tensor]]  
    computed_metrics = compute_metrics(loader, args.metrics, device, K, ignore=args.ignored_classes)
    if args.save_folder:
        savedir: Path = Path(args.save_folder)
        savedir.mkdir(parents=True, exist_ok=True)
    print(f'>>> {args.pred_folder}')
    for key, v in computed_metrics.items():
        np_dict: dict[str, np.ndarray] = {k: t.cpu().numpy() for k, t in v.items()}
        stacked: np.ndarray = np.stack(list(np_dict.values()))
        assert (ss := stacked.shape) == (len(stems), K), (ss, key)
        print(key, stacked.mean(axis=0))
        if args.save_folder:
            dest_npz: Path = savedir / f'{key}.npz'
            assert not dest_npz.exists() or args.overwrite
            np.savez(dest_npz, **np_dict)
            dest_npy: Path = savedir / f'{key}.npy'
            assert not dest_npy.exists() or args.overwrite
            np.save(dest_npy, stacked)


if __name__ == '__main__':
    main()
