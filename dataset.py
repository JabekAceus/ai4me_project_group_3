import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def make_dataset(root, subset, data_format='png'):
    root = Path(root)
    img_path = root / subset / 'img'
    gt_path = root / subset / 'gt'
    dist_path = root / subset / 'dist'
    
    images = sorted(img_path.glob(f"*.{data_format}"))
    gts = [gt_path / f"{p.stem}.png" for p in images]
    dists = [dist_path / f"{p.stem}.npy" for p in images] if dist_path.exists() else [None] * len(images)
    
    return list(zip(images, gts, dists))

def get_augmentations(aug_type='none'):
    if aug_type == 'heavy':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=25, p=0.8),
            A.ElasticTransform(p=0.7, alpha=120, sigma=120 * 0.07, alpha_affine=120 * 0.05),
            A.GridDistortion(p=0.5),
            A.RandomBrightnessContrast(p=0.7),
            A.GaussNoise(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    elif aug_type == 'standard':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.7),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    elif aug_type == 'elastic':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.7),
            A.ElasticTransform(p=0.7, alpha=120, sigma=120 * 0.06, alpha_affine=120 * 0.04),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    return A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])

class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, K=5, data_format='png', augmentations='none', debug=False):
        self.K = K
        self.files = make_dataset(root_dir, subset, data_format)
        self.augment_pipeline = get_augmentations(augmentations)
        if debug: self.files = self.files[:10]
        print(f">> Created {subset} dataset with {len(self)} images (format: {data_format}, augs: {augmentations})...")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        img_path, gt_path, dist_path = self.files[index]
        img = np.load(img_path) if img_path.suffix == '.npy' else np.array(Image.open(img_path))
        gt = np.array(Image.open(gt_path))
        dist_map = np.load(dist_path) if dist_path and dist_path.exists() else np.zeros(img.shape[:2], dtype=np.float32)
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        augmented = self.augment_pipeline(image=img, masks=[gt, dist_map])
        img_tensor = augmented['image'].float()
        gt_tensor = augmented['masks'][0].long()
        dist_tensor = augmented['masks'][1].float().unsqueeze(0)
        gt_one_hot = torch.nn.functional.one_hot(gt_tensor // 63, num_classes=self.K).permute(2, 0, 1).float()
        
        return {"images": img_tensor, "gts": gt_one_hot, "dist_maps": dist_tensor, "stems": img_path.stem}

