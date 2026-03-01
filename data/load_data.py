import os
import random
from glob import glob
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, NormalizeIntensityd, RandZoomd, Resized, ToTensord, LoadImaged, EnsureChannelFirstd
)
# -------------------------
# Utilities
# -------------------------
def _stem(path_or_name: str) -> str:
    """Return filename stem without extension."""
    if path_or_name is None:
        return ""
    return os.path.splitext(os.path.basename(str(path_or_name)))[0]


def _norm_img_field(img_field: Any) -> str:
    """
    Normalize image identifier from CSV to a stem (no extension).
    Accepts '0001', '0001.png' or 'subdir/0001.png' -> returns '0001'
    """
    if img_field is None:
        return ""
    return _stem(img_field)


def load_csv_mappings(
        text_csv: Optional[str] = None,
        cot_mask_csv: Optional[str] = None,
        cot_no_mask_csv: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Simple loader: read CSVs and return mappings keyed by image stem.
    Returns {"text": {...}, "cot_mask": {...}, "cot_no_mask": {...}}.
    - text_csv: columns [image, text]
    - cot_mask_csv: columns [image, cot] (generated with mask for labeled data)
    - cot_no_mask_csv: columns [image, cot] (generated without mask for unlabeled data)
    Missing files/values become None.
    """
    maps = {"text": {}, "cot_mask": {}, "cot_no_mask": {}}

    def _stem(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return os.path.splitext(os.path.basename(str(x)))[0]

    # Load Text
    if text_csv:
        df = pd.read_csv(text_csv)
        for _, row in df.iterrows():
            s = _stem(row.get("image"))
            if not s:
                continue
            val = row.get("text", None)
            maps["text"][s] = None if pd.isna(val) else str(val)

    # Load CoT (Mask version)
    if cot_mask_csv:
        df = pd.read_csv(cot_mask_csv)
        for _, row in df.iterrows():
            s = _stem(row.get("image"))
            if not s:
                continue
            val = row.get("cot", None)
            maps["cot_mask"][s] = None if pd.isna(val) else str(val)

    # Load CoT (No-Mask version)
    if cot_no_mask_csv:
        df = pd.read_csv(cot_no_mask_csv)
        for _, row in df.iterrows():
            s = _stem(row.get("image"))
            if not s:
                continue
            val = row.get("cot", None)
            maps["cot_no_mask"][s] = None if pd.isna(val) else str(val)

    return maps

class SemiSegDataset(Dataset):
    """
    Single dataset supporting modes: "labeled", "unlabeled", "val", "test".

    Args:
      - frames_dir, masks_dir: directories containing .png files (matching stems expected)
      - mode: "labeled"/"unlabeled"/"val"/"test"
      - label_pct: fraction of TRAIN to keep as labeled (others become unlabeled)
      - ratios: train/val/test integer ratios, default (8,1,1)
      - image_size: output H,W for both image and mask
      - transform: torchvision transform applied to PIL image -> tensor; default Resize->ToTensor->Normalize
      - seed: random seed for deterministic split
      - meta_maps: dict returned by load_csv_mappings(...)
      - return_meta: if True, __getitem__ returns (img_t, mask_t, meta_dict); else returns (img_t, mask_t)
    """

    def __init__(
            self,
            frames_dir: str,
            masks_dir: str,
            mode: str = "labeled",
            label_pct: float = 0.25,
            ratios: Tuple[int, int, int] = (8, 1, 1),
            image_size: Tuple[int, int] = (224, 224),
            seed: int = 42,
            meta_maps: Optional[Dict[str, Dict]] = None,
            return_meta: bool = False,
    ):
        assert mode in ("labeled", "unlabeled", "val", "test"), "mode must be labeled/unlabeled/val/test"
        self.mode = mode
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.label_pct = label_pct
        self.ratios = ratios
        self.image_size = image_size
        self.seed = seed
        self.return_meta = return_meta
        self.meta_maps = meta_maps
        self.transform_train = Compose([
            LoadImaged(["image", "gt"], reader='PILReader'),
            EnsureChannelFirstd(["image", "gt"]),
            RandZoomd(['image', 'gt'], min_zoom=0.95, max_zoom=1.2, mode=["bicubic", "nearest"], prob=0.1),
            Resized(["image"], spatial_size=image_size, mode='bicubic'),
            Resized(["gt"], spatial_size=image_size, mode='nearest'),
            NormalizeIntensityd(['image'], channel_wise=True),
            ToTensord(["image", "gt"]),
        ])
        self.transform_test = Compose([
            LoadImaged(["image", "gt"], reader='PILReader'),
            EnsureChannelFirstd(["image", "gt"]),
            Resized(["image"], spatial_size=image_size, mode='bicubic'),
            Resized(["gt"], spatial_size=image_size, mode='nearest'),
            NormalizeIntensityd(['image'], channel_wise=True),
            ToTensord(["image", "gt"]),
        ])

        # prepare file pairs and splits (only png)
        self._prepare_pairs_and_splits()
        self._prepare_mode_items()

    def _prepare_pairs_and_splits(self):
        frames = glob(os.path.join(self.frames_dir, "*.png"))
        masks = glob(os.path.join(self.masks_dir, "*.png"))

        f_map = {_stem(p): p for p in frames}
        m_map = {_stem(p): p for p in masks}
        pairs = []
        for k, fp in f_map.items():
            if k in m_map:
                pairs.append((fp, m_map[k]))
        pairs.sort()
        assert len(pairs) > 0, "No matching frame/mask .png pairs found."

        random.seed(self.seed)
        random.shuffle(pairs)
        total = len(pairs)
        rsum = sum(self.ratios)
        n_train = int(total * (self.ratios[0] / rsum))
        n_val = int(total * (self.ratios[1] / rsum))
        self.train_pairs = pairs[:n_train]
        self.val_pairs = pairs[n_train:n_train + n_val]
        self.test_pairs = pairs[n_train + n_val:]

        # labeled indices within train
        if len(self.train_pairs) > 0:
            n_labeled = max(1, int(len(self.train_pairs) * self.label_pct))
            self._labeled_idx = set(random.sample(range(len(self.train_pairs)), n_labeled))
        else:
            self._labeled_idx = set()

    def _prepare_mode_items(self):
        if self.mode == "labeled":
            self.items = [self.train_pairs[i] for i in range(len(self.train_pairs)) if i in self._labeled_idx]
        elif self.mode == "unlabeled":
            self.items = [self.train_pairs[i] for i in range(len(self.train_pairs)) if i not in self._labeled_idx]
        elif self.mode == "val":
            self.items = self.val_pairs
        else:  # test
            self.items = self.test_pairs

        self.n_items = len(self.items)

    def __len__(self):
        return self.n_items

    def _get_meta_for_stem(self, stem: str) -> Dict[str, Any]:
        """Return unified meta dict for a given stem."""
        meta = {
            "image": stem,
            "text": None,
            "cot": None,
        }

        if stem in self.meta_maps.get("text", {}):
            meta["text"] = self.meta_maps["text"].get(stem)

        cot_val = None
        if self.mode == "labeled":
            if stem in self.meta_maps.get("cot_mask", {}):
                cot_val = self.meta_maps["cot_mask"].get(stem)
        else:
            if stem in self.meta_maps.get("cot_no_mask", {}):
                cot_val = self.meta_maps["cot_no_mask"].get(stem)

        meta["cot"] = cot_val

        return meta

    def _transform_image(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3 and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img

    def _transform_mask(self, gt: torch.Tensor) -> torch.Tensor:
        if gt.dim() == 3 and gt.shape[0] > 1:
            gt = gt[0:1, ...]
        gt = gt.float()
        if gt.max() > 1.0:
            gt = gt / 255.0
        return gt

    def __getitem__(self, idx):
        frame_p, mask_p = self.items[idx]

        img_data = {"image": frame_p, "gt": mask_p}
        if self.mode == "labeled":
            img_data = self.transform_train(img_data)
            img_data['image'] = self._transform_image(img_data['image'])
            img_data['gt'] = self._transform_mask(img_data['gt'])
        elif self.mode == "unlabeled":
            img_data = self.transform_train(img_data)
            img_data['image'] = self._transform_image(img_data['image'])
            img_data['gt'] = torch.zeros_like(img_data['gt'])  # dummy mask
        else:
            img_data = self.transform_test(img_data)
            img_data['image'] = self._transform_image(img_data['image'])
            img_data['gt'] = self._transform_mask(img_data['gt'])
        if not self.return_meta:
            return img_data['image'], img_data['gt']

        stem = _stem(frame_p)
        meta = self._get_meta_for_stem(stem)
        return img_data['image'], img_data['gt'], meta


class MedReasonerDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            mode: str = "labeled",
            label_pct: float = 1.0,
            image_size: Tuple[int, int] = (224, 224),
            seed: int = 42,
            return_meta: bool = True,
    ):
        assert mode in ("labeled", "unlabeled", "val", "test"), "mode 必须是 labeled/unlabeled/val/test 之一"

        self.mode = mode
        self.root_dir = root_dir
        self.label_pct = label_pct
        self.image_size = image_size
        self.seed = seed
        self.return_meta = return_meta
        sub_dir = "test" if mode == "test" else "train"
        self.frames_dir = os.path.join(root_dir, sub_dir, "frames")
        self.masks_dir = os.path.join(root_dir, sub_dir, "masks")
        self.csv_path = os.path.join(root_dir, f"{sub_dir}.csv")
        self.transform_train = Compose([
            LoadImaged(["image", "gt"], reader='PILReader'),
            EnsureChannelFirstd(["image", "gt"]),
            RandZoomd(['image', 'gt'], min_zoom=0.95, max_zoom=1.2, mode=["bicubic", "nearest"], prob=0.1),
            Resized(["image"], spatial_size=image_size, mode='bicubic'),
            Resized(["gt"], spatial_size=image_size, mode='nearest'),
            NormalizeIntensityd(['image'], channel_wise=True),
            ToTensord(["image", "gt"]),
        ])
        self.transform_test = Compose([
            LoadImaged(["image", "gt"], reader='PILReader'),
            EnsureChannelFirstd(["image", "gt"]),
            Resized(["image"], spatial_size=image_size, mode='bicubic'),
            Resized(["gt"], spatial_size=image_size, mode='nearest'),
            NormalizeIntensityd(['image'], channel_wise=True),
            ToTensord(["image", "gt"]),
        ])
        self._prepare_data()

    def _prepare_data(self):
        df = pd.read_csv(self.csv_path)
        self.meta_map = df.set_index('id').to_dict(orient='index')

        all_frames = glob(os.path.join(self.frames_dir, "*.png"))
        valid_pairs = []

        for f_path in all_frames:
            stem = _stem(f_path)
            m_path = os.path.join(self.masks_dir, f"{stem}.png")
            if os.path.exists(m_path) and f"{stem}.png" in self.meta_map:
                valid_pairs.append((f_path, m_path))

        valid_pairs.sort()
        random.seed(self.seed)
        random.shuffle(valid_pairs)

        if self.mode in ("labeled", "unlabeled"):
            n_labeled = max(1, int(len(valid_pairs) * self.label_pct))
            if self.mode == "labeled":
                self.items = valid_pairs[:n_labeled]
            else:
                self.items = valid_pairs[n_labeled:]
        else:
            self.items = valid_pairs

        self.n_items = len(self.items)

    def _transform_image(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3 and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img

    def _transform_mask(self, gt: torch.Tensor) -> torch.Tensor:
        if gt.dim() == 3 and gt.shape[0] > 1:
            gt = gt[0:1, ...]
        gt = gt.float()
        if gt.max() > 1.0:
            gt = gt / 255.0
        return gt

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        frame_p, mask_p = self.items[idx]
        stem = _stem(frame_p)

        img_data = {"image": frame_p, "gt": mask_p}

        if self.mode in ("labeled", "unlabeled"):
            img_data = self.transform_train(img_data)
        else:
            img_data = self.transform_test(img_data)

        img_t = self._transform_image(img_data['image'])

        if self.mode == "unlabeled":
            mask_t = torch.zeros_like(img_data['gt'])
        else:
            mask_t = self._transform_mask(img_data['gt'])

        if not self.return_meta:
            return img_t, mask_t

        csv_row = self.meta_map.get(f"{stem}.png", {})
        meta = {
            "image": stem,
            "text": csv_row.get("problem"),
            "cot": csv_row.get("answer"),
        }

        return img_t, mask_t, meta