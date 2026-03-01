import os
import time
from typing import Tuple
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from data import SemiSegDataset, load_csv_mappings
from utils import DiceCELoss, compute_all_metrics, compute_clip_loss
from model import CERSIncomplete
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CoTAlignmentAdapter(nn.Module):
    def __init__(self,
                 segmentation_model: nn.Module,
                 cot_bert_path: str,
                 embed_dim: int = 512,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.seg_model = segmentation_model
        self.cot_bert = AutoModel.from_pretrained(cot_bert_path, trust_remote_code=True)
        self.cot_tokenizer = AutoTokenizer.from_pretrained(cot_bert_path, trust_remote_code=True)
        self.cot_hidden_size = self.cot_bert.config.hidden_size
        if hasattr(segmentation_model, 'encoder'):
            if isinstance(segmentation_model.encoder, torch.nn.Module):
                self.img_feat_dim = 768 if 'swin' in str(type(segmentation_model.encoder)).lower() else 2048
        else:
            self.img_feat_dim = 768

        self.visual_projection = nn.Linear(self.img_feat_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(self.cot_hidden_size, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)  # log(1/0.07)

    def forward_image_features(self, images):
        feats_list = self.seg_model.encoder(images)
        last_feat = feats_list[-1]  # (B, C, H, W)
        img_embeds = last_feat.flatten(2).mean(2)
        img_embeds = self.visual_projection(img_embeds)
        img_embeds = F.normalize(img_embeds, p=2, dim=1)
        return img_embeds

    def forward_text_features(self, cots: list):
        tokens = self.cot_tokenizer(
            cots,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        outputs = self.cot_bert(**tokens)
        cls_embeds = outputs.last_hidden_state[:, 0, :]

        txt_embeds = self.text_projection(cls_embeds)
        txt_embeds = F.normalize(txt_embeds, p=2, dim=1)
        return txt_embeds

    def forward(self, images, cots):
        img_embeds = self.forward_image_features(images)
        txt_embeds = self.forward_text_features(cots)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_embeds @ txt_embeds.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def train_one_epoch_joint(adapter_model, loader, optimizer, device, loss_fn_seg, clip_weight=0.5):
    adapter_model.train()
    adapter_model.seg_model.train()

    running_loss = 0.0
    running_seg_loss = 0.0
    running_clip_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc="Training Joint"):
        imgs, masks, metas = batch
        imgs = imgs.to(device)
        masks = masks.to(device)

        cots = [m for m in metas.get('cot', '')]
        valid_indices = [i for i, c in enumerate(cots) if len(str(c)) > 1]

        optimizer.zero_grad()
        texts_for_seg = [m for m in metas.get('text', '')]
        outl, outr = adapter_model.seg_model(imgs, texts_for_seg)

        loss_seg_l = loss_fn_seg(sigmoid_probs(outl), masks.float())
        loss_seg_r = loss_fn_seg(sigmoid_probs(outr), masks.float())
        loss_seg = loss_seg_l + loss_seg_r

        loss_clip = torch.tensor(0.0, device=device)
        if len(valid_indices) > 1:
            valid_imgs = imgs[valid_indices]
            valid_cots = [cots[i] for i in valid_indices]
            logits_img, logits_txt = adapter_model(valid_imgs, valid_cots)
            loss_clip = compute_clip_loss(logits_img, logits_txt)

        total_loss = loss_seg + (clip_weight * loss_clip)

        total_loss.backward()
        optimizer.step()

        bs = imgs.shape[0]
        running_loss += total_loss.item() * bs
        running_seg_loss += loss_seg.item() * bs
        running_clip_loss += loss_clip.item() * bs
        n += bs

    return {
        "total": running_loss / max(1, n),
        "seg": running_seg_loss / max(1, n),
        "clip": running_clip_loss / max(1, n)
    }

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()

    metrics_acc_l = {
        'Dice': [],
        'pDice': [],
        'mDice': [],
        'IoU': [],
        'mIoU': []
    }

    metrics_acc_r = {
        'Dice': [],
        'pDice': [],
        'mDice': [],
        'IoU': [],
        'mIoU': []
    }

    with torch.no_grad():
        for img, mask, meta in tqdm(loader, desc="Validating"):
            img, mask = img.to(device), mask.to(device)
            text = [m for m in meta.get('text', '')]

            outl, outr = model(img, text)  # expected [B,1,H,W] (logits)
            logits_l = sigmoid_probs(outl)
            logits_r = sigmoid_probs(outr)

            meter_l = compute_all_metrics(logits_l, mask)
            meter_r = compute_all_metrics(logits_r, mask)
            for k, v in meter_l.items():
                v_np = v.detach().cpu().numpy()
                if v_np.ndim == 0:
                    metrics_acc_l[k].append(float(v_np))
                else:
                    metrics_acc_l[k].extend(v_np.reshape(-1).tolist())

            for k, v in meter_r.items():
                v_np = v.detach().cpu().numpy()
                if v_np.ndim == 0:
                    metrics_acc_r[k].append(float(v_np))
                else:
                    metrics_acc_r[k].extend(v_np.reshape(-1).tolist())

    mean_metrics_l = {k: float(np.mean(v)) if len(v) > 0 else float('nan') for k, v in metrics_acc_l.items()}
    mean_metrics_r = {k: float(np.mean(v)) if len(v) > 0 else float('nan') for k, v in metrics_acc_r.items()}
    mean_dice_l = mean_metrics_l.get('Dice', float('nan'))
    mean_dice_r = mean_metrics_r.get('Dice', float('nan'))
    print(f"Val left Results - Dice: {mean_metrics_l.get('Dice', float('nan'))}, "
          f"pDice: {mean_metrics_l.get('pDice', float('nan'))}, "
          f"IoU: {mean_metrics_l.get('IoU', float('nan'))}, "
          f"mIoU: {mean_metrics_l.get('mIoU', float('nan'))}")
    print(f"Val right Results - Dice: {mean_metrics_r.get('Dice', float('nan'))}, "
          f"pDice: {mean_metrics_r.get('pDice', float('nan'))}, "
          f"IoU: {mean_metrics_r.get('IoU', float('nan'))}, "
          f"mIoU: {mean_metrics_r.get('mIoU', float('nan'))}")
    return max(mean_dice_l, mean_dice_r)

def fit_dicece(
    seg_model: nn.Module,
    cot_bert_path: str,
    frames_dir: str,
    masks_dir: str,
    save_model_path: str = "best_joint_model.pth",
    cot_adapter_save_path: str = "best_cot_adapter.pth",
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    val_batch_size: int = 32,
    lr: float = 1e-4,
    clip_weight: float = 0.5,
    max_epochs: int = 200,
    patience: int = 30,
    dice_weight: float = 1.0,
    device: str = None,
    label_pct: float = 0.01,
    seed: int = 42,
    num_workers: int = 4,
    text_csv: str = None,
    cot_mask_csv: str = None,
    cot_no_mask_csv: str = None,
):
    adapter = CoTAlignmentAdapter(
        segmentation_model=seg_model,
        cot_bert_path=cot_bert_path,
        device=device
    ).to(device)

    # meta maps (not required)
    meta_maps = load_csv_mappings(text_csv=text_csv, cot_mask_csv=cot_mask_csv, cot_no_mask_csv=cot_no_mask_csv)

    # datasets: labeled (train) and val
    train_ds = SemiSegDataset(frames_dir, masks_dir, mode="labeled", label_pct=label_pct,
                              image_size=image_size, seed=seed, meta_maps=meta_maps, return_meta=True)
    val_ds = SemiSegDataset(frames_dir, masks_dir, mode="val", label_pct=label_pct,
                            image_size=image_size, seed=seed, meta_maps=meta_maps, return_meta=True)
    test_ds = SemiSegDataset(frames_dir, masks_dir, mode="test", label_pct=label_pct,
                             image_size=image_size, seed=seed, meta_maps=meta_maps, return_meta=True)

    print(f"Train labeled count: {len(train_ds)}, Val count: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,  # 200 epochs
        eta_min=1e-4
    )

    # use Dice + CE loss from utils (for binary we convert logits to 2 channels)
    loss_fn = DiceCELoss(lambda_dice=dice_weight, lambda_ce=1.0)

    best_val_dice = -float('inf')
    epochs_no_improve = 0
    best_epoch = -1
    final_epoch = 0

    for epoch in range(1, max_epochs + 1):
        final_epoch = epoch
        t0 = time.time()
        train_loss = train_one_epoch_joint(adapter, train_loader, optimizer, device, loss_fn, clip_weight)
        val_dice = eval_one_epoch(adapter.seg_model, val_loader, device)
        eval_one_epoch(adapter.seg_model, test_loader, device)
        t1 = time.time()

        print(f"Epoch {epoch:03d}  train_loss={train_loss['total']:.4f}  clip_loss={train_loss['clip']:.4f} "
              f"val_dice={val_dice:.4f}  time={t1-t0:.1f}s")

        # early stopping on val_dice (higher is better)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(adapter.seg_model.state_dict(), save_model_path)
            save_dict = {
                'cot_bert': adapter.cot_bert.state_dict(),
                'text_projection': adapter.text_projection.state_dict(),
                'visual_projection': adapter.visual_projection.state_dict(),
                'logit_scale': adapter.logit_scale
            }
            torch.save(save_dict, cot_adapter_save_path)
            print(f"Saved best models to {save_model_path} and {cot_adapter_save_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. Best val Dice {best_val_dice:.4f} at epoch {best_epoch}.")
            break

    scheduler.step()
    # load best weights
    if os.path.exists(save_model_path):
        model.load_state_dict(torch.load(save_model_path, map_location=device))
        eval_one_epoch(model, test_loader, device)

    return model, {"best_val_dice": best_val_dice, "best_epoch": best_epoch, "final_epoch": final_epoch}

# -------------------------
# Minimal example usage
# ------------------------
if __name__ == "__main__":
    bert_model_name = "your_path_to_BiomedVLP-CXR-BERT-specialized"
    device = 'cuda'
    backbone_type = "convnext_t"
    model = CERSIncomplete(
        bert_model_name=bert_model_name,
        backbone_type=backbone_type,
        pretrained_backbone=None,
        device=device)

    frames_dir = "your_data_root/frames"
    masks_dir = "your_data_root/masks"
    save_path = f"your_data_root/warm_up/pretrain_model.pth"
    cot_save_path = f"your_data_root/warm_up/cot_model.pth"

    trained_model, info = fit_dicece(
        seg_model=model,
        cot_bert_path="your_path_to_Bio_ClinicalBERT",
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        save_model_path=save_path,
        cot_adapter_save_path=cot_save_path,
        image_size=(224, 224),
        batch_size=16,
        val_batch_size=32,
        lr=1e-2,
        max_epochs=200,
        patience=50,
        dice_weight=1.0,
        label_pct=0.75,
        seed=42,
        num_workers=8,
        device=device,
        text_csv="your_data_root/text.csv",
        cot_mask_csv="your_data_root/reports/cot_output_gpt5_mask.csv",
        cot_no_mask_csv="your_data_root/reports/cot_output_gpt5_no_mask.csv",
    )

    print("Done. Best info:", info)
