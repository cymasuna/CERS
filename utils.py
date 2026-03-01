import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-7, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions: [BS, 1, H, W]
        # targets: [BS, 1, H, W]
        if predictions.dim() == 4:
            predictions = predictions.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        predictions = predictions.float()
        targets = targets.float()
        B = predictions.shape[0]
        pred_flat = predictions.reshape(B, -1)
        target_flat = targets.reshape(B, -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice_per_sample = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss_per_sample = 1.0 - dice_per_sample

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample


class DiceCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-7, weight: torch.Tensor = None,
                 lambda_dice: float = 1.0, lambda_ce: float = 1.0, reduction: str = 'mean'):
        super(DiceCELoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth, reduction=reduction)
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions: [BS, 1, H, W]
        # targets: [BS, 1, H, W]
        dice_loss = self.dice_loss(predictions, targets)
        if targets.dim() == 3:
            targets_bce = targets.unsqueeze(1)
        else:
            targets_bce = targets

        ce_loss = self.bce_loss(predictions, targets_bce.float())
        combined = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        return combined


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.dim() == 4:
            predictions = predictions.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        predictions = torch.clamp(predictions, 1e-7, 1 - 1e-7)
        targets = targets.float()

        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super(GeneralizedDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.dim() == 4:
            predictions = predictions.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        predictions = predictions.float()
        targets = targets.float()
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice

def compute_dice(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    """
    Per-sample Dice for binary segmentation.
    Inputs: predictions, targets: [B, 1, H, W]
    Returns: tensor shape (B,), on CPU.
    """
    # squeeze channel -> [B, H, W]
    preds = predictions.squeeze(1).float()
    tgts = targets.squeeze(1).float()

    if preds.min() < 0 or preds.max() > 1:
        probs = torch.sigmoid(preds)
    else:
        probs = preds.clamp(0, 1)

    pred_bin = (probs > 0.5).float()

    B = pred_bin.shape[0]
    pred_flat = pred_bin.view(B, -1)
    tgt_flat = tgts.view(B, -1)

    intersection = (pred_flat * tgt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1)

    both_zero = (union == 0)
    dice_per = (2.0 * intersection + smooth) / (union + smooth)
    dice_per = torch.where(both_zero, torch.ones_like(dice_per), dice_per)

    return dice_per.cpu()

def compute_pdice(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    """
    Per-sample Probabilistic Dice (use continuous predictions).
    Inputs: [B, 1, H, W]
    Returns: tensor (B,), on CPU.
    """
    preds = predictions.squeeze(1).float()
    tgts = targets.squeeze(1).float()

    if preds.min() < 0 or preds.max() > 1:
        probs = torch.sigmoid(preds)
    else:
        probs = preds.clamp(0, 1)

    B = probs.shape[0]
    p_flat = probs.view(B, -1)
    t_flat = tgts.view(B, -1)

    numerator = 2.0 * (p_flat * t_flat).sum(dim=1)
    denominator = (p_flat * p_flat).sum(dim=1) + t_flat.sum(dim=1)

    both_zero = (denominator == 0)
    pdice_per = (numerator + smooth) / (denominator + smooth)
    pdice_per = torch.where(both_zero, torch.ones_like(pdice_per), pdice_per)

    return pdice_per.cpu()

def compute_mDice_binary(predictions: torch.Tensor,
                         targets: torch.Tensor,
                         threshold: float = 0.5,
                         smooth: float = 1e-7) -> torch.Tensor:
    """
    Per-sample mean Dice over {background, foreground}.
    Inputs: [B, 1, H, W]
    Returns: tensor (B,), on CPU.
    """
    preds = predictions.squeeze(1).float()
    tgts = targets.squeeze(1).float()

    if preds.min() < 0 or preds.max() > 1:
        probs = torch.sigmoid(preds)
    else:
        probs = preds.clamp(0, 1)

    pred_bin = (probs > threshold)
    tgt_bin = (tgts > 0.5)

    B = pred_bin.shape[0]
    pred_flat = pred_bin.view(B, -1)
    tgt_flat = tgt_bin.view(B, -1)

    # Foreground dice
    inter_fg = (pred_flat & tgt_flat).sum(dim=1).float()
    sum_fg = pred_flat.sum(dim=1).float() + tgt_flat.sum(dim=1).float()
    both_zero_fg = (sum_fg == 0)
    dice_fg = (2.0 * inter_fg + smooth) / (sum_fg + smooth)
    dice_fg = torch.where(both_zero_fg, torch.ones_like(dice_fg), dice_fg)

    # Background dice (invert masks)
    inter_bg = ((~pred_flat) & (~tgt_flat)).sum(dim=1).float()
    sum_bg = (~pred_flat).sum(dim=1).float() + (~tgt_flat).sum(dim=1).float()
    both_zero_bg = (sum_bg == 0)
    dice_bg = (2.0 * inter_bg + smooth) / (sum_bg + smooth)
    dice_bg = torch.where(both_zero_bg, torch.ones_like(dice_bg), dice_bg)

    mDice_per = (dice_fg + dice_bg) / 2.0
    return mDice_per.cpu()

def compute_iou(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    """
    Per-sample IoU for foreground.
    Inputs: [B, 1, H, W]
    Returns: tensor (B,), on CPU.
    """
    preds = predictions.squeeze(1).float()
    tgts = targets.squeeze(1).float()

    if preds.min() < 0 or preds.max() > 1:
        probs = torch.sigmoid(preds)
    else:
        probs = preds.clamp(0, 1)

    pred_bin = (probs > 0.5).float()
    tgt_bin = (tgts > 0.5).float()

    B = pred_bin.shape[0]
    p_flat = pred_bin.view(B, -1)
    t_flat = tgt_bin.view(B, -1)

    inter = (p_flat * t_flat).sum(dim=1)
    union = (p_flat + t_flat - p_flat * t_flat).sum(dim=1)  # equivalent to logical or
    both_zero = (union == 0)
    iou_per = (inter + smooth) / (union + smooth)
    iou_per = torch.where(both_zero, torch.ones_like(iou_per), iou_per)

    return iou_per.cpu()

def compute_mIoU_binary(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-7) -> torch.Tensor:
    """
    Per-sample mean IoU over {background, foreground}.
    Inputs: [B, 1, H, W]
    Returns: tensor (B,), on CPU.
    """
    preds = predictions.squeeze(1).float()
    tgts = targets.squeeze(1).float()

    if preds.min() < 0 or preds.max() > 1:
        probs = torch.sigmoid(preds)
    else:
        probs = preds.clamp(0, 1)

    pred_bin = (probs > threshold)
    tgt_bin = (tgts > 0.5)

    B = pred_bin.shape[0]
    pred_flat = pred_bin.view(B, -1)
    tgt_flat = tgt_bin.view(B, -1)

    # foreground IoU
    inter_fg = (pred_flat & tgt_flat).sum(dim=1).float()
    union_fg = (pred_flat | tgt_flat).sum(dim=1).float()
    both_zero_fg = (union_fg == 0)
    iou_fg = (inter_fg + smooth) / (union_fg + smooth)
    iou_fg = torch.where(both_zero_fg, torch.ones_like(iou_fg), iou_fg)

    # background IoU
    inter_bg = ((~pred_flat) & (~tgt_flat)).sum(dim=1).float()
    union_bg = ((~pred_flat) | (~tgt_flat)).sum(dim=1).float()
    both_zero_bg = (union_bg == 0)
    iou_bg = (inter_bg + smooth) / (union_bg + smooth)
    iou_bg = torch.where(both_zero_bg, torch.ones_like(iou_bg), iou_bg)

    mIoU_per = (iou_fg + iou_bg) / 2.0
    return mIoU_per.cpu()

def compute_all_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                       spacing: tuple = (1.0, 1.0)) -> dict:
    return {
        'Dice': compute_dice(predictions, targets),
        'pDice': compute_pdice(predictions, targets),
        'mDice': compute_mDice_binary(predictions, targets),
        'IoU': compute_iou(predictions, targets),
        'mIoU': compute_mIoU_binary(predictions, targets),
    }

def compute_clip_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2