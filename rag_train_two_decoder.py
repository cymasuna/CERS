import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from tqdm import tqdm
from model import CERS
from faiss_manager import FaissIndexManager
from data import SemiSegDataset, load_csv_mappings
from utils import DiceCELoss, compute_all_metrics
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, default="your_data_root/frames/", help='Path to frames')
    parser.add_argument('--masks_dir', type=str, default="your_data_root/masks/", help='Path to masks')
    parser.add_argument('--text_csv', type=str, default="your_data_root/text.csv")
    parser.add_argument('--cot_mask_csv', type=str, default="your_data_root/reports/cot_output_gpt5_mask.csv")
    parser.add_argument('--cot_no_mask_csv', type=str, default="your_data_root/reports/cot_output_gpt5_no_mask.csv")
    parser.add_argument('--pretrained_ckpt', type=str, default=f"your_data_root/warm_up/pretrain_model_convnext_t.pth", help='Pretrained CERS path')
    parser.add_argument('--backbone_type', type=str, default="convnext_t")
    parser.add_argument('--bert_model_path', type=str, default="/mnt/data_ssd/ymchen/model_saved/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument('--bert_for_cot', type=str, default="/mnt/data_ssd/ymchen/model_saved/Bio_ClinicalBERT")
    parser.add_argument('--faiss_index_path', type=str, default=f"your_data_root/faiss/index/image.ex.index")
    parser.add_argument('--save_dir', type=str, default="your_data_root/checkpoints")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--ratio', type=float, default=0.75)
    parser.add_argument('--consistency', type=float, default=0.003, help='Consistency loss weight')
    parser.add_argument('--consistency_rampup', type=float, default=200)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--k_coarse', type=int, default=50)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--index_update_freq', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


class EarlyStopping:
    def __init__(self, patience=50, delta=0, path='checkpoint.pt', verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, score, model, optimizer, epoch):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model, optimizer, epoch)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model, optimizer, epoch):
        if self.verbose:
            print(f'Validation Dice increased ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dice': score
        }, self.path)


def sharpening(P):
    temp = 0.1
    T = 1 / temp
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_rag_embeddings(faiss_manager, current_cots, topk, device):
    _, I, _ = faiss_manager.search_by_text(current_cots, k=topk)
    I = I.copy()
    I[I == -1] = 0
    rag_feats_np = faiss_manager.image_embeddings[I]
    feats_rag = torch.from_numpy(rag_feats_np).to(device)
    return feats_rag

def get_rag_embeddings_by_image(faiss_manager, query_feats, topk, device):
    query_feats_np = query_feats.cpu().detach().numpy().astype('float32')
    _, I, _ = faiss_manager.search_image_by_embedding(query_feats_np, k=topk)
    I = I.copy()
    I[I == -1] = 0
    rag_feats_np = faiss_manager.image_embeddings[I]
    feats_rag = torch.from_numpy(rag_feats_np).to(device)
    return feats_rag

def get_hybrid_embeddings(faiss_manager, current_images_feats, current_cots, k_coarse, k_fine, device):
    if isinstance(current_images_feats, torch.Tensor):
        current_images_feats = current_images_feats.cpu().detach().numpy().astype('float32')
    rag_feats_np, _ = faiss_manager.search_hybrid_visual_then_semantic(
        image_query_embeddings=current_images_feats,
        text_query_strings=current_cots,
        k_coarse=k_coarse,
        k_fine=k_fine
    )
    feats_rag = torch.from_numpy(rag_feats_np).to(device)
    return feats_rag

def validate(model, loader, device, faiss_manager, args, mode=0):
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
            cot = [m for m in meta.get('cot', '')]

            feats = model(img, text, return_encoder_feats=True)
            feats_rag = get_hybrid_embeddings(faiss_manager, feats, cot, args.k_coarse, args.topk, device)

            if mode == 0:
                out_l, out_r = model(img, text, feats_rag=feats_rag)
            else:
                out_l, out_r = model(img, text)

            out_l = torch.sigmoid(out_l)
            out_r = torch.sigmoid(out_r)

            meter_l = compute_all_metrics(out_l, mask)
            meter_r = compute_all_metrics(out_r, mask)
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

def main():
    args = get_args()
    device = torch.device(args.device)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    meta_maps = load_csv_mappings(args.text_csv, args.cot_mask_csv, args.cot_no_mask_csv)
    ds_labeled = SemiSegDataset(args.frames_dir, args.masks_dir, label_pct=args.ratio, mode="labeled", meta_maps=meta_maps, return_meta=True)
    ds_unlabeled = SemiSegDataset(args.frames_dir, args.masks_dir, label_pct=args.ratio, mode="unlabeled", meta_maps=meta_maps, return_meta=True)
    ds_val = SemiSegDataset(args.frames_dir, args.masks_dir, mode="val", meta_maps=meta_maps, return_meta=True)
    ds_test = SemiSegDataset(args.frames_dir, args.masks_dir, mode="test", meta_maps=meta_maps, return_meta=True)

    loader_l = DataLoader(ds_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    loader_u = DataLoader(ds_unlabeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    loader_v = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    loader_t = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("loader_l: ", len(ds_labeled), "loader_u: ", len(ds_unlabeled), "loader_v: ", len(ds_val))

    model = CERS(
        bert_model_name=args.bert_model_path,
        backbone_type=args.backbone_type,
        device=device).to(device)

    if os.path.isfile(args.pretrained_ckpt):
        print(f"Loading pretrained: {args.pretrained_ckpt}")
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        model.load_state_dict(sd, strict=False)

    faiss_manager = FaissIndexManager(device, args.bert_for_cot, batch_size=args.batch_size)
    if os.path.exists(args.faiss_index_path + ".embeddings.npy"):
        faiss_manager.load_index(args.faiss_index_path, args.faiss_index_path.replace("image", "text"))
    else:
        print("Building initial Faiss index...")
        faiss_manager.build_index_from_datasets([(ds_labeled, "labeled")], model)
        faiss_manager.save_index(args.faiss_index_path, args.faiss_index_path.replace("image", "text"))

    criterion_labeled = DiceCELoss()
    criterion_cons = MSELoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=args.patience, path=os.path.join(args.save_dir, f'best_model.pth'))

    iter_per_epoch = max(len(loader_l), len(loader_u))
    global_step = 0

    for epoch in range(args.warm_up):
        model.train()
        pbar = tqdm(range(len(loader_l)), desc=f"Epoch {epoch + 1}/{args.epochs}")
        loader_l_iter = iter(loader_l)

        for _ in pbar:
            img_l, mask_l, meta_l = next(loader_l_iter)
            img_l, mask_l = img_l.to(device), mask_l.to(device)
            text_l = [m for m in meta_l.get('text', '')]
            cot_l = [m for m in meta_l.get('cot', '')]

            feats = model(img_l, text_l, return_encoder_feats=True)
            rag_feats_l = get_hybrid_embeddings(faiss_manager, feats, cot_l, args.k_coarse, args.topk, device)

            pred_l_l, pred_r_l = model(img_l, text_l, feats_rag=rag_feats_l)

            loss_l = criterion_labeled(torch.sigmoid(pred_l_l), mask_l.float())
            loss_r = criterion_labeled(torch.sigmoid(pred_r_l), mask_l.float())
            loss = loss_l + loss_r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            global_step += 1
        validate(model, loader_t, device, faiss_manager, args, 0, 0)

    print("==> Re-building Faiss Index with Warmed-up Teacher...")
    faiss_manager.update_index_ema([(ds_labeled, "labeled")], model, beta=0.0)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,  # 200 epochs
        eta_min=1e-6
    )

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(range(iter_per_epoch), desc=f"Epoch {epoch + 1}/{args.epochs}")
        loader_l_iter = iter(loader_l)
        loader_u_iter = iter(loader_u)

        for _ in pbar:
            try:
                img_l, mask_l, meta_l = next(loader_l_iter)
            except:
                loader_l_iter = iter(loader_l); img_l, mask_l, meta_l = next(loader_l_iter)
            try:
                img_u, mask_u, meta_u = next(loader_u_iter)
            except:
                loader_u_iter = iter(loader_u); img_u, mask_u, meta_u = next(loader_u_iter)

            img_l, mask_l = img_l.to(device), mask_l.to(device)
            img_u = img_u.to(device)

            text_l = [m for m in meta_l.get('text', '')]
            cot_l = [m for m in meta_l.get('cot', '')]
            text_u = [m for m in meta_u.get('text', '')]
            cot_u = [m for m in meta_u.get('cot', '')]

            rag_query_l = model(img_l, text_l, return_encoder_feats=True)
            rag_query_u = model(img_u, text_u, return_encoder_feats=True)
            rag_feats_l = get_hybrid_embeddings(
                faiss_manager, rag_query_l, cot_l, k_coarse=args.k_coarse, k_fine=args.topk, device=device
            )
            rag_feats_u = get_hybrid_embeddings(
                faiss_manager, rag_query_u, cot_u, k_coarse=args.k_coarse, k_fine=args.topk, device=device
            )

            pred_l_l, pred_r_l = model(img_l, text_l, feats_rag=rag_feats_l)
            pred_l_l = torch.sigmoid(pred_l_l)
            pred_r_l = torch.sigmoid(pred_r_l)

            pred_l_u, pred_r_u = model(img_u, text_u, feats_rag=rag_feats_u)
            pred_l_u = torch.sigmoid(pred_l_u)
            pred_r_u = torch.sigmoid(pred_r_u)

            loss_sup_l = criterion_labeled(pred_l_l, mask_l.float())
            loss_sup_r = criterion_labeled(pred_r_l, mask_l.float())
            loss_sup = loss_sup_l + loss_sup_r

            pseudo_l_l = sharpening(pred_l_l).detach()
            pseudo_l_u = sharpening(pred_l_u).detach()
            pseudo_r_l = sharpening(pred_r_l).detach()
            pseudo_r_u = sharpening(pred_r_u).detach()

            cons_weight = args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
            loss_cons_u_1 = criterion_cons(pred_l_u, pseudo_r_u)
            loss_cons_u_2 = criterion_cons(pred_r_u, pseudo_l_u)
            loss_cons_l_1 = criterion_cons(pred_l_l, pseudo_r_l)
            loss_cons_l_2 = criterion_cons(pred_r_l, pseudo_l_l)
            loss_cons = loss_cons_u_1 + loss_cons_u_2 + loss_cons_l_1 + loss_cons_l_2

            loss = loss_sup + cons_weight * loss_cons
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            pbar.set_postfix(loss=loss.item(), sup=loss_sup.item(), cons=cons_weight * loss_cons.item())

        scheduler.step()
        if (epoch + 1) % args.index_update_freq == 0:
            print("Updating Faiss Index (Image Embeddings)...")
            faiss_manager.update_index_ema([(ds_labeled, "labeled")], model, beta=args.ema_decay)

        val_dice = validate(model, loader_v, device, faiss_manager, args)
        validate(model, loader_t, device, faiss_manager, args)
        early_stopping(val_dice, model, optimizer, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    main()