"""
DIQA Training Script with MoE Transformer
==========================================
Network Architecture:
  - Swin-B backbone (pretrained, timm swin_base_patch4_window7_224), 4 stages
  - Stage features projected to 512 channels and pooled to 7x7 (49 tokens)
  - Distortion classification head: pool -> FC(256) -> FC(7), BCEWithLogits loss
  - MoE routing: distortion-driven sampling (spatial: top_spatial^2 of 49, default 5x5=25; stage: top_stage of 4, default 2)
  - Dual-domain Transformer (spatial 49 tokens + 4 stages)
  - MOS regression head: pooled features -> FC(256) -> FC(1), normalized L1 loss

Hyperparameters (configurable via argparse):
  --epochs, --top_spatial, --top_stage, --n_transformer_layers
  --lambda_distortion, --lambda_mos, --lr, --batch_size, etc.
"""

import os
import argparse
import math
import numpy as np
import pandas as pd
import timm
import ipdb
from losszoo import RankHingedLoss, NormalizedL1Loss  
from losszoo  import norm_loss_with_normalization
import hashlib
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import functional as TF

from scipy.stats import pearsonr, spearmanr


# ============================================================
# Argument Parser
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(description="DIQA MoE Transformer Training")
    # Data
    parser.add_argument("--split_dir", type=str, default="/home/cbl/IQA/DIQA/splits/split_11")
    parser.add_argument("--image_dir", type=str, default="/second_data/cbl/zzh/images/")
    # Training
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    # Architecture
    parser.add_argument("--top_spatial", type=int, default=5,
                        help="Spatial tokens sampled per stage (7x7=49 -> sample top_spatial^2)")
    parser.add_argument("--top_stage", type=int, default=2,
                        help="Stages sampled from 4 stages (softmax + multinomial)")
    parser.add_argument("--n_transformer_layers", type=int, default=1,
                        help="Number of Transformer encoder layers")
    parser.add_argument("--feat_dim", type=int, default=512,
                        help="Feature dimension after projection")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads in Transformer")
    parser.add_argument("--dropout", type=float, default=0.1)
    # Loss weights
    parser.add_argument("--lambda_distortion", type=float, default=0.05,
                        help="Weight for distortion BCE loss")
    parser.add_argument("--lambda_mos", type=float, default=1.0,
                        help="Weight for MOS normalized L1 loss")
    # Patch
    parser.add_argument("--train_patch_size", type=int, default=224)
    parser.add_argument("--test_patch_size", type=int, default=224)
    parser.add_argument("--test_stride", type=int, default=32)
    parser.add_argument("--short_side", type=int, default=256)
    # Save
    parser.add_argument("--save_dir", type=str, default="./checkpoints/")
    parser.add_argument("--exp_name", type=str, default="split_1")
    return parser.parse_args()


# ============================================================
# Dataset
# ============================================================
def resize_short_side(img, short_side=512):
    """Resize image so that the short side == short_side."""
    w, h = img.size
    if h < w:
        new_h = short_side
        new_w = int(w * short_side / h)
    else:
        new_w = short_side
        new_h = int(h * short_side / w)
    return img.resize((new_w, new_h), Image.BICUBIC)


DISTORTION_COLS = [
    "noise_mean", "blur_mean", "overexposure_mean",
    "lowlight_mean", "warp_mean", "stain_mean", "occlusion_mean"
]




def resize_short_side(img, short_side):
    w, h = img.size
    if h < w:
        new_h = short_side
        new_w = int(w * short_side / h)
    else:
        new_w = short_side
        new_h = int(h * short_side / w)
    return img.resize((new_w, new_h), Image.BILINEAR)

def get_cache_dir(image_dir, short_side):
    """缓存目录放在当前工作目录下"""
    return os.path.join(os.getcwd(), f"cache_resized_s{short_side}")

def check_and_preprocess(csv_path, image_dir, short_side=512, force=False):
    """
    预处理所有图像到 cache 目录（.npy 格式）。
    - 检测 cache 是否存在
    - 读一张图验证 short_side 是否匹配
    - 不匹配则重新生成，匹配则跳过
    """
    cache_dir = get_cache_dir(image_dir, short_side)
    df = pd.read_csv(csv_path)
    filenames = df["filename"].tolist()

    def npy_path(fname):
        return os.path.join(cache_dir, fname.replace("/", "_") + ".npy")

    need_rebuild = False

    if not os.path.exists(cache_dir):
        need_rebuild = True
        print(f"[Cache] 目录不存在，开始预处理 -> {cache_dir}")
    elif force:
        need_rebuild = True
        print(f"[Cache] 强制重建缓存")
    else:
        # 检测是否有缓存文件，并验证第一张图的 short_side
        first_npy = npy_path(filenames[0])
        if not os.path.exists(first_npy):
            need_rebuild = True
            print(f"[Cache] 缓存文件缺失，重新生成")
        else:
            arr = np.load(first_npy)  # shape: (H, W, 3) uint8
            H, W = arr.shape[:2]
            actual_short = min(H, W)
            if actual_short != short_side:
                need_rebuild = True
                print(f"[Cache] short_side 不匹配 (cache={actual_short}, want={short_side})，重新生成")
            else:
                print(f"[Cache] 验证通过，跳过预处理 (short_side={short_side})")

    if need_rebuild:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[Cache] 预处理 {len(filenames)} 张图像...")
        for i, fname in enumerate(filenames):
            img_path = os.path.join(image_dir, fname)
            img = Image.open(img_path).convert("RGB")
            img = resize_short_side(img, short_side)
            arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
            np.save(npy_path(fname), arr)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(filenames)}")
        print(f"[Cache] 预处理完成 -> {cache_dir}")

    return cache_dir


class IQADataset(Dataset):
    def __init__(self, csv_path, image_dir, mode="train",
                 patch_size=224, stride=64, short_side=512,
                 num_train_patches=1):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.mode = mode
        self.patch_size = patch_size
        self.stride = stride
        self.short_side = short_side
        self.num_train_patches = num_train_patches  # 训练时每张图采多少 patch

        # 预处理并获取 cache 目录
        self.cache_dir = check_and_preprocess(csv_path, image_dir, short_side)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # 预先构建 npy 路径列表，避免每次 __getitem__ 拼字符串
        self._npy_paths = [
            os.path.join(self.cache_dir, row["filename"].replace("/", "_") + ".npy")
            for _, row in self.df.iterrows()
        ]

    def __len__(self):
        return len(self.df)

    def _load_image_tensor(self, idx):
        """从 .npy 加载图像并转为 float tensor，速度远快于 PIL decode"""
        arr = np.load(self._npy_paths[idx])          # uint8 (H,W,3)
        img_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        return img_t

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mos = torch.tensor(float(row["mos"]), dtype=torch.float32)
        distortions = torch.tensor(
            [float(row[c]) for c in DISTORTION_COLS], dtype=torch.float32
        )

        img_t = self._load_image_tensor(idx)

        if self.mode == "train":
            patches = self._random_crops(img_t)   # [num_patches, 3, ps, ps]
            if self.num_train_patches == 1:
                patches = patches[0]               # 保持原来单 patch 行为
            return patches, mos, distortions
        else:
            patches = self._sliding_window(img_t)  # [N, 3, ps, ps]
            return patches, mos, distortions

    def _random_crops(self, img_t):
        _, H, W = img_t.shape
        ps = self.patch_size
        max_h = max(H - ps, 0)
        max_w = max(W - ps, 0)
        patches = []
        for _ in range(self.num_train_patches):
            top  = np.random.randint(0, max_h + 1)
            left = np.random.randint(0, max_w + 1)
            p = img_t[:, top:top + ps, left:left + ps]
            p = self.normalize(p)
            patches.append(p)
        return torch.stack(patches)

    def _sliding_window(self, img_t):
        _, H, W = img_t.shape
        ps, s = self.patch_size, self.stride
        ys = list(range(0, max(H - ps + 1, 1), s)) or [0]
        xs = list(range(0, max(W - ps + 1, 1), s)) or [0]
        patches = []
        for y in ys:
            for x in xs:
                p = img_t[:, y:y + ps, x:x + ps]
                p = self.normalize(p)
                patches.append(p)
        return torch.stack(patches, dim=0)


def collate_test(batch):
    """Custom collate for val/test: patches have different counts per image."""
    patches_list, mos_list, dist_list = [], [], []
    for patches, mos, dist in batch:
        patches_list.append(patches)
        mos_list.append(mos)
        dist_list.append(dist)
    return patches_list, torch.stack(mos_list), torch.stack(dist_list)


# ============================================================
# Model Components
# ============================================================

class SwinBackbone(nn.Module):
    """
    Extract multi-scale features from Swin Transformer
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # features_only=True 会直接输出每个stage特征
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
             pretrained_cfg_overlay={
                'file': '/home/cbl/.cache/huggingface/hub/models--timm--swin_base_patch4_window7_224.ms_in22k_ft_in1k/snapshots/a6a1eb2321b4f556fa0fa243fb777d47679f13c9/model.safetensors'
            }
        )

    def forward(self, x):

        feats = self.backbone(x)

        f1, f2, f3, f4 = feats
        f1 = f1.permute(0,3,1,2)
        f2 = f2.permute(0,3,1,2)
        f3 = f3.permute(0,3,1,2)
        f4 = f4.permute(0,3,1,2)

        return [f1, f2, f3, f4]

class StageProjector(nn.Module):
    """Project each stage feature to 512 channels and pool to 7x7."""
    def __init__(self, in_channels_list, feat_dim=512):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, feat_dim, 1),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, features):
        # features: list of 4 tensors
        projected = []
        for i, f in enumerate(features):
            p = self.projectors[i](f)
            p = self.pool(p)   # [B, 512, 7, 7]
            projected.append(p)
        return projected  # list of 4 x [B, 512, 7, 7]



class DistortionHead(nn.Module):
    """Global pooling + FC -> 7 distortion logits (trained with BCEWithLogits). Also exposes fc1 feature for MoE routing."""
    def __init__(self, feat_dim=512, n_distortions=7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(256, n_distortions) 

    def forward(self, stage_feats):
        x = stage_feats[-1]              # [B, feat_dim, 7, 7]
        x = self.pool(x).flatten(1)      # [B, feat_dim]
        feat = self.fc1(x)               # [B, 256]  <-- MoE路由依据
        logits = self.fc2(feat)          # [B, 7]
        return logits, feat              # 同时返回中间特征



class MoERouter(nn.Module):
    """
    MoE routing driven by DistortionHead fc1 output (distortion-aware feature):
      - 所有 stage 共享同一套 spatial 采样索引 (由失真特征决定,softmax + multinomial)
      - Stage 采样也由失真特征决定 (softmax + multinomial)
    
    router_dim: DistortionHead fc1输出维度,默认256
    spatial_tokens: 7x7=49个空间位置(StageProjector 池化后)
    selected_spatial: top_spatial^2 个被选中的空间位置(默认 5x5=25)
    """
    def __init__(self, router_dim=256, feat_dim=512, n_stages=4,
                 top_spatial=6, top_stage=3):
        super().__init__()
        self.top_spatial = top_spatial
        self.top_stage = top_stage
        self.n_stages = n_stages
        self.spatial_tokens = 7 * 7       # 49个空间位置
        self.selected_spatial = top_spatial * top_spatial

        # 用失真特征预测49个空间位置的分数（所有stage共享），softmax+multinomial 采样
        self.spatial_score_head = nn.Sequential(
            nn.Linear(router_dim+self.spatial_tokens, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.spatial_tokens)
        )
       
        # 用失真特征预测4个stage的分数，softmax+multinomial 采样
        self.stage_score_head =  nn.Sequential(
            nn.Linear(router_dim+n_stages, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_stages)
        )
       
    def forward(self, stage_feats, distortion_feat):
        B = stage_feats[0].shape[0]
        
        # [B, n_stages, feat_dim, H, W]
        tmp_stage_feats = torch.stack(stage_feats, dim=1)
        
        
        feat_spatial = tmp_stage_feats.mean(dim=(1, 3, 4)).flatten(1)   
        feat_spatial = tmp_stage_feats.mean(dim=(1, 2)).flatten(1)       
        feat_stage   = tmp_stage_feats.mean(dim=(2, 3, 4))               # [B, n_stages]

        distortion_feat_spatial = torch.cat([distortion_feat, feat_spatial], dim=1)
        distortion_feat_stage   = torch.cat([distortion_feat, feat_stage],   dim=1)

        stage_scores   = self.stage_score_head(distortion_feat_stage)    # [B, n_stages]
        # 将 scores 转为概率分布，然后按概率采样 top_stage 个
        stage_probs = torch.softmax(stage_scores, dim=1)  # 归一化为概率
        stage_indices = torch.multinomial(stage_probs, self.top_stage, replacement=False)

        
        spatial_scores  = self.spatial_score_head(distortion_feat_spatial)         # [B, 49]
        spatial_probs = torch.softmax(spatial_scores, dim=1)  # 归一化为概率
        spatial_indices = torch.multinomial(spatial_probs, self.selected_spatial, replacement=False)

        
        # stage_flat: [B, n_stages, spatial_tokens, feat_dim]
        stage_flat = tmp_stage_feats.permute(0, 1, 3, 4, 2).reshape(
            B, self.n_stages, self.spatial_tokens, -1
        )  # [B, n_stages, 49, feat_dim]

        # 按 stage_indices 取选中的 stage: [B, top_stage, spatial_tokens, feat_dim]
        stage_idx_exp = stage_indices.unsqueeze(-1).unsqueeze(-1).expand(
            B, self.top_stage, self.spatial_tokens, stage_flat.shape[-1]
        )
        selected_stages = torch.gather(stage_flat, dim=1, index=stage_idx_exp)
        # [B, top_stage, spatial_tokens, feat_dim]

        # 按 spatial_indices 取选中的空间位置: [B, top_stage, k, feat_dim]
        k = self.selected_spatial
        spatial_idx_exp = spatial_indices.unsqueeze(1).unsqueeze(-1).expand(
            B, self.top_stage, k, stage_flat.shape[-1]
        )
        selected_feats = torch.gather(selected_stages, dim=2, index=spatial_idx_exp)
        # [B, top_stage, k, feat_dim]

        return selected_feats, spatial_indices, stage_indices






class DualDomainEncoderLayer(nn.Module):
    """
    One 2D Transformer layer:
      - Spatial attention (within each stage)
      - Stage attention (across stages at same spatial index)
      - Fusion + FFN
    """

    def __init__(self, feat_dim, n_heads, dropout=0.1):
        super().__init__()

        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.stage_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(dropout)
        )

        self.norm3 = nn.LayerNorm(feat_dim)

    def forward(self, x):
        """
        x: [B, S, N, D]
        """
        B, S, N, D = x.shape

        # =========================
        # Spatial Attention
        # =========================
        x_sp = x.reshape(B * S, N, D)
        sp_out, _ = self.spatial_attn(x_sp, x_sp, x_sp)
        sp_out = sp_out.reshape(B, S, N, D)

        # =========================
        # Stage Attention
        # =========================
        x_st = x.permute(0, 2, 1, 3).reshape(B * N, S, D)
        st_out, _ = self.stage_attn(x_st, x_st, x_st)
        st_out = st_out.reshape(B, N, S, D).permute(0, 2, 1, 3)

        # =========================
        # Fusion (parallel)
        # =========================
        x = x + sp_out + st_out
        x = self.norm1(x)

        # =========================
        # FFN
        # =========================
        x_ffn = self.ffn(x)
        x = self.norm3(x + x_ffn)

        return x


class DualDomainTransformer(nn.Module):
    """
    True 2D Transformer:
      - Each token attends spatially and across stages
      - Final global average pooling over (S, N)
    """

    def __init__(self, feat_dim=512, n_heads=8, n_layers=3,
                 top_stage=3, top_spatial_sq=64, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            DualDomainEncoderLayer(feat_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.mos_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        x: [B, S, N, D]
        """
        for layer in self.layers:
            x = layer(x)

        # 全 token 平均（保留通道维）
        x_out = x.mean(dim=(1, 2))  # [B, D]

        mos_pred = self.mos_head(x_out).squeeze(-1)

        return mos_pred, x_out     



class DIQAModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Backbone
        self.backbone = SwinBackbone(pretrained=True)

        # Swin-B stage channel sizes: [128, 256, 512, 1024]（共4个stage）
        in_channels = [128, 256, 512, 1024]
        self.projector = StageProjector(in_channels, feat_dim=args.feat_dim)

        # Distortion head
        self.distortion_head = DistortionHead(feat_dim=args.feat_dim, n_distortions=7)

        # MoE router
        self.router = MoERouter(
            router_dim=256,
            feat_dim=args.feat_dim,
            n_stages=4,
            top_spatial=args.top_spatial,
            top_stage=args.top_stage
        )


        # Dual-domain Transformer
        self.transformer = DualDomainTransformer(
            feat_dim=args.feat_dim,
            n_heads=args.n_heads,
            n_layers=args.n_transformer_layers,
            top_stage=args.top_stage,
            top_spatial_sq=args.top_spatial * args.top_spatial,
            dropout=args.dropout
        )
    def forward(self, x):
        """
        x: [B, 3, H, W]
        Returns: mos_pred [B], distortion_logits [B, 7]
        """
        raw_feats   = self.backbone(x)                         # list of 4
        stage_feats = self.projector(raw_feats)                # list of 4 x [B, feat_dim, 7, 7]

        # DistortionHead 返回 (logits, fc1_feat)
        dist_logits, distortion_feat = self.distortion_head(stage_feats)  # [B,7], [B,256]

        # MoE路由：由失真特征驱动，空间位置所有stage共享
        selected_feats, spatial_idx, stage_idx = self.router(
            stage_feats, distortion_feat
        )
        # selected_feats: [B, top_stage, top_spatial^2, feat_dim]

        mos_pred, _ = self.transformer(selected_feats)         # [B]
        return mos_pred, dist_logits

# ============================================================
# Training / Evaluation
# ============================================================

def train_epoch(model, loader, optimizer, args, device):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    rankloss = RankHingedLoss()
    nrml1loss = NormalizedL1Loss()
    nrmloss = norm_loss_with_normalization()
    total_loss = 0.0
    total_mos_loss = 0.0
    total_dist_loss = 0.0
    n = 0

    for patches, mos, distortions in loader:
        patches = patches.to(device)
        mos = mos.to(device)
        distortions = distortions.to(device)  # [B, 7] in [0,1]

        mos_pred, dist_logits = model(patches)
       

        loss_mos = nrmloss(mos_pred, mos) 
        loss_dist = bce(dist_logits, distortions)
        loss = args.lambda_mos * loss_mos + args.lambda_distortion * loss_dist

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        B = patches.shape[0]
        total_loss += loss.item() * B
        total_mos_loss += loss_mos.item() * B
        total_dist_loss += loss_dist.item() * B
        n += B

    return total_loss / n, total_mos_loss / n, total_dist_loss / n


@torch.no_grad()
def evaluate(model, loader, device, mode="val"):
    """
    For val/test, each sample is a set of patches.
    Predict per-patch MOS, average for image-level MOS.
    """
    model.eval()
    pred_mos_list = []
    gt_mos_list = []

    for patches_list, mos_batch, distortions_batch in loader:
        # patches_list: list of [N_i, 3, ps, ps]
        mos_batch = mos_batch.to(device)
        for i, patches in enumerate(patches_list):
            patches = patches.to(device)
            # Run in sub-batches to save memory
            sub_bs = 32
            preds = []
            for start in range(0, patches.shape[0], sub_bs):
                sub = patches[start:start + sub_bs]
                mos_pred, _ = model(sub)
                preds.append(mos_pred)
            avg_pred = torch.cat(preds).mean().item()
            pred_mos_list.append(avg_pred)
            gt_mos_list.append(mos_batch[i].item())

    pred = np.array(pred_mos_list)
    gt = np.array(gt_mos_list)
    plcc, _ = pearsonr(pred, gt)
    srcc, _ = spearmanr(pred, gt)
    mae = np.mean(np.abs(pred - gt))
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    return {"PLCC": plcc, "SRCC": srcc, "MAE": mae, "RMSE": rmse}


# ============================================================
# Main
# ============================================================

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.exp_name}_best.pth")

    # Datasets
    train_csv = os.path.join(args.split_dir, "train.csv")
    val_csv   = os.path.join(args.split_dir, "val.csv")
    test_csv  = os.path.join(args.split_dir, "test.csv")

    train_dataset = IQADataset(train_csv, args.image_dir, mode="train",
                               patch_size=args.train_patch_size,
                               short_side=args.short_side)
    val_dataset   = IQADataset(val_csv,   args.image_dir, mode="val",
                               patch_size=args.test_patch_size,
                               stride=args.test_stride,
                               short_side=args.short_side)
    test_dataset  = IQADataset(test_csv,  args.image_dir, mode="test",
                               patch_size=args.test_patch_size,
                               stride=args.test_stride,
                               short_side=args.short_side)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_test, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_test, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Model
    model = DIQAModel(args).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    # Optimizer with different LR for backbone vs head
    backbone_params = list(model.module.backbone.parameters()) if hasattr(model, 'module') else list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not any(n.startswith(f'{"module." if hasattr(model, "module") else ""}backbone') for _ in [0])]
   
    def get_param_groups(m):
        backbone = m.module.backbone if hasattr(m, 'module') else m.backbone
        backbone_ids = set(id(p) for p in backbone.parameters())
        backbone_pg = [p for p in m.parameters() if id(p) in backbone_ids]
        other_pg = [p for p in m.parameters() if id(p) not in backbone_ids]
        return backbone_pg, other_pg

    backbone_pg, other_pg = get_param_groups(model)
    optimizer = torch.optim.AdamW([
        {"params": backbone_pg, "lr": args.lr * 0.1},
        {"params": other_pg, "lr": args.lr}
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_val_plcc = -1.0
    best_epoch = 0
    log_file = os.path.join(args.save_dir, f"{args.exp_name}_log.csv")

    with open(log_file, "w") as f:
        f.write("epoch,train_loss,train_mos,train_dist,val_plcc,val_srcc,val_mae,val_rmse\n")

    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mos, train_dist = train_epoch(model, train_loader, optimizer, args, device)
        val_metrics = evaluate(model, val_loader, device, mode="val")
        scheduler.step()

        log_line = (f"Epoch [{epoch:3d}/{args.epochs}] "
                    f"Loss={train_loss:.4f} (MOS={train_mos:.4f}, Dist={train_dist:.4f}) | "
                    f"Val PLCC={val_metrics['PLCC']:.4f} SRCC={val_metrics['SRCC']:.4f} "
                    f"MAE={val_metrics['MAE']:.4f} RMSE={val_metrics['RMSE']:.4f}")
        print(log_line)

        with open(log_file, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_mos:.6f},{train_dist:.6f},"
                    f"{val_metrics['PLCC']:.6f},{val_metrics['SRCC']:.6f},"
                    f"{val_metrics['MAE']:.6f},{val_metrics['RMSE']:.6f}\n")

        # Save best model based on PLCC
        if val_metrics["PLCC"] > best_val_plcc:
            best_val_plcc = val_metrics["PLCC"]
            best_epoch = epoch
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                "epoch": epoch,
                "state_dict": state,
                "val_metrics": val_metrics,
                "args": vars(args)
            }, save_path)
            print(f"  --> Best model saved (PLCC={best_val_plcc:.4f})")

    print(f"\nBest epoch: {best_epoch}, Val PLCC: {best_val_plcc:.4f}")

    # Test with best model
    print("\n" + "="*60)
    print("Testing with best model...")
    checkpoint = torch.load(save_path, map_location=device)
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate(model, test_loader, device, mode="test")
    print(f"Test Results: PLCC={test_metrics['PLCC']:.4f} SRCC={test_metrics['SRCC']:.4f} "
          f"MAE={test_metrics['MAE']:.4f} RMSE={test_metrics['RMSE']:.4f}")

    with open(log_file, "a") as f:
        f.write(f"\nTest,,,,"
                f"{test_metrics['PLCC']:.6f},{test_metrics['SRCC']:.6f},"
                f"{test_metrics['MAE']:.6f},{test_metrics['RMSE']:.6f}\n")

    return test_metrics


if __name__ == "__main__":
    main()
