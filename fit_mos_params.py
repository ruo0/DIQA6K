# -*- coding: utf-8 -*-
"""
fit_mos_params.py — 拟合模型原始输出 -> MOS 的 5 参数映射（VQEG/ITU 标准单调拟合）

背景：
    训练使用归一化损失（norm-in-norm / normalized L1），模型原始输出是无标定的
    （范围有正有负，与 1-5 的 MOS 不一致）。虽然 PLCC/SRCC 很高，但绝对值不可直接读。
    本脚本在测试集上把 (原始预测, 真实 MOS) 做 5 参数非线性拟合：

        f(x) = b1 * (0.5 - 1 / (1 + exp(b2 * (x - b3)))) + b4 * x + b5

    得到 b1..b5 后保存为 JSON，供 demo.py 加载，把单图预测映射回 MOS 尺度。

用法：
    # 方式 A：直接用已有预测 csv（需含 pred_mos / gt_mos 两列）
    python fit_mos_params.py --csv test_predictions.csv --out mos_fit_params.json

    # 方式 B：给定 checkpoint + 测试集，自动推理并拟合（推荐，保证数据与模型一致）
    python fit_mos_params.py \
        --ckpt ./checkpoints-logit-swin-all-best/split_5_best.pth \
        --split_dir ./splits/split_5 \
        --image_dir ./DIQA-6K/images \
        --out mos_fit_params.json

输出：mos_fit_params.json（含 b1..b5、样本数、拟合前后 PLCC/SRCC）
"""
import os
import json
import argparse
import functools
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "checkpoints-logit-swin-all-best", "split_5_best.pth")


# ---------------------------------------------------------------
# 5 参数单调拟合（VQEG 视频质量评估标准形式）
# ---------------------------------------------------------------
def logistic5(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def fit_logistic5(pred, gt):
    """最小二乘拟合 5 参数，返回 (popt: dict, 拟合后预测)"""
    p0 = [1.0, 1.0, 0.0, 0.2, 0.0]
    bounds = ([0.0, 1e-4, -50.0, -1.0, -10.0],
              [10.0, 10.0, 50.0, 1.0, 10.0])
    try:
        popt, _ = curve_fit(logistic5, pred, gt, p0=p0, bounds=bounds,
                            maxfev=20000)
    except RuntimeError:
        # 退化保护：退化为线性拟合
        A = np.vstack([pred, np.ones_like(pred)]).T
        b4, b5 = np.linalg.lstsq(A, gt, rcond=None)[0]
        popt = [0.0, 1.0, 0.0, float(b4), float(b5)]
    fit = logistic5(pred, *popt)
    return {"b1": float(popt[0]), "b2": float(popt[1]),
            "b3": float(popt[2]), "b4": float(popt[3]),
            "b5": float(popt[4])}, fit


def report(pred, gt, tag=""):
    plcc, _ = pearsonr(pred, gt)
    srcc, _ = spearmanr(pred, gt)
    print(f"[{tag:14s}] PLCC={plcc:.4f}  SRCC={srcc:.4f}")
    return plcc, srcc


# ---------------------------------------------------------------
# 方式 A：从已有 csv 拟合
# ---------------------------------------------------------------
def fit_from_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    pred_col = next((c for c in df.columns if c.lower() in ("pred_mos", "pred", "mos_pred")), None)
    gt_col = next((c for c in df.columns if c.lower() in ("gt_mos", "gt", "mos", "label")), None)
    if pred_col is None or gt_col is None:
        raise ValueError(f"csv 中找不到预测/MOS 列，现有列: {list(df.columns)}")
    pred = df[pred_col].astype(float).values
    gt = df[gt_col].astype(float).values
    print(f"loaded {len(pred)} pairs from {csv_path} (pred={pred_col}, gt={gt_col})")
    return pred, gt


# ---------------------------------------------------------------
# 方式 B：用 checkpoint 推理测试集再拟合
# ---------------------------------------------------------------
def load_model(ckpt_path, device):
    import importlib.util
    import sys
    import types
    from types import SimpleNamespace
    import torch

    # 训练脚本第 23 行 import ipdb 但未使用；本地环境无 ipdb，注入空 stub
    if "ipdb" not in sys.modules:
        _stub = types.ModuleType("ipdb")
        _stub.set_trace = lambda *a, **k: None
        sys.modules["ipdb"] = _stub

    spec = importlib.util.spec_from_file_location(
        "diqa_train", os.path.join(HERE, "logitandpool-moe-swin.py"))
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    # SwinBackbone 的 pretrained_cfg_overlay 硬编码了训练机路径，这里去掉，
    # 权重完全由 checkpoint 的 state_dict 提供
    import timm
    _orig_create = timm.create_model

    def _safe_create(*a, **k):
        k.pop("pretrained_cfg_overlay", None)
        k["pretrained"] = False
        return _orig_create(*a, **k)

    timm.create_model = _safe_create

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = SimpleNamespace(**ckpt["args"])
    model = train_mod.DIQAModel(args).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, train_mod


def infer_split(model, train_mod, split_dir, image_dir, device,
                patch_size=224, stride=32, short_side=256):
    import torch
    from PIL import Image
    from torchvision import transforms

    df = train_mod.pd.read_csv(os.path.join(split_dir, "test.csv"))
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    preds, gts = [], []
    for _, row in df.iterrows():
        img = Image.open(os.path.join(image_dir, row["filename"])).convert("RGB")
        img = train_mod.resize_short_side(img, short_side)
        arr = np.asarray(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        _, H, W = t.shape
        ys = list(range(0, max(H - patch_size + 1, 1), stride)) or [0]
        xs = list(range(0, max(W - patch_size + 1, 1), stride)) or [0]
        patches = torch.stack(
            [norm(t[:, y:y + patch_size, x:x + patch_size])
             for y in ys for x in xs])
        mos_preds = []
        with torch.no_grad():
            for s in range(0, len(patches), 32):
                sub = patches[s:s + 32].to(device)
                mp, _ = model(sub)
                mos_preds.append(mp.detach().cpu())
        preds.append(torch.cat(mos_preds).mean().item())
        gts.append(float(row["mos"]))
    return np.array(preds), np.array(gts)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fit 5-param MOS mapping for DIQA-Router")
    ap.add_argument("--csv", default=None, help="已有预测 csv（含 pred_mos/gt_mos）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint 路径（方式 B）")
    ap.add_argument("--split_dir", default=None, help="含 test.csv 的 split 目录（方式 B）")
    ap.add_argument("--image_dir", default=None, help="原始图像目录（方式 B）")
    ap.add_argument("--out", default=os.path.join(HERE, "mos_fit_params.json"))
    ap.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    args = ap.parse_args()

    if args.csv:
        pred, gt = fit_from_csv(args.csv)
    else:
        if not args.split_dir or not args.image_dir:
            ap.error("请提供 --csv，或同时提供 --split_dir 与 --image_dir")
        import torch
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"device: {device}")
        model, train_mod = load_model(args.ckpt, device)
        print("inferring test split...")
        pred, gt = infer_split(model, train_mod, args.split_dir, args.image_dir, device)

    print(f"samples: {len(pred)}")
    report(pred, gt, "raw (uncalibrated)")
    params, fit = fit_logistic5(pred, gt)
    report(fit, gt, "after 5-param fit")
    params["n"] = int(len(pred))
    params["plcc_raw"] = float(pearsonr(pred, gt)[0])
    params["srcc_raw"] = float(spearmanr(pred, gt)[0])
    params["plcc_fit"] = float(pearsonr(fit, gt)[0])
    params["srcc_fit"] = float(spearmanr(fit, gt)[0])

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nsaved -> {args.out}")
    print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
