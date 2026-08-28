# -*- coding: utf-8 -*-
"""
demo.py — DIQA-Router 单图推理演示

用法（作为库）：
    from demo import demo
    score, distortion_type = demo("path/to/image.jpg")

用法（命令行）：
    python demo.py --img path/to/image.jpg
    python demo.py --img path/to/image.jpg --ckpt ./checkpoints-logit-swin-all-best/split_5_best.pth

说明：
    1) 默认加载 split_5 训练的模型（--ckpt 可覆盖）。
    2) 训练用归一化损失，模型原始输出无标定（有正有负）。demo 会先加载
       mos_fit_params.json（由 fit_mos_params.py 生成，把原始预测与真实 MOS 做
       5 参数拟合），再把单图原始预测映射回 1-5 的 MOS 尺度作为最终分数。
       如果找不到该 json，则原样返回原始输出并给出提示。
    3) 失真类型取 7 类失真 logits 中概率最大的一类（noise/blur/overexposure/
       lowlight/warp/stain/occlusion）。

依赖：torch, timm, torchvision, numpy, Pillow
"""
import os
import json
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "checkpoints-logit-swin-all-best", "split_5_best.pth")
DEFAULT_FIT = os.path.join(HERE, "mos_fit_params.json")

DISTORTION_NAMES = ["noise", "blur", "overexposure", "lowlight",
                    "warp", "stain", "occlusion"]

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# ---------------------------------------------------------------
# 5 参数拟合映射（与 fit_mos_params.py 一致）
# ---------------------------------------------------------------
def logistic5(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def apply_mos_mapping(raw_score, params):
    """把模型原始输出映射到 MOS 尺度；params 为 mos_fit_params.json 内容"""
    if params is None:
        return float(raw_score)
    return float(logistic5(raw_score, params["b1"], params["b2"],
                           params["b3"], params["b4"], params["b5"]))


# ---------------------------------------------------------------
# 模型加载（含 timm 本地路径兼容处理）
# ---------------------------------------------------------------
_train_mod = None


def _get_train_module():
    """importlib 加载 logitandpool-moe-swin.py（文件名带连字符，不能直接 import）"""
    global _train_mod
    if _train_mod is None:
        import importlib.util
        import sys
        import types
        # 训练脚本第 23 行 import ipdb 但未使用；注入空 stub 兼容无 ipdb 环境
        if "ipdb" not in sys.modules:
            _stub = types.ModuleType("ipdb")
            _stub.set_trace = lambda *a, **k: None
            sys.modules["ipdb"] = _stub
        spec = importlib.util.spec_from_file_location(
            "diqa_train", os.path.join(HERE, "logitandpool-moe-swin.py"))
        _train_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_train_mod)
    return _train_mod


_model_cache = {}


def load_model(ckpt_path, device):
    """加载 checkpoint；结果按 ckpt 路径缓存"""
    if ckpt_path in _model_cache:
        return _model_cache[ckpt_path]

    import timm
    from types import SimpleNamespace

    train_mod = _get_train_module()

    # SwinBackbone 的 pretrained_cfg_overlay 硬编码了训练机路径，这里去掉；
    # 权重完全由 checkpoint 的 state_dict 提供
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
    _model_cache[ckpt_path] = model
    return model


def load_fit_params(fit_path):
    """加载 5 参数拟合 json；不存在则返回 None"""
    if fit_path and os.path.exists(fit_path):
        with open(fit_path, "r") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------
# 单图推理
# ---------------------------------------------------------------
def predict_image(model, img_path, device, patch_size=224, stride=32, short_side=256):
    """
    与训练/评测一致的推理流程：
      resize 短边 -> 归一化 -> 滑动窗口切 224x224 patch -> 逐 patch 预测 -> 平均
    返回 (mos_raw, dist_logits_avg)
    """
    train_mod = _get_train_module()

    img = Image.open(img_path).convert("RGB")
    img = train_mod.resize_short_side(img, short_side)
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)

    _, H, W = t.shape
    ys = list(range(0, max(H - patch_size + 1, 1), stride)) or [0]
    xs = list(range(0, max(W - patch_size + 1, 1), stride)) or [0]
    patches = torch.stack(
        [NORMALIZE(t[:, y:y + patch_size, x:x + patch_size])
         for y in ys for x in xs])

    mos_preds, dist_logits = [], []
    with torch.no_grad():
        for s in range(0, len(patches), 32):
            sub = patches[s:s + 32].to(device)
            mp, dl = model(sub)
            mos_preds.append(mp.detach().cpu())
            dist_logits.append(dl.detach().cpu())
    mos_raw = torch.cat(mos_preds).mean().item()
    dist_logits_avg = torch.cat(dist_logits).mean(dim=0)  # [7]
    return mos_raw, dist_logits_avg


# ---------------------------------------------------------------
# 对外主接口
# ---------------------------------------------------------------
def demo(img_path, ckpt_path=None, fit_path=None, device=None):
    """
    输入一张图像，返回 (score, distortion_type)。
      score:          校准到 MOS 尺度的质量分数（约 1-5，越高越好）
      distortion_type: 概率最大的失真类型字符串
    """
    ckpt_path = ckpt_path or DEFAULT_CKPT
    fit_path = fit_path or DEFAULT_FIT
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(ckpt_path, device)
    params = load_fit_params(fit_path)
    if params is None:
        print(f"[demo] 未找到拟合参数 {fit_path}，返回原始未校准分数；"
              f"建议先运行 fit_mos_params.py 生成")

    mos_raw, dist_logits = predict_image(model, img_path, device)
    score = apply_mos_mapping(mos_raw, params)

    probs = torch.sigmoid(dist_logits).numpy()
    distortion_type = DISTORTION_NAMES[int(np.argmax(probs))]
    return score, distortion_type


# ---------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DIQA-Router single-image demo")
    ap.add_argument("--img", required=True, help="输入图像路径")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint 路径")
    ap.add_argument("--fit", default=DEFAULT_FIT, help="5 参数拟合 json 路径")
    ap.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {dev}")
    score, dtype = demo(args.img, args.ckpt, args.fit, dev)
    print("=" * 50)
    print(f"Image           : {args.img}")
    print(f"MOS score       : {score:.3f}")
    print(f"Distortion type : {dtype}")
    print("=" * 50)
