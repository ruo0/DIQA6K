# DIQA-Router: Distortion-Aware Feature Selection for Document Image Quality Assessment

Official implementation of **"Route What Matters: Distortion-Aware Feature Selection for Document Image Quality Assessment"**.

DIQA-Router is a distortion-aware framework that learns *where* and *at which feature level* to perceive document quality. It dynamically routes informative spatial regions and hierarchical features via a distortion-aware router, followed by a dual-domain Transformer that models spatial dependencies and cross-scale interactions.

## Repository Structure

```
.
├── logitandpool-moe-swin.py   # training / evaluation entry point
├── losszoo.py                 # loss functions (norm-in-norm, rank, etc.)
├── README.md
└── .gitignore                 # excludes cache / checkpoints / dataset
```

Expected data layout (paths are configurable via `--split_dir` / `--image_dir`):

```
data/
├── images/                    # original document images
└── splits/
    └── split_1/               # repeated 10 times (split_1 .. split_10)
        ├── train.csv          # columns: filename, mos, 7 distortion columns
        ├── val.csv
        └── test.csv
```

Each CSV contains a `filename` column and a `mos` column, plus the seven distortion columns:
`noise_mean, blur_mean, overexposure_mean, lowlight_mean, warp_mean, stain_mean, occlusion_mean`.

## Requirements

- Python >= 3.8
- PyTorch >= 1.9, torchvision
- timm
- numpy, pandas, Pillow, scipy
- `losszoo.py` (included in this repo)

## How It Runs

1. **Preprocessing (cached)**: on first run, every image is resized so its shorter side equals `--short_side` (default 256) and stored as `.npy` files in `cache_resized_s256/` next to the working directory. Subsequent runs reuse the cache if the `short_side` matches.
2. **Training**: for each epoch, the model is trained on randomly cropped `--train_patch_size` (224) patches, then evaluated on the validation set using sliding-window patches (`--test_stride`). The checkpoint with the best validation PLCC is saved.
3. **Testing**: after training, the best checkpoint is loaded and evaluated on the test set, reporting PLCC / SRCC / MAE / RMSE.

## Training

```bash
python logitandpool-moe-swin.py \
    --split_dir ./data/splits/split_1 \
    --image_dir ./data/images \
    --save_dir ./checkpoints \
    --exp_name split_1
```

## Hyper-parameters

| Argument | Default | Meaning |
|---|---|---|
| `--split_dir` | `/home/cbl/.../split_11` | directory containing `train.csv` / `val.csv` / `test.csv` |
| `--image_dir` | `/second_data/...` | directory of the original document images |
| `--epochs` | 150 | number of training epochs |
| `--batch_size` | 32 | mini-batch size |
| `--lr` | 1e-4 | learning rate (backbone uses `lr * 0.1`) |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--num_workers` | 8 | dataloader workers |
| `--seed` | 42 | random seed (training & evaluation are reproducible) |
| `--top_spatial` | 5 | router samples `top_spatial^2` (25) of the 49 spatial tokens per stage |
| `--top_stage` | 2 | router samples `top_stage` (2) of the 4 feature stages |
| `--n_transformer_layers` | 1 | number of dual-domain Transformer encoder layers |
| `--feat_dim` | 512 | feature dimension after projection |
| `--n_heads` | 8 | attention heads in the Transformer |
| `--dropout` | 0.1 | dropout rate |
| `--lambda_distortion` | 0.05 | weight of the distortion BCE loss |
| `--lambda_mos` | 1.0 | weight of the MOS loss (normalized L1 / norm-in-norm) |
| `--train_patch_size` | 224 | patch size for random crops during training |
| `--test_patch_size` | 224 | patch size for evaluation |
| `--test_stride` | 32 | sliding-window stride at evaluation (overlapping patches) |
| `--short_side` | 256 | shorter side after resizing during preprocessing |
| `--save_dir` | `./checkpoints/` | where checkpoints and logs are saved |
| `--exp_name` | `split_1` | experiment name used in saved filenames |

## Notes

- **Backbone weights**: `SwinBackbone` loads a local weight file via `pretrained_cfg_overlay` (a machine-specific path). On a new machine, either adjust that path or remove the `pretrained_cfg_overlay` argument so timm downloads the default ImageNet-pretrained `swin_base_patch4_window7_224` weights.
- The router samples from the predicted softmax distributions during both training and inference; all randomness is controlled by `--seed` for reproducibility.

## Results (DIQA-6K)

| Method | PLCC | SRCC |
|---|---|---|
| **DIQA-Router (Ours)** | **0.9502** | **0.9501** |

See the paper for full intra-dataset, cross-dataset, and ablation results.

## Citation

```bibtex
@article{chen2025router,
  title={Route What Matters: Distortion-Aware Feature Selection for Document Image Quality Assessment},
  author={Chen, Baoliang and Wu, Ruochen and Zhou, Zhihua and Zhang, Qiudan and Wang, Xu},
  journal={arXiv preprint},
  year={2025}
}
```

## License

TBD

## Contact

For dataset access and questions, please contact the authors.
