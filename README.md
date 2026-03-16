# LoRA Fine-tuning of ViT-B/16 on Food-101 + Bayesian Laplace

Replication and extension of [rezaakb/peft-vit](https://github.com/rezaakb/peft-vit) —
Parameter Efficient Fine-tuning of Self-supervised ViTs without Catastrophic Forgetting (CVPR eLVM 2024).

This project fine-tunes a Vision Transformer (ViT-B/16) pretrained on ImageNet-21k using
Low-Rank Adaptation (LoRA) on the Food-101 dataset, then fits a Bayesian Laplace approximation
over the LoRA weights to obtain calibrated uncertainty estimates.

---

## What's in this repo

```
lora-vit-food101/
├── train_colab.ipynb       ← Main Colab notebook (train + compare + Laplace)
├── train_lora.py           ← Standalone training script (called by notebook)
├── laplace_lora_eval.py    ← Bayesian Laplace evaluation over LoRA weights
├── configs/lora/
│   ├── food101_r4.yaml     ← LoRA r=4  config (~147K trainable params)
│   ├── food101_r8.yaml     ← LoRA r=8  config (~295K trainable params)
│   └── food101_r16.yaml    ← LoRA r=16 config (~590K trainable params)
└── README.md
```

---

## What gets trained

| Component | Details |
|-----------|---------|
| Base model | `google/vit-base-patch16-224-in21k` (85.9M params, frozen) |
| Pretrained on | ImageNet-21k (14M images, 21k classes) |
| Fine-tuned on | Food-101 (75,750 train / 25,250 test, 101 classes) |
| Method | LoRA — low-rank matrices injected into attention q/v layers |
| Trainable params | ~147K (r=4) · ~295K (r=8) · ~590K (r=16) |

---

## Quickstart (Google Colab)

1. Open `train_colab.ipynb` in Colab
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run cell 1 (install deps), then **Runtime → Restart runtime**
4. Run cells 2–8 in order

Everything is automated. Checkpoints are saved to your Google Drive at:
```
MyDrive/vit_lora_food101/deterministic/finetuned_vitb16-in21k_food101/
    r4_ep20_query-value/
        best.pt              ← best checkpoint by val accuracy
        epoch_01.pt ... epoch_20.pt
        meta.json            ← config + full training history
        laplace_results.json ← MAP vs Laplace metrics
        laplace.pkl          ← serialized Laplace posterior
```

---

## Training script

`train_lora.py` accepts CLI arguments and can be run standalone:

```bash
# Basic run
python train_lora.py --rank 4 --epochs 20 --lr 0.01

# Custom modules and save location
python train_lora.py \
    --rank 8 \
    --epochs 20 \
    --lr 0.01 \
    --batch_size 64 \
    --target_modules query value key \
    --save_dir /path/to/checkpoints

# All options
python train_lora.py --help
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--rank` | 4 | LoRA rank — higher = more expressive, more params |
| `--epochs` | 20 | Training epochs |
| `--lr` | 0.01 | SGD learning rate |
| `--batch_size` | 64 | Batch size |
| `--lora_alpha` | rank×2 | LoRA scaling factor |
| `--target_modules` | query value | Attention layers to apply LoRA to |
| `--save_dir` | /content/checkpoints | Where to save checkpoints |

---

## Bayesian Laplace over LoRA weights

After MAP training, `laplace_lora_eval.py` fits a diagonal Laplace approximation
**exclusively over the LoRA A/B matrices** — the frozen ViT backbone is untouched.

This works naturally because PEFT sets `requires_grad=False` on all frozen weights,
so `subset_of_weights='all'` in `laplace-torch` targets only the LoRA parameters.

```bash
python laplace_lora_eval.py \
    --meta_path /path/to/meta.json \
    --fit_samples 2000
```

**What it reports:**

| Metric | MAP | Laplace |
|--------|-----|---------|
| Accuracy | ✓ (preserved) | ✓ (identical to MAP) |
| NLL ↓ | baseline | should improve |
| ECE ↓ | baseline | should improve |

Laplace is guaranteed to preserve MAP accuracy. The improvement shows up in calibration:
the model becomes less overconfident on uncertain predictions.

**Output files saved alongside `meta.json`:**
- `laplace_results.json` — MAP vs Laplace metrics (accuracy, NLL, ECE)
- `laplace.pkl` — serialized Laplace object for reuse without refitting

**Loading the Laplace model for inference:**
```python
import pickle

with open('laplace.pkl', 'rb') as f:
    la = pickle.load(f)

# Calibrated probabilistic predictions
probs = la(imgs, pred_type='glm', link_approx='probit')  # shape: [N, 101]
```

---

## Evaluation

The notebook evaluates two things:

**1. Fine-tune domain (Food-101 test set)**
- Uses the 101-class head trained with LoRA
- Measures how well the model learned the new task

**2. Pretrained domain (Imagenette validation set)**
- Imagenette is a 10-class public subset of ImageNet (~100MB, no auth needed)
- The LoRA backbone weights are kept, but the classifier head is swapped back
  to the pretrained `google/vit-base-patch16-224` ImageNet-1k head
- Measures how much the backbone **forgot** its original representations
- This is the paper's core claim: LoRA preserves pretrained knowledge better
  than full fine-tuning

---

## Paper reference

The original paper trains on CIFAR-100 (not Food-101), so numbers are not directly comparable,
but the experimental setup (LoRA rank sweep, forgetting measurement) is equivalent.

| Method | # Params | CIFAR-100 | IN-1k | MEAN |
|--------|----------|-----------|-------|------|
| LoRA r=4  | 301K | 87.91% | 66.82% | 77.37% |
| LoRA r=8  | 448K | 88.27% | 65.99% | 77.13% |
| LoRA r=16 | 743K | 87.84% | 65.06% | 76.45% |

Note: param counts differ from ours because the paper targets `query`, `key`, `value`, `out_proj`;
we default to `query`, `value` only. Add `--target_modules query value key` to match more closely.

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
peft>=0.4.0
laplace-torch
jsonargparse[signatures]==4.23.1
torchmetrics>=1.0.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Citation

```bibtex
@InProceedings{Bafghi_2024_CVPR,
    author    = {Bafghi, Reza Akbarian and Harilal, Nidhin and Monteleoni, Claire and Raissi, Maziar},
    title     = {Parameter Efficient Fine-tuning of Self-supervised ViTs without Catastrophic Forgetting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3679-3684}
}
```
