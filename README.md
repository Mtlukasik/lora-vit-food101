# LoRA Fine-tuning of ViT-B/16 on Food-101

Replication of [rezaakb/peft-vit](https://github.com/rezaakb/peft-vit) — LoRA fine-tuning of a Vision Transformer pretrained on ImageNet-21k, applied to Food-101.

## Quickstart (Google Colab)

Open `notebooks/train_colab.ipynb` and run all cells. Everything is automated.

## What gets trained

| Component | Details |
|-----------|---------|
| Base model | `google/vit-base-patch16-224-in21k` (85.9M params frozen) |
| Trainable | LoRA matrices injected into attention q/v projections |
| Dataset | Food-101 (101 classes, 75,750 train / 25,250 test) |
| Trainable params (r=4) | ~301K |

## Manual setup (local)

```bash
git clone https://github.com/YOUR_USERNAME/lora-vit-food101
cd lora-vit-food101
pip install -r requirements.txt

# Clone the original repo (contains main.py + src/)
git clone https://github.com/rezaakb/peft-vit upstream
cp upstream/main.py .
cp -r upstream/src/* src/

# Train LoRA r=4 on Food-101
python main.py fit --config configs/lora/food101_r4.yaml
```

## Results (from paper)

| Method | # Params | Food-101 (approx) |
|--------|----------|-------------------|
| LoRA r=4 | 301K | ~88% |
| LoRA r=8 | 448K | ~88% |
| Full fine-tune | 85.9M | lower (forgetting) |

## Config variants

- `configs/lora/food101_r4.yaml` — r=4, recommended
- `configs/lora/food101_r8.yaml` — r=8
- `configs/lora/food101_r16.yaml` — r=16
