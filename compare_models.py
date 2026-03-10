import sys, os, json, glob
sys.path.insert(0, '/content/peft-vit/src')
sys.path.insert(0, '/content/peft-vit')

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = '/content/checkpoints'
DATA_DIR = '/content/data'

# ── Datasets ──────────────────────────────────────────────────────────────────
val_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Food-101 test set (fine-tune domain)
food_ds = torchvision.datasets.Food101(DATA_DIR, split='test', transform=val_tf, download=False)
food_dl = DataLoader(food_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# ImageNet-1k validation via torchvision (pretrained domain)
# Uses a small 50-image-per-class subset available without full download
# Falls back gracefully if not available
imagenet_dl = None
imagenet_path = '/content/data/imagenet-val'
if os.path.exists(imagenet_path):
    imagenet_ds = torchvision.datasets.ImageFolder(imagenet_path, transform=val_tf)
    imagenet_dl = DataLoader(imagenet_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    print(f"ImageNet val set found: {len(imagenet_ds)} images")
else:
    print("No ImageNet val set found at /content/data/imagenet-val — skipping IN-1k eval.")
    print("To add it: organize 1k-class folders at that path (ImageFolder format).\n")

# ── Eval function ─────────────────────────────────────────────────────────────
def evaluate(model, dataloader, label=''):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(imgs).logits
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)
    acc = 100 * correct / n
    print(f"  {label:<40} {acc:.2f}%  ({correct}/{n})")
    return acc

def load_model(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    run_dir = os.path.dirname(meta_path)
    ckpt    = os.path.join(run_dir, 'best.pt')

    base = ViTForImageClassification.from_pretrained(
        meta['model_name'],
        num_labels=meta['num_classes'],
        ignore_mismatched_sizes=True
    )
    lora_cfg = LoraConfig(
        r=meta['rank'],
        lora_alpha=meta['lora_alpha'],
        target_modules=meta['target_modules'],
        lora_dropout=0.1,
        bias='none'
    )
    model = get_peft_model(base, lora_cfg)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model = model.to(DEVICE)
    return model, meta

# ── Find all runs ─────────────────────────────────────────────────────────────
meta_files = sorted(glob.glob(os.path.join(CKPT_DIR, '**/meta.json'), recursive=True))
if not meta_files:
    print(f"No runs found in {CKPT_DIR}. Train at least one model first.")
    sys.exit(1)

print(f"\nFound {len(meta_files)} trained run(s):\n")

results = []
for meta_path in meta_files:
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"{'─'*60}")
    print(f"Run:     {meta['run_name']}")
    print(f"Config:  r={meta['rank']} alpha={meta['lora_alpha']} modules={meta['target_modules']}")
    print(f"Trained: {meta['epochs']} epochs | lr={meta['lr']} | best_val={meta['best_val_acc']:.2f}%")
    print(f"Evaluating...")

    model, meta = load_model(meta_path)

    food_acc = evaluate(model, food_dl,  'Food-101 test (fine-tune domain)')
    in1k_acc = evaluate(model, imagenet_dl, 'ImageNet-1k val (pretrained domain)') if imagenet_dl else None

    results.append({
        'run':        meta['run_name'],
        'rank':       meta['rank'],
        'alpha':      meta['lora_alpha'],
        'modules':    meta['target_modules'],
        'epochs':     meta['epochs'],
        'food101':    food_acc,
        'imagenet':   in1k_acc,
        'best_train': meta['best_val_acc'],
    })
    del model
    torch.cuda.empty_cache()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'RUN':<35} {'r':>4} {'Food-101':>10} {'IN-1k':>10}")
print(f"{'─'*60}")
for r in sorted(results, key=lambda x: x['food101'], reverse=True):
    in1k = f"{r['imagenet']:.2f}%" if r['imagenet'] is not None else "N/A"
    print(f"{r['run']:<35} {r['rank']:>4} {r['food101']:>9.2f}% {in1k:>10}")
print(f"{'='*60}\n")
