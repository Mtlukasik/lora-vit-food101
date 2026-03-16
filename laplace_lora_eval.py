"""
Bayesian Laplace approximation over LoRA weights of a fine-tuned ViT.

Usage:
    python laplace_lora_eval.py \
        --meta_path /content/drive/MyDrive/vit_lora_food101/deterministic/finetuned_vitb16-in21k_food101/r4_ep20_query-value/meta.json

What this does:
    1. Loads your MAP fine-tuned model (best.pt)
    2. Fits a diagonal Laplace approximation ONLY over the LoRA A/B matrices
       (PEFT already keeps requires_grad=False on all frozen weights, so
        subset_of_weights='all' naturally targets only LoRA params)
    3. Optimizes prior precision via marginal likelihood
    4. Evaluates on Food-101 test set:
        - MAP accuracy (same as before)
        - Laplace accuracy (should be identical — Laplace preserves MAP predictions)
        - NLL (lower = better calibration)
        - ECE (lower = better calibration)
    5. Saves results to meta_path/../laplace_results.json
"""

import sys, os, json, argparse
sys.path.insert(0, '/content/peft-vit/src')
sys.path.insert(0, '/content/peft-vit')

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
from laplace import Laplace

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--meta_path', type=str, required=True)
parser.add_argument('--data_dir',  type=str, default='/content/data')
parser.add_argument('--batch_size',type=int, default=64)
parser.add_argument('--fit_samples', type=int, default=2000,
                    help='Number of train samples to use when fitting Hessian (subset for speed)')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── Load meta ─────────────────────────────────────────────────────────────────
with open(args.meta_path) as f:
    meta = json.load(f)
run_dir = os.path.dirname(args.meta_path)
print(f'Run: {meta["run_name"]}')
print(f'LoRA rank={meta["rank"]} alpha={meta["lora_alpha"]} modules={meta["target_modules"]}')

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = T.Compose([
    T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Datasets ──────────────────────────────────────────────────────────────────
train_ds_full = torchvision.datasets.Food101(args.data_dir, split='train', transform=train_tf, download=True)
test_ds       = torchvision.datasets.Food101(args.data_dir, split='test',  transform=val_tf,  download=False)

# Subset of train for fitting Hessian — full train set is slow, 2k samples is enough for diag
indices = torch.randperm(len(train_ds_full))[:args.fit_samples].tolist()
fit_ds  = Subset(train_ds_full, indices)

fit_dl  = DataLoader(fit_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
print(f'Fit Hessian on {len(fit_ds)} train samples | Test on {len(test_ds)} samples')

# ── Load MAP model ────────────────────────────────────────────────────────────
base = ViTForImageClassification.from_pretrained(
    meta['model_name'], num_labels=meta['num_classes'], ignore_mismatched_sizes=True
)
lora_cfg = LoraConfig(
    r=meta['rank'], lora_alpha=meta['lora_alpha'],
    target_modules=meta['target_modules'], lora_dropout=0.1, bias='none'
)
model = get_peft_model(base, lora_cfg)
model.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt'), map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Confirm only LoRA params have requires_grad=True
lora_params = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
total_lora   = sum(p for _, p in lora_params)
print(f'\nLoRA params with grad: {len(lora_params)} tensors, {total_lora:,} total params')
print('Sample names:', [n for n, _ in lora_params[:4]])

# ── MAP evaluation ─────────────────────────────────────────────────────────────
print('\n── MAP evaluation ──')
map_correct, map_nll, n = 0, 0.0, 0
ce = torch.nn.CrossEntropyLoss(reduction='sum')
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast('cuda'):
            logits = model(imgs).logits
        map_correct += (logits.argmax(1) == labels).sum().item()
        map_nll     += ce(logits, labels).item()
        n += len(labels)
map_acc = 100 * map_correct / n
map_nll = map_nll / n
print(f'MAP  Accuracy : {map_acc:.2f}%')
print(f'MAP  NLL      : {map_nll:.4f}')

# ── Fit Laplace over LoRA params ──────────────────────────────────────────────
# PEFT sets requires_grad=False on all frozen weights automatically,
# so subset_of_weights='all' + diag Hessian targets only LoRA A/B matrices.
print('\n── Fitting Laplace (diag Hessian over LoRA params) ──')
la = Laplace(
    model,
    likelihood='classification',
    subset_of_weights='all',   # 'all' = all params with requires_grad=True = LoRA only
    hessian_structure='diag',  # diagonal — feasible for ~150K-600K params
)
la.fit(fit_dl)
print('Hessian fitted.')

# Optimize prior precision via marginal likelihood (no val set needed)
print('Optimizing prior precision...')
la.optimize_prior_precision(method='marglik')
print(f'Prior precision: {la.prior_precision}')

# ── Laplace evaluation ────────────────────────────────────────────────────────
print('\n── Laplace evaluation ──')
la_probs_all, labels_all = [], []

for imgs, labels in test_dl:
    imgs = imgs.to(DEVICE)
    with torch.amp.autocast('cuda'):
        # pred_type='glm' + link_approx='probit' is fast and well-calibrated
        probs = la(imgs, pred_type='glm', link_approx='probit')
    la_probs_all.append(probs.cpu())
    labels_all.append(labels)

la_probs_all = torch.cat(la_probs_all)   # [N, num_classes]
labels_all   = torch.cat(labels_all)     # [N]

la_preds   = la_probs_all.argmax(1)
la_correct = (la_preds == labels_all).sum().item()
la_acc     = 100 * la_correct / len(labels_all)
la_nll     = -la_probs_all[range(len(labels_all)), labels_all].log().mean().item()

# Expected Calibration Error (ECE) — 15 bins
def compute_ece(probs, labels, n_bins=15):
    confidences, predictions = probs.max(1)
    accuracies = predictions.eq(labels)
    ece = 0.0
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_acc  = accuracies[mask].float().mean().item()
            bin_conf = confidences[mask].mean().item()
            ece += mask.sum().item() * abs(bin_acc - bin_conf)
    return ece / len(labels)

map_ece_val = compute_ece(
    torch.softmax(
        torch.cat([
            model(imgs.to(DEVICE)).logits.cpu()
            for imgs, _ in test_dl
        ]), dim=1
    ),
    labels_all
)
la_ece = compute_ece(la_probs_all, labels_all)

print(f'Laplace Accuracy : {la_acc:.2f}%  (MAP: {map_acc:.2f}%)')
print(f'Laplace NLL      : {la_nll:.4f}   (MAP: {map_nll:.4f})')
print(f'Laplace ECE      : {la_ece:.4f}   (MAP: {map_ece_val:.4f})')
print(f'\nCalibration improvement: ΔNLL={map_nll-la_nll:+.4f}  ΔECE={map_ece_val-la_ece:+.4f}')

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    'run_name':    meta['run_name'],
    'rank':        meta['rank'],
    'lora_alpha':  meta['lora_alpha'],
    'modules':     meta['target_modules'],
    'fit_samples': args.fit_samples,
    'map': {
        'accuracy': map_acc,
        'nll':      map_nll,
        'ece':      map_ece_val,
    },
    'laplace': {
        'accuracy': la_acc,
        'nll':      la_nll,
        'ece':      la_ece,
        'prior_precision': la.prior_precision.item() if la.prior_precision.numel() == 1
                           else la.prior_precision.tolist(),
    },
}
out_path = os.path.join(run_dir, 'laplace_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {out_path}')

# ── Save Laplace object ───────────────────────────────────────────────────────
import pickle
la_pkl_path = os.path.join(run_dir, 'laplace.pkl')
with open(la_pkl_path, 'wb') as f:
    pickle.dump(la, f)
print(f'Laplace object saved to {la_pkl_path}')
print('\nTo reload later (no refitting needed):')
print('  import pickle')
print('  with open("laplace.pkl", "rb") as f: la = pickle.load(f)')
print('  probs = la(imgs, pred_type="glm", link_approx="probit")')
