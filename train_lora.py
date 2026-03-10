import sys, os, json, argparse
sys.path.insert(0, '/content/peft-vit/src')
sys.path.insert(0, '/content/peft-vit')

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model

# ── Config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--rank',       type=int,   default=4)
parser.add_argument('--epochs',     type=int,   default=20)
parser.add_argument('--lr',         type=float, default=0.01)
parser.add_argument('--batch_size', type=int,   default=64)
parser.add_argument('--lora_alpha', type=int,   default=None)  # defaults to rank*2
parser.add_argument('--target_modules', nargs='+', default=['query', 'value'])
parser.add_argument('--save_dir',   type=str,   default='/content/checkpoints')
args = parser.parse_args()

RANK           = args.rank
EPOCHS         = args.epochs
LR             = args.lr
BATCH_SIZE     = args.batch_size
LORA_ALPHA     = args.lora_alpha or RANK * 2
TARGET_MODULES = args.target_modules
SAVE_DIR       = args.save_dir
NUM_CLASSES    = 101
DATA_DIR       = '/content/data'
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'

run_name = f"lora_r{RANK}_alpha{LORA_ALPHA}_ep{EPOCHS}_{'_'.join(TARGET_MODULES)}"
save_path = os.path.join(SAVE_DIR, run_name)
os.makedirs(save_path, exist_ok=True)

print(f"\n{'='*60}")
print(f"Run: {run_name}")
print(f"Device: {DEVICE} | Rank: {RANK} | Alpha: {LORA_ALPHA} | Epochs: {EPOCHS}")
print(f"LR: {LR} | Batch: {BATCH_SIZE} | Modules: {TARGET_MODULES}")
print(f"Save dir: {save_path}")
print(f"{'='*60}\n")

# ── Data ──────────────────────────────────────────────────────────────────────
train_tf = T.Compose([
    T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
val_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

train_ds = torchvision.datasets.Food101(DATA_DIR, split='train', transform=train_tf, download=True)
val_ds   = torchvision.datasets.Food101(DATA_DIR, split='test',  transform=val_tf,  download=False)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

# ── Model ─────────────────────────────────────────────────────────────────────
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
lora_cfg = LoraConfig(
    r=RANK, lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=0.1, bias='none'
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
model = model.to(DEVICE)

# ── Training ──────────────────────────────────────────────────────────────────
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = torch.cuda.amp.GradScaler()

history = {'train_loss': [], 'val_acc': []}
best_acc = 0.0

for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss, steps = 0.0, 0
    for i, (imgs, labels) in enumerate(train_dl):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = model(imgs, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        steps += 1
        if i % 100 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} step {i}/{len(train_dl)} loss={loss.item():.4f}")
    scheduler.step()
    avg_loss = total_loss / steps

    # Validate
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(imgs).logits
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)
    val_acc = 100 * correct / n
    history['train_loss'].append(avg_loss)
    history['val_acc'].append(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | loss={avg_loss:.4f} | val_acc={val_acc:.2f}%")

    # Save best checkpoint
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(save_path, 'best.pt'))
        print(f"  ↑ New best: {best_acc:.2f}% — saved best.pt")

    # Save latest checkpoint every epoch
    torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch+1:02d}.pt'))

# ── Save config + history ─────────────────────────────────────────────────────
meta = {
    'run_name':       run_name,
    'rank':           RANK,
    'lora_alpha':     LORA_ALPHA,
    'target_modules': TARGET_MODULES,
    'epochs':         EPOCHS,
    'lr':             LR,
    'batch_size':     BATCH_SIZE,
    'best_val_acc':   best_acc,
    'history':        history,
    'model_name':     'google/vit-base-patch16-224-in21k',
    'dataset':        'food101',
    'num_classes':    NUM_CLASSES,
}
with open(os.path.join(save_path, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nDone! Best val acc: {best_acc:.2f}%")
print(f"Saved to: {save_path}")
