#1. Налаштування
import copy
import os
import random
import time
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.flop_counter import FlopCounterMode
from torchvision import datasets, models
from torchvision import transforms as T
warnings.filterwarnings("ignore")
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15
FREEZE_EPOCHS = 3
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
LR_SCRATCH = 1e-3
HEAD_LR_MULT = 10
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)

DATA_ROOT = Path("data")
OUT_DIR = Path("outputs")
MY_IMAGES_DIR = Path("my_images")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

import torch_directml

if torch_directml.is_available():
    DEVICE = torch_directml.device()
    print("Відеокарта AMD (DirectML)")
else:
    DEVICE = torch.device("cpu")
    
USE_AMP = False
OUT_DIR.mkdir(exist_ok=True)
assert 0 < FREEZE_EPOCHS < EPOCHS and EPOCHS >= 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_grid(n, cols, cell_w, cell_h):
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_w, rows * cell_h), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat


set_seed(SEED)
print("Device:", DEVICE, "| torch", torch.__version__)


#2. Дані (Caltech-101, stratified split 70 / 15 / 15)
def find_dataset_dir(root):
    for d in (root / "caltech101" / "101_ObjectCategories", root / "101_ObjectCategories"):
        if d.is_dir():
            return d
    datasets.Caltech101(root=str(root), download=True)
    return root / "caltech101" / "101_ObjectCategories"


DATA_DIR = find_dataset_dir(DATA_ROOT)
CLASSES = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != "BACKGROUND_Google")

ALL_PATHS, ALL_LABELS = [], []
for class_idx, class_name in enumerate(CLASSES):
    files = sorted((DATA_DIR / class_name).glob("*.jpg"))
    ALL_PATHS += files
    ALL_LABELS += [class_idx] * len(files)
ALL_LABELS = np.array(ALL_LABELS)
assert len(CLASSES) == 101, len(CLASSES)

all_idx = np.arange(len(ALL_PATHS))
idx_train, idx_tmp = train_test_split(all_idx, test_size=0.30, stratify=ALL_LABELS, random_state=SEED)
idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, stratify=ALL_LABELS[idx_tmp], random_state=SEED)
assert not (set(idx_train) & set(idx_val) or set(idx_train) & set(idx_test) or set(idx_val) & set(idx_test))

counts = np.bincount(ALL_LABELS)
print(f"Класів: {len(CLASSES)} | зображень: {len(ALL_PATHS)} | на клас: min={counts.min()}, max={counts.max()}")
print(f"train/val/test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

normalize = T.Normalize(MEAN, STD)
eval_tf = T.Compose([T.Resize(IMG_SIZE * 8 // 7), T.CenterCrop(IMG_SIZE), T.ToTensor(), normalize])
aug_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.ToTensor(),
    normalize,
])


class CaltechDataset(Dataset):
    def __init__(self, indices, transform):
        self.paths = [ALL_PATHS[i] for i in indices]
        self.labels = [int(ALL_LABELS[i]) for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            img = img.convert("RGB")
        return self.transform(img), self.labels[i]


def make_loader(indices, transform, shuffle):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        CaltechDataset(indices, transform),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        generator=generator if shuffle else None,
    )


val_loader = make_loader(idx_val, eval_tf, shuffle=False)
test_loader = make_loader(idx_test, eval_tf, shuffle=False)


#3. Моделі
def build_model(arch, pretrained):
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    else:
        raise ValueError(arch)
    return model


def head_of(model):
    return model.fc if hasattr(model, "fc") else model.classifier


def gradcam_layer(model):
    return model.layer4[-1] if hasattr(model, "layer4") else model.features[-1]


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model):
    model = copy.deepcopy(model).cpu().eval()
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
    return counter.get_total_flops() / 2 / 1e9


@torch.no_grad()
def latency_ms(model, device, warmup=20, runs=100):
    model = copy.deepcopy(model).to(device).eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


#4. Навчання
STAGES = {
    "tl": [
        dict(epochs=FREEZE_EPOCHS, frozen=True, lr=LR_HEAD, head_mult=1),
        dict(epochs=EPOCHS - FREEZE_EPOCHS, frozen=False, lr=LR_FINETUNE, head_mult=HEAD_LR_MULT),
    ],
    "scratch": [dict(epochs=EPOCHS, frozen=False, lr=LR_SCRATCH, head_mult=1)],
}

RUNS = [
    dict(id="resnet18_tl", label="ResNet18 (TL, aug)", arch="resnet18", pretrained=True, augment=True),
    dict(id="mobilenet_v2_tl", label="MobileNetV2 (TL, aug)", arch="mobilenet_v2", pretrained=True, augment=True),
    dict(id="resnet18_scratch", label="ResNet18 (з нуля, aug)", arch="resnet18", pretrained=False, augment=True),
    dict(id="resnet18_tl_noaug", label="ResNet18 (TL, без aug)", arch="resnet18", pretrained=True, augment=False),
]


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits, targets = [], []
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            out = model(x)
        out = out.float()
        total_loss += F.cross_entropy(out, y, reduction="sum").item()
        logits.append(out.cpu())
        targets.append(y.cpu())
    logits, targets = torch.cat(logits), torch.cat(targets)
    top5 = logits.topk(5, dim=1).indices
    preds = top5[:, 0]
    return {
        "loss": total_loss / len(targets),
        "top1": (preds == targets).float().mean().item(),
        "top5": (top5 == targets[:, None]).any(dim=1).float().mean().item(),
        "f1": f1_score(targets.numpy(), preds.numpy(), labels=np.arange(len(CLASSES)), average="macro", zero_division=0),
        "preds": preds.numpy(),
        "targets": targets.numpy(),
    }


def train_one_epoch(model, loader, optimizer, scaler, frozen):
    model.train()
    if frozen:
        head = head_of(model)
        for child in model.children():
            if child is not head:
                child.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            out = model(x)
            loss = F.cross_entropy(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        seen += len(y)
    return total_loss / seen, correct / seen


def make_optimizer(model, lr, frozen, head_mult):
    head_params = list(head_of(model).parameters())
    head_ids = {id(p) for p in head_params}
    groups = [{"params": head_params, "lr": lr * head_mult}]
    if not frozen:
        groups.append({"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def run_experiment(run):
    set_seed(SEED)
    model = build_model(run["arch"], run["pretrained"]).to(DEVICE)
    train_loader = make_loader(idx_train, aug_tf if run["augment"] else eval_tf, shuffle=True)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    ckpt_path = OUT_DIR / f"{run['id']}.pt"
    head_ids = {id(p) for p in head_of(model).parameters()}
    history, best, epoch = [], None, 0
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for stage in STAGES["tl" if run["pretrained"] else "scratch"]:
        for p in model.parameters():
            p.requires_grad = (not stage["frozen"]) or id(p) in head_ids
        optimizer = make_optimizer(model, stage["lr"], stage["frozen"], stage["head_mult"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage["epochs"])
        for _ in range(stage["epochs"]):
            epoch += 1
            sync()
            start = time.perf_counter()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, stage["frozen"])
            sync()
            epoch_time = time.perf_counter() - start
            scheduler.step()
            val = evaluate(model, val_loader)
            row = dict(epoch=epoch, frozen=stage["frozen"], train_loss=train_loss, train_acc=train_acc,
                       val_loss=val["loss"], val_top1=val["top1"], val_top5=val["top5"], val_f1=val["f1"],
                       epoch_time=epoch_time)
            history.append(row)
            if best is None or row["val_f1"] > best["val_f1"]:
                best = row
                torch.save(model.state_dict(), ckpt_path)
            print(f"[{run['id']}] {epoch:02d}/{EPOCHS} {'frozen' if stage['frozen'] else 'full  '} "
                  f"loss {train_loss:.3f}/{val['loss']:.3f} | val top1 {val['top1']:.4f} top5 {val['top5']:.4f} "
                  f"F1 {val['f1']:.4f} | {epoch_time:.1f}s")

    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else np.nan
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    history = pd.DataFrame(history)
    history.to_csv(OUT_DIR / f"{run['id']}_history.csv", index=False)
    return dict(
        **run,
        history=history,
        best=best,
        epoch_time=history.loc[~history["frozen"], "epoch_time"].mean(),
        peak_mem=peak_mem,
        model=model.cpu(),
    )

results = {}
for run in RUNS:
    results[run["id"]] = run_experiment(run)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

#5. Оцінка
for r in results.values():
    model = r["model"].to(DEVICE)
    r["test"] = evaluate(model, test_loader)
    r["params"] = count_params(model)
    r["gmacs"] = count_gmacs(model)
    r["lat_cpu"] = latency_ms(model, torch.device("cpu"))
    r["lat_gpu"] = latency_ms(model, DEVICE) if DEVICE.type == "cuda" else np.nan
    r["model"] = model.cpu()

table = pd.DataFrame([{
    "Модель": r["label"],
    "Параметри (М)": r["params"],
    "FLOPs (Г, MAC)": r["gmacs"],
    "Час епохи (с)": r["epoch_time"],
    "Latency CPU (мс)": r["lat_cpu"],
    "Latency GPU (мс)": r["lat_gpu"],
    "Пам'ять GPU (МБ)": r["peak_mem"],
    "Val Acc": r["best"]["val_top1"],
    "Val Macro-F1": r["best"]["val_f1"],
    "Test Acc": r["test"]["top1"],
    "Test Top-5": r["test"]["top5"],
    "Test Macro-F1": r["test"]["f1"],
} for r in results.values()]).round(3)
table.to_csv(OUT_DIR / "comparison_table.csv", index=False)
table

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
panels = [("train_loss", "Train loss"), ("val_loss", "Val loss"),
          ("val_top1", "Val accuracy (top-1)"), ("val_f1", "Val macro-F1")]
for ax, (col, title) in zip(axes.ravel(), panels):
    for r in results.values():
        ax.plot(r["history"]["epoch"], r["history"][col], marker="o", ms=3, label=r["label"])
    ax.set_title(title)
    ax.set_xlabel("Епоха")
    ax.grid(True, alpha=0.3)
axes[0, 0].legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "training_curves.png", dpi=150)
plt.show()


fig, axes = make_grid(len(results), 2, 8, 7)
for ax, r in zip(axes, results.values()):
    cm = confusion_matrix(r["test"]["targets"], r["test"]["preds"], labels=np.arange(len(CLASSES)))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"{r['label']} | Test Acc={r['test']['top1']:.3f}, F1={r['test']['f1']:.3f}")
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Істинний клас")
plt.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=150)
plt.show()

best_id = max(results, key=lambda k: results[k]["best"]["val_f1"])
cm = confusion_matrix(results[best_id]["test"]["targets"], results[best_id]["test"]["preds"],
                      labels=np.arange(len(CLASSES)))
np.fill_diagonal(cm, 0)
print(f"Найчастіші помилки ({results[best_id]['label']}), істина -> передбачення: кількість")
for true_i, pred_i in zip(*np.unravel_index(np.argsort(cm, axis=None)[::-1][:10], cm.shape)):
    if cm[true_i, pred_i]:
        print(f"  {CLASSES[true_i]} -> {CLASSES[pred_i]}: {cm[true_i, pred_i]}")

#6. Grad-CAM

def grad_cam(model, x):
    store = {}
    hook = gradcam_layer(model).register_forward_hook(lambda module, inp, out: store.update(act=out))
    model.eval()
    logits = model(x.unsqueeze(0).to(DEVICE).requires_grad_(True))
    pred = logits.argmax(1).item()
    grads = torch.autograd.grad(logits[0, pred], store["act"])[0]
    hook.remove()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), pred


def to_image(x):
    return (x.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1)


test_ds = CaltechDataset(idx_test, eval_tf)
cam_idx = np.random.RandomState(SEED).choice(len(test_ds), 2, replace=False)
samples = [test_ds[i] for i in cam_idx]

fig, axes = make_grid((1 + len(results)) * 2, 2, 4.5, 4.5)
for col, (x, y) in enumerate(samples):
    axes[col].imshow(to_image(x))
    axes[col].set_title(f"Істина: {CLASSES[y]}")
    axes[col].axis("off")
for row, r in enumerate(results.values(), start=1):
    model = r["model"].to(DEVICE)
    for col, (x, y) in enumerate(samples):
        cam, pred = grad_cam(model, x)
        ax = axes[row * 2 + col]
        ax.imshow(to_image(x))
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"{r['label']}\nпередбачення: {CLASSES[pred]}", fontsize=9)
        ax.axis("off")
    r["model"] = model.cpu()
plt.tight_layout()
fig.savefig(OUT_DIR / "gradcam.png", dpi=150)
plt.show()


#7 Збереження найкращої моделі та інференс на власних зображеннях

torch.save({"arch": results[best_id]["arch"], "state_dict": results[best_id]["model"].state_dict(),
            "classes": CLASSES}, OUT_DIR / "best_model.pt")
print("Найкраща модель:", results[best_id]["label"], "->", OUT_DIR / "best_model.pt")

ckpt = torch.load(OUT_DIR / "best_model.pt", map_location=DEVICE)
best_model = build_model(ckpt["arch"], pretrained=False)
best_model.load_state_dict(ckpt["state_dict"])
best_model.to(DEVICE).eval()


@torch.no_grad()
def predict_top_k(model, path, k=5):
    with Image.open(path) as img:
        x = eval_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs, idx = F.softmax(model(x), dim=1)[0].topk(k)
    return [(ckpt["classes"][i], p) for p, i in zip(probs.tolist(), idx.tolist())]


own_files = sorted(p for p in MY_IMAGES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:10]
if not own_files:
    MY_IMAGES_DIR.mkdir(exist_ok=True)
    print(f"Папка {MY_IMAGES_DIR}/ порожня: додайте 5-10 зображень і запустіть цю комірку ще раз.")
else:
    fig, axes = make_grid(len(own_files), 5, 4, 4.5)
    for ax, path in zip(axes, own_files):
        top = predict_top_k(best_model, path)
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{top[0][0]} ({top[0][1]:.2f})", fontsize=10)
        ax.axis("off")
        print(path.name, "->", ", ".join(f"{name} {prob:.2f}" for name, prob in top))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "own_images.png", dpi=150)
    plt.show()


#8. Висновки

t = {k: v["test"] for k, v in results.items()}
pct = lambda v: f"{100 * v:.1f}%"

top_id = max(t, key=lambda k: t[k]["f1"])
print(f"Найкраща модель за Test Macro-F1: {results[top_id]['label']} "
      f"(Acc {pct(t[top_id]['top1'])}, Top-5 {pct(t[top_id]['top5'])}, F1 {t[top_id]['f1']:.3f}).")

a, b = t["resnet18_tl"], t["resnet18_scratch"]
print(f"Transfer learning vs з нуля (ResNet18): Acc {pct(a['top1'])} vs {pct(b['top1'])}, "
      f"Macro-F1 {a['f1']:.3f} vs {b['f1']:.3f}, різниця Acc = {100 * (a['top1'] - b['top1']):+.1f} п.п.")

a, b = t["resnet18_tl"], t["resnet18_tl_noaug"]
print(f"Вплив аугментацій (ResNet18 TL): Acc {pct(a['top1'])} з aug vs {pct(b['top1'])} без aug, "
      f"Macro-F1 {a['f1']:.3f} vs {b['f1']:.3f}, різниця Acc = {100 * (a['top1'] - b['top1']):+.1f} п.п.")

r18, mnet = results["resnet18_tl"], results["mobilenet_v2_tl"]
print(f"ResNet18 vs MobileNetV2 (TL): параметри {r18['params']:.1f}M vs {mnet['params']:.1f}M, "
      f"FLOPs {r18['gmacs']:.2f}G vs {mnet['gmacs']:.2f}G, "
      f"latency CPU {r18['lat_cpu']:.1f} vs {mnet['lat_cpu']:.1f} мс, "
      f"Acc {pct(r18['test']['top1'])} vs {pct(mnet['test']['top1'])}.")
