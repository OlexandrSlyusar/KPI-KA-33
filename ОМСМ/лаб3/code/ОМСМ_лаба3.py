#1. Середовище та залежності
import os
import sys

os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import copy
import importlib.util
import json
import math
import platform
import random
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import traceback
import urllib.request
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from importlib import metadata
from pathlib import Path

REQUIRED = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "PIL": "pillow",
            "sklearn": "scikit-learn", "tqdm": "tqdm"}
OPTIONAL = {"umap": "umap-learn"}


def ensure_packages():
    if importlib.util.find_spec("torch") is None:
        sys.exit("PyTorch не знайдено: оберіть у Visual Studio середовище omsm_lab2 venv (PyTorch з ROCm).")
    for group, required in ((REQUIRED, True), (OPTIONAL, False)):
        missing = [pip for mod, pip in group.items() if importlib.util.find_spec(mod) is None]
        if not missing:
            continue
        pins = []
        for pkg in ("torch", "torchvision", "numpy"):
            try:
                pins.append(f"{pkg}=={metadata.version(pkg)}")
            except metadata.PackageNotFoundError:
                pass
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(pins))
        print(f"Встановлення пакетів: {' '.join(missing)} ...", flush=True)
        code = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing, "-c", f.name]).returncode
        if code and required:
            sys.exit("Не вдалося встановити залежності - див. повідомлення pip вище.")


ensure_packages()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn import metrics as skm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


#2. Конфігурація експерименту
@dataclass(frozen=True)
class Config:
    category: str = "bottle"
    img: int = 256
    batch: int = 16
    epochs: int = 400
    patience: int = 50
    lr: float = 1e-3
    lr_factor: float = 0.5
    lr_patience: int = 12
    min_lr: float = 1e-5
    val_frac: float = 0.15
    seed: int = 0
    latents: tuple = (32, 64, 128)
    main_latent: int = 128
    threshold_q: float = 0.95
    score_sigma: float = 4.0
    score_topk: float = 0.01
    vae_beta: float = 1.0
    dae_sigma: float = 0.1
    noise_levels: tuple = (0.05, 0.1, 0.2)
    threshold_qs: tuple = (0.90, 0.95, 0.99, 1.0)
    lat_reps: tuple = (100, 20, 20)
    n_vis: int = 12
    out: str = "outputs"


SMOKE = dict(img=64, epochs=3, patience=2, lr_patience=1, latents=(16, 32), main_latent=32, lat_reps=(5, 2, 2),
             out="outputs_smoke")
MODELS = ("ConvAE", "ResConvAE")
BONUS_MODELS = ("VAE", "DAE")
LABELS = {"ConvAE": "ConvAE", "ResConvAE": "Residual ConvAE", "VAE": "Conv VAE", "DAE": "Denoising ConvAE"}
PARAM_STUDY = (("Без аугментацій", "noaug", {"aug": False}), ("lr = 3e-4", "lr3e-4", {"lr": 3e-4}),
               ("batch = 32", "b32", {"batch": 32}), ("MAE (L1) loss", "mae", {"loss": "mae"}))
TYPE_COLORS = {"good": "tab:green", "broken_large": "tab:red", "broken_small": "tab:orange",
               "contamination": "tab:purple"}
MVTEC_URL = {
    "bottle": "https://www.mydrive.ch/shares/163640/9a565b9a02f45cd135e46b1c36e01ec5/download/420937370-1629958698/bottle.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/163645/8346f81c9e093d2ebbb3a73f98372217/download/420937545-1629959162/hazelnut.tar.xz",
}
FLIP_AUG = {"bottle", "hazelnut", "carpet", "grid", "leather", "tile", "wood"}
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "mvtec_ad"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AMP = DEV.type == "cuda"
C = Config()
OUT = BONUS = WEIGHTS = ROOT / C.out


def configure(cfg):
    global C, OUT, BONUS, WEIGHTS
    C, OUT = cfg, ROOT / cfg.out
    BONUS, WEIGHTS = OUT / "bonus", OUT / "weights"
    for p in (OUT, BONUS, WEIGHTS):
        p.mkdir(parents=True, exist_ok=True)


def label(name, z):
    return f"{LABELS[name]} (z={z})"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


#3. Консольний інтерфейс
ANSI = re.compile(r"\x1b\[[0-9;]*m")
COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
if os.name == "nt":
    os.system("")


def paint(text, code):
    return f"\x1b[{code}m{text}\x1b[0m" if COLOR else text


class Tee:
    def __init__(self, path):
        self.stream, self.file = sys.stdout, open(path, "a", encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        self.file.write(ANSI.sub("", s))

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return self.stream.isatty()

    def close(self):
        sys.stdout = self.stream
        self.file.close()


def box(lines, color="36"):
    w = max(len(ANSI.sub("", s)) for s in lines) + 2
    print(paint("╔" + "═" * w + "╗", color))
    for s in lines:
        print(paint("║ ", color) + s + " " * (w - 1 - len(ANSI.sub("", s))) + paint("║", color))
    print(paint("╚" + "═" * w + "╝", color))


def stage(text):
    print("\n" + paint(f"━━ {text} " + "━" * max(3, 86 - len(text)), "1;36"), flush=True)


def info(text):
    print(paint("  • ", "36") + text, flush=True)


def done(text):
    print(paint("  ✓ ", "32") + text, flush=True)


def warn(text):
    print(paint("  ! ", "33") + text, flush=True)


def fmt(v, digits=3):
    if isinstance(v, (float, np.floating)):
        return "–" if not np.isfinite(v) else (f"{v:.{digits}f}" if abs(v) >= 10 ** -digits or v == 0 else f"{v:.1e}")
    return str(v)


def table(df, title=None, digits=3):
    rows = [[str(c) for c in df.columns]] + [[fmt(v, digits) for v in r] for r in df.itertuples(index=False)]
    wid = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    num = [all(re.fullmatch(r"[-–+0-9.e]+", r[i]) for r in rows[1:]) for i in range(len(wid))]
    line = lambda a, b, c: "  " + a + b.join("─" * (w + 2) for w in wid) + c
    if title:
        print(paint(f"\n  {title}", "1"))
    print(line("┌", "┬", "┐"))
    for k, r in enumerate(rows):
        print("  │" + "│".join(f" {v.rjust(w) if k and n else v.ljust(w)} " for v, w, n in zip(r, wid, num)) + "│")
        if k == 0:
            print(line("├", "┼", "┤"))
    print(line("└", "┴", "┘"), flush=True)


#4. Дані MVTec AD
def download(url, dest):
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f, \
            tqdm(total=int(r.headers.get("Content-Length", 0)), unit="B", unit_scale=True, ncols=100,
                 desc=f"  {dest.name}") as bar:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.replace(dest)


def ensure_category(cat):
    root = DATA_ROOT / cat
    if (root / "train" / "good").is_dir():
        return root
    if cat not in MVTEC_URL:
        sys.exit(f"Немає посилання для категорії {cat}: розпакуйте її вручну в {DATA_ROOT}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    archive = DATA_ROOT / f"{cat}.tar.xz"
    for attempt in range(3):
        if archive.exists():
            break
        try:
            download(MVTEC_URL[cat], archive)
        except OSError as e:
            warn(f"спроба {attempt + 1}: {e}")
    if not archive.exists():
        sys.exit(f"Завантажте {cat}.tar.xz з https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads "
                 f"у {DATA_ROOT}")
    info(f"Розпакування {archive.name} ...")
    with tarfile.open(archive, "r:xz") as tf:
        tf.extractall(DATA_ROOT, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    return root


def read_image(path, size, mode, resample):
    with Image.open(path) as im:
        return np.asarray(im.convert(mode).resize((size, size), resample))


def load_category(cat, size):
    cache = DATA_ROOT / f"cache_{cat}_{size}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    root = ensure_category(cat)
    info(f"Підготовка зображень {cat} ({size}×{size}) ...")
    train = sorted((root / "train" / "good").glob("*.png"))
    test = sorted((root / "test").glob("*/*.png"))
    mask = lambda p: (read_image(p, size, "L", Image.NEAREST) > 127) if p.exists() else np.zeros((size, size), bool)
    data = dict(train=np.stack([read_image(p, size, "RGB", Image.BILINEAR) for p in train]),
                train_names=np.array([p.name for p in train]),
                test=np.stack([read_image(p, size, "RGB", Image.BILINEAR) for p in test]),
                test_masks=np.stack([mask(root / "ground_truth" / p.parent.name / f"{p.stem}_mask.png") for p in test]),
                test_types=np.array([p.parent.name for p in test]),
                test_names=np.array([f"{p.parent.name}/{p.name}" for p in test]))
    np.savez_compressed(cache, **data)
    return data


def prepare_data():
    raw = load_category(C.category, C.img)
    perm = np.random.default_rng(C.seed).permutation(len(raw["train"]))
    n_val = int(round(C.val_frac * len(perm)))
    va, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    d = dict(train=raw["train"][tr], val=raw["train"][va], train_names=raw["train_names"][tr],
             val_names=raw["train_names"][va], test=raw["test"], masks=raw["test_masks"], types=raw["test_types"],
             names=raw["test_names"])
    d["y"] = (d["types"] != "good").astype(int)
    d["xtr"], d["xval"] = to_tensor(d["train"]), to_tensor(d["val"])
    split = {"category": C.category, "img_size": C.img, **{k: d[f"{k}_names"].tolist() for k in ("train", "val")},
             "test": d["names"].tolist()}
    (OUT / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    types, counts = np.unique(d["types"], return_counts=True)
    info(f"{C.category}: train {len(d['train'])} | validation {len(d['val'])} (норма) | test {len(d['test'])} "
         f"({', '.join(f'{t} {c}' for t, c in zip(types, counts))})")
    return d


def to_tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).contiguous().to(DEV)


def flip(x):
    if C.category not in FLIP_AUG:
        return x
    shape = (x.shape[0], 1, 1, 1)
    x = torch.where(torch.rand(shape, device=x.device) < 0.5, x.flip(3), x)
    return torch.where(torch.rand(shape, device=x.device) < 0.5, x.flip(2), x)


def add_noise(images, sigma, seed):
    noise = np.random.default_rng(seed).normal(0, sigma * 255, images.shape).astype(np.float32)
    return np.clip(images.astype(np.float32) + noise, 0, 255).round().astype(np.uint8)


#5. Архітектури автоенкодерів
def down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True))


def up(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.LeakyReLU(0.2, True))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.act(x + self.body(x))


class AutoEncoder(nn.Module):
    widths = (32, 64, 128, 256, 256)

    def __init__(self, latent_dim, img_size, residual=False):
        super().__init__()
        self.latent_dim = latent_dim
        w, rev = self.widths, self.widths[::-1]
        block = lambda layer, c: [layer, ResBlock(c)] if residual else [layer]
        self.encoder = nn.Sequential(*[m for cin, c in zip((3,) + w, w) for m in block(down(cin, c), c)])
        self.side = img_size // 2 ** len(w)
        flat = w[-1] * self.side ** 2
        self.to_z = nn.Linear(flat, latent_dim)
        self.from_z = nn.Sequential(nn.Linear(latent_dim, flat), nn.LeakyReLU(0.2, True))
        self.decoder = nn.Sequential(*[m for cin, c in zip(rev, rev[1:]) for m in block(up(cin, c), c)],
                                     nn.ConvTranspose2d(rev[-1], 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        return self.to_z(self.encoder(x).flatten(1))

    def decode(self, z):
        return self.decoder(self.from_z(z).view(len(z), self.widths[-1], self.side, self.side))

    def forward(self, x):
        return self.decode(self.encode(x))


class VAE(AutoEncoder):
    def __init__(self, latent_dim, img_size):
        super().__init__(latent_dim, img_size)
        self.to_logvar = nn.Linear(self.to_z.in_features, latent_dim)

    def encode_stats(self, x):
        h = self.encoder(x).flatten(1)
        return self.to_z(h), self.to_logvar(h).clamp(-10, 10)

    def encode(self, x):
        return self.encode_stats(x)[0]


def build_model(name, z):
    return VAE(z, C.img) if name == "VAE" else AutoEncoder(z, C.img, residual=name == "ResConvAE")


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def load_model(name, z, tag=None):
    model = build_model(name, z).to(DEV)
    model.load_state_dict(torch.load(WEIGHTS / f"{tag or f'{name}_z{z}'}.pt", map_location=DEV)["state_dict"])
    return model.eval()


#6. Карти похибки, SSIM та anomaly score
def gauss(m, sigma=None):
    sigma = sigma or C.score_sigma
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32, device=m.device) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum()).view(1, 1, 1, -1)
    x = F.conv2d(F.pad(m[:, None], (r, r, r, r), mode="reflect"), k)
    return F.conv2d(x, k.transpose(2, 3))[:, 0]


def topk_mean(m):
    flat = m.flatten(1)
    return flat.topk(max(1, round(C.score_topk * flat.shape[1])), dim=1).values.mean(1)


def ssim(x, y, win=7):
    p, n = win // 2, win * win
    f = lambda t: F.avg_pool2d(F.pad(t, (p, p, p, p), mode="reflect"), win, 1)
    ux, uy = f(x), f(y)
    cov = lambda a, b, ua, ub: n / (n - 1) * (f(a * b) - ua * ub)
    vx, vy, vxy = cov(x, x, ux, ux), cov(y, y, uy, uy), cov(x, y, ux, uy)
    s = (2 * ux * uy + 1e-4) * (2 * vxy + 9e-4) / ((ux ** 2 + uy ** 2 + 1e-4) * (vx + vy + 9e-4))
    return s.mean(1), s[..., p:-p, p:-p].mean((1, 2, 3))


@torch.no_grad()
def infer(model, images, batch=16):
    model.eval()
    keys = ("recon", "err", "smooth", "score", "mae", "ssim", "ssim_score", "z")
    acc = {k: [] for k in keys}
    for i in range(0, len(images), batch):
        x = to_tensor(images[i:i + batch]).float() / 255
        z = model.encode(x)
        rec = (model.decode(z).float().clamp(0, 1) * 255).round() / 255
        err = ((x - rec) ** 2).mean(1)
        smooth = gauss(err)
        smap, s = ssim(x, rec)
        for k, v in zip(keys, ((rec * 255).byte().permute(0, 2, 3, 1), err, smooth, topk_mean(smooth),
                               (x - rec).abs().mean((1, 2, 3)), s, topk_mean(gauss(1 - smap)), z.float())):
            acc[k].append(v.cpu())
    out = {k: torch.cat(v).numpy() for k, v in acc.items()}
    out["mse"] = out["err"].reshape(len(out["err"]), -1).mean(1)
    return out


#7. Навчання
def losses(model, name, x, loss="mse", noise=None, sample=True):
    if name == "VAE":
        with torch.autocast(DEV.type, enabled=AMP):
            mu, logvar = model.encode_stats(x)
            rec = model.decode(mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu)
        mse = F.mse_loss(rec.float(), x)
        mu, logvar = mu.float(), logvar.float()
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()
        return mse + C.vae_beta * kl / x[0].numel(), mse, kl
    with torch.autocast(DEV.type, enabled=AMP):
        rec = model((x + noise).clamp(0, 1) if noise is not None else x).float()
    mse = F.mse_loss(rec, x)
    return (F.l1_loss(rec, x) if loss == "mae" else mse), mse, torch.zeros((), device=DEV)


@torch.no_grad()
def validate(model, name, xval, loss, noise):
    model.eval()
    acc = torch.zeros(3, device=DEV)
    for i in range(0, len(xval), 32):
        x = xval[i:i + 32].float() / 255
        acc += torch.stack(losses(model, name, x, loss, None if noise is None else noise[i:i + 32], False)) * len(x)
    return (acc / len(xval)).tolist()


def train(name, z, d, tag=None, out=None, lr=None, batch=None, aug=True, loss="mse"):
    tag, out, lr, batch = tag or f"{name}_z{z}", out or OUT, lr or C.lr, batch or C.batch
    set_seed(C.seed)
    model = build_model(name, z).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=C.lr_factor, patience=C.lr_patience,
                                                       min_lr=C.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    gen = torch.Generator().manual_seed(C.seed)
    xtr, xval = d["xtr"], d["xval"]
    val_noise = None
    if name == "DAE":
        val_noise = (torch.randn(tuple(xval.shape), generator=torch.Generator().manual_seed(C.seed + 1))
                     * C.dae_sigma).to(DEV)
    steps = max(1, len(xtr) // batch)
    best, best_ep, state, hist = float("inf"), 0, None, []
    t_all = time.perf_counter()
    bar = tqdm(range(1, C.epochs + 1), desc=f"  {tag:<20}", leave=True, dynamic_ncols=True,
               bar_format="{desc} {percentage:3.0f}%|{bar:18}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")
    for ep in bar:
        model.train()
        sync()
        t0 = time.perf_counter()
        acc = torch.zeros(3, device=DEV)
        for idx in torch.randperm(len(xtr), generator=gen)[:steps * batch].view(steps, batch):
            x = xtr[idx.to(DEV)].float() / 255
            x = flip(x) if aug else x
            noise = torch.randn_like(x) * C.dae_sigma if name == "DAE" else None
            opt.zero_grad(set_to_none=True)
            total, mse, kl = losses(model, name, x, loss, noise)
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
            acc += torch.stack((total, mse, kl)).detach()
        tr = (acc / steps).tolist()
        sync()
        ep_time = time.perf_counter() - t0
        vl, vm, vk = validate(model, name, xval, loss, val_noise)
        sched.step(vl)
        cur_lr = opt.param_groups[0]["lr"]
        row = dict(epoch=ep, train_loss=tr[0], val_loss=vl, train_mse=tr[1], val_mse=vm, lr=cur_lr, epoch_time=ep_time)
        if name == "VAE":
            row.update(train_kl=tr[2], val_kl=vk)
        hist.append(row)
        if vl < best:
            best, best_ep = vl, ep
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        bar.set_postfix_str(f"train {tr[0]:.5f}, val {vl:.5f}, best {best_ep}, lr {cur_lr:.0e}")
        if ep - best_ep >= C.patience:
            break
    bar.close()
    total_time = time.perf_counter() - t_all
    model.load_state_dict(state)
    torch.save({"state_dict": state, "model": name, "latent": z, "img_size": C.img, "category": C.category},
               WEIGHTS / f"{tag}.pt")
    pd.DataFrame(hist).to_csv(out / f"history_{tag}.csv", index=False)
    stop = f"Early Stopping на {len(hist)}-й епосі" if len(hist) < C.epochs else f"{len(hist)} епох"
    done(f"{tag}: {n_params(model) / 1e6:.2f} M параметрів, найкраща епоха {best_ep} ({stop}), val loss {best:.5f}, "
         f"{total_time:.0f} с")
    return model.eval(), history_info(pd.DataFrame(hist), total_time)


def history_info(h, total_time=None):
    best = int(h.val_loss.idxmin()) + 1
    return dict(history=h, best_epoch=best, epochs=len(h), best_val=float(h.val_loss.min()),
                epoch_time=float(h.epoch_time.median()), train_time=total_time or float(h.epoch_time.sum()))


#8. Оцінювання
@torch.no_grad()
def bench(model, device, batch, reps):
    model.eval()
    x = torch.rand(batch, 3, C.img, C.img, device=device)
    for _ in range(max(2, reps // 5)):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


def latency(model, trials=1):
    med = lambda f: float(np.median([f() for _ in range(trials)]))
    r = {"gpu_b1": np.nan, "gpu_b32": np.nan}
    if DEV.type == "cuda":
        r["gpu_b1"] = med(lambda: bench(model, DEV, 1, C.lat_reps[0]))
        r["gpu_b32"] = med(lambda: bench(model, DEV, 32, C.lat_reps[1]) / 32)
    cpu = copy.deepcopy(model).cpu()
    r["cpu_b1"] = med(lambda: bench(cpu, torch.device("cpu"), 1, C.lat_reps[2]))
    return r


def recon_quality(o, m):
    return dict(MSE=float(o["mse"][m].mean()), MAE=float(o["mae"][m].mean()),
                PSNR=float(np.mean(10 * np.log10(1 / o["mse"][m]))), SSIM=float(o["ssim"][m].mean()))


def evaluate(name, z, model, d, tag=None, out=None, lat=True, save=True, quiet=False):
    tag = tag or f"{name}_z{z}"
    val, test = infer(model, d["val"]), infer(model, d["test"])
    y, s = d["y"], test["score"]
    thr = float(np.quantile(val["score"], C.threshold_q))
    pred = (s > thr).astype(int)
    p, r, _ = skm.precision_recall_curve(y, s)
    good = y == 0
    res = dict(model=name, latent=z, tag=tag, threshold=thr, score=s, pred=pred, val_score=val["score"],
               roc_auc=skm.roc_auc_score(y, s), pr_auc=skm.average_precision_score(y, s), f1=skm.f1_score(y, pred),
               precision=skm.precision_score(y, pred, zero_division=0), recall=skm.recall_score(y, pred),
               accuracy=skm.accuracy_score(y, pred), f1_max=float(np.nanmax(2 * p * r / np.clip(p + r, 1e-12, None))),
               cm=skm.confusion_matrix(y, pred, labels=[0, 1]),
               pixel_auc=skm.roc_auc_score(d["masks"].ravel(), test["smooth"].ravel()),
               good=recon_quality(test, good), all=recon_quality(test, slice(None)), params=n_params(model),
               per_type={t: skm.roc_auc_score(y[(d["types"] == "good") | (d["types"] == t)],
                                              s[(d["types"] == "good") | (d["types"] == t)])
                         for t in sorted(set(d["types"]) - {"good"})},
               test=test, val=val, latency=latency(model) if lat else {})
    if save:
        pd.DataFrame(dict(image=d["names"], type=d["types"], label=y, score=s, pred=pred, mse=test["mse"],
                          ssim=test["ssim"], ssim_score=test["ssim_score"])).to_csv(
            (out or OUT) / f"test_predictions_{tag}.csv", index=False)
    if not quiet:
        lt = res["latency"]
        lat_txt = f" | GPU {fmt(lt['gpu_b1'], 2)} мс, CPU {fmt(lt['cpu_b1'], 1)} мс" if lat else ""
        info(f"ROC-AUC {res['roc_auc']:.3f} | PR-AUC {res['pr_auc']:.3f} | F1 {res['f1']:.3f} | pixel ROC-AUC "
             f"{res['pixel_auc']:.3f} | SSIM норми {res['good']['SSIM']:.3f}{lat_txt}")
    return res


def result_row(r, info_, **extra):
    lat = r.get("latency", {})
    return {"Модель": LABELS[r["model"]], "Latent dim": r["latent"], **extra, "Параметри (М)": r["params"] / 1e6,
            "MSE ↓": r["good"]["MSE"], "MAE ↓": r["good"]["MAE"], "PSNR (дБ) ↑": r["good"]["PSNR"],
            "SSIM ↑": r["good"]["SSIM"], "MSE test (усі)": r["all"]["MSE"], "SSIM test (усі)": r["all"]["SSIM"],
            "ROC-AUC ↑": r["roc_auc"], "PR-AUC ↑": r["pr_auc"], "F1 ↑": r["f1"], "Precision": r["precision"],
            "Recall": r["recall"], "Accuracy": r["accuracy"], "F1-max (оракул)": r["f1_max"],
            "Pixel ROC-AUC": r["pixel_auc"], "Поріг (q за val)": r["threshold"],
            "Найкраща епоха": info_["best_epoch"], "Епох": info_["epochs"],
            "Час епохи (с, медіана)": info_["epoch_time"], "Час навчання (с)": info_["train_time"],
            "Latency GPU b1 (мс)": lat.get("gpu_b1", np.nan), "GPU b32 (мс/зобр.)": lat.get("gpu_b32", np.nan),
            "Latency CPU b1 (мс)": lat.get("cpu_b1", np.nan)}


VIEW = {"Модель": "Модель", "Тип": "Тип", "Latent dim": "z", "Параметри (М)": "Парам., М", "MSE ↓": "MSE",
        "SSIM ↑": "SSIM", "ROC-AUC ↑": "ROC-AUC", "PR-AUC ↑": "PR-AUC", "F1 ↑": "F1", "Pixel ROC-AUC": "Pixel AUC",
        "Latency GPU b1 (мс)": "GPU, мс", "Latency CPU b1 (мс)": "CPU, мс"}


def view(df, drop=()):
    cols = [c for c in VIEW if c in df.columns and VIEW[c] not in drop]
    return df[cols].rename(columns=VIEW)


def save_csv(df, name, out=None):
    df.round(6).to_csv((out or OUT) / name, index=False, encoding="utf-8-sig")


#9. Візуалізації
def save_fig(fig, name, out=None, dpi=130):
    top = 1 - 0.5 / fig.get_figheight() if fig._suptitle is not None else 1
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig((out or OUT) / name, dpi=dpi)
    plt.close(fig)


def panel(img, t=None, cmap=None, vmax=None, heat=None, hmax=None, mask=None, tc="black"):
    return dict(img=img, t=t, cmap=cmap, vmax=vmax, heat=heat, hmax=hmax, mask=mask, tc=tc)


def grid(rows, name, title=None, row_labels=None, size=2.1, out=None, fs=8):
    nr, nc = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(nr, nc, figsize=(size * nc, size * nr + (0.5 if title else 0)), squeeze=False)
    for i, row in enumerate(rows):
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            p = row[j] if j < len(row) else None
            if p is None:
                ax.axis("off")
                continue
            ax.imshow(p["img"], cmap=p["cmap"], vmin=0, vmax=p["vmax"])
            if p["heat"] is not None:
                ax.imshow(p["heat"], cmap="jet", alpha=0.5, vmin=0, vmax=p["hmax"])
            if p["mask"] is not None and p["mask"].any():
                ax.contour(p["mask"], levels=[0.5], colors="white", linewidths=0.8)
            if p["t"]:
                ax.set_title(p["t"], fontsize=fs, color=p["tc"])
        if row_labels:
            axes[i, 0].set_ylabel(row_labels[i], fontsize=fs + 1)
    if title:
        fig.suptitle(title, fontsize=11)
    save_fig(fig, name, out, dpi=120)


def heat_scale(o):
    return float(np.quantile(o["smooth"], 0.999)), float(np.quantile(o["err"], 0.999))


def plot_dataset(d, per_type=3):
    rows, labels = [], []
    for t in sorted(set(d["types"]), key=lambda t: (t != "good", t)):
        idx = np.where(d["types"] == t)[0][:per_type]
        rows.append([q for i in idx for q in (panel(d["test"][i], d["names"][i]),
                                              panel(d["masks"][i], "маска GT", "gray", 1))])
        labels.append(f"{t}\n({(d['types'] == t).sum()} у test)")
    grid(rows, "dataset_examples.png", f"MVTec AD / {C.category}: тестові зображення і маски дефектів "
         f"({C.img}×{C.img})", labels)


def pick_examples(d, n=None, seed=None):
    n, rng = n or C.n_vis, np.random.default_rng(C.seed if seed is None else seed)
    types = sorted(set(d["types"]), key=lambda t: (t != "good", t))
    return sorted(i for t in types for i in rng.choice(np.where(d["types"] == t)[0], n // len(types), replace=False))


def plot_recon_grid(r, d, idx, out=None):
    o, (hs, he) = r["test"], heat_scale(r["test"])
    heads = ("Реконструкція", "Карта похибки", "Heatmap + GT")
    cells = []
    for k, i in enumerate(idx):
        ok = "✓" if r["pred"][i] == d["y"][i] else "✗"
        first = k in (0, (len(idx) + 1) // 2)
        cells.append([panel(d["test"][i], f"{d['names'][i]}\n{r['score'][i]:.4f} {ok}"),
                      panel(o["recon"][i], heads[0] if first else None),
                      panel(o["err"][i], heads[1] if first else None, "inferno", he),
                      panel(d["test"][i], heads[2] if first else None, heat=o["smooth"][i], hmax=hs,
                            mask=d["masks"][i])])
    half = (len(cells) + 1) // 2
    rows = [cells[i] + (cells[i + half] if i + half < len(cells) else []) for i in range(half)]
    grid(rows, f"recon_grid_{r['tag']}.png", f"{label(r['model'], r['latent'])}: реконструкція {len(idx)} тестових "
         f"зображень (score, ✓/✗ – прогноз за порогом {r['threshold']:.4f}; білий контур – GT-маска)", out=out, fs=7.5)


def plot_best_worst(r, d):
    order = np.argsort(r["test"]["mse"])
    idx = list(order[:5]) + list(order[-5:][::-1])
    _, he = heat_scale(r["test"])
    head = lambda k, i: (f"{'найкраща' if k < 5 else 'найгірша'} #{k % 5 + 1}\n{d['names'][i]}\n"
                         f"MSE {r['test']['mse'][i]:.4f} SSIM {r['test']['ssim'][i]:.3f}")
    rows = [[panel(d["test"][i], head(k, i)) for k, i in enumerate(idx)], [panel(r["test"]["recon"][i]) for i in idx],
            [panel(r["test"]["err"][i], cmap="inferno", vmax=he) for i in idx]]
    grid(rows, f"best_worst_{r['tag']}.png", f"{label(r['model'], r['latent'])}: 5 найкращих і 5 найгірших "
         "реконструкцій test (за MSE)", ["Оригінал", "Реконструкція", "Похибка"], fs=7.5)


def plot_outcomes(r, d, out=None):
    y, p, hs = d["y"], r["pred"], heat_scale(r["test"])[0]
    groups = {"TP": (y == 1) & (p == 1), "TN": (y == 0) & (p == 0), "FP": (y == 0) & (p == 1),
              "FN": (y == 1) & (p == 0)}
    rows = []
    for g, m in groups.items():
        idx = np.where(m)[0]
        idx = idx[np.argsort(-r["score"][idx] if g in ("TP", "FP") else r["score"][idx])][:3]
        row = [q for i in idx for q in (panel(d["test"][i], d["names"][i]),
                                        panel(d["test"][i], f"score {r['score'][i]:.4f}", heat=r["test"]["smooth"][i],
                                              hmax=hs, mask=d["masks"][i]))]
        rows.append(row or [panel(np.ones((8, 8, 3)), f"{g}: немає")])
    grid(rows, f"outcomes_{r['tag']}.png", f"{label(r['model'], r['latent'])}: правильні (TP, TN) і помилкові (FP, FN) "
         f"прогнози, поріг {r['threshold']:.4f}", list(groups), size=2.25, out=out)


def plot_training(infos):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 4.6), squeeze=False)
    for ax, m in zip(axes[0], MODELS):
        for j, z in enumerate(C.latents):
            h, be = infos[(m, z)]["history"], infos[(m, z)]["best_epoch"]
            ax.plot(h.epoch, h.train_loss, "--", color=f"C{j}", alpha=0.7, label=f"train, z={z}")
            ax.plot(h.epoch, h.val_loss, color=f"C{j}", label=f"val, z={z}")
            ax.scatter([be], [h.val_loss[be - 1]], color=f"C{j}", s=25, zorder=5)
        ax.set(yscale="log", title=f"{LABELS[m]}: train / validation loss (MSE)", xlabel="Епоха", ylabel="MSE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    save_fig(fig, "training_curves.png")


def plot_roc_pr(results, d, name="roc_pr_curves.png", out=None, style_by_model=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for r in results:
        main = r["latent"] == C.main_latent
        kw = dict(lw=2.4 if main else 1.0, alpha=1 if main else 0.6,
                  ls=("-" if r["model"] == MODELS[0] else "--") if style_by_model else "-")
        fpr, tpr, _ = skm.roc_curve(d["y"], r["score"])
        p, rc, _ = skm.precision_recall_curve(d["y"], r["score"])
        axes[0].plot(fpr, tpr, label=f"{label(r['model'], r['latent'])}: AUC {r['roc_auc']:.3f}", **kw)
        axes[1].plot(rc, p, label=f"{label(r['model'], r['latent'])}: AP {r['pr_auc']:.3f}", **kw)
    axes[0].plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
    axes[1].axhline(d["y"].mean(), ls=":", color="grey", lw=0.8, label=f"випадковий ({d['y'].mean():.2f})")
    axes[0].set(title="ROC-криві (test)", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[1].set(title="PR-криві (test, позитивний клас – дефект)", xlabel="Recall", ylabel="Precision")
    for ax, loc in zip(axes, ("lower right", "lower left")):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc=loc)
    save_fig(fig, name, out)


def plot_scores(results, d):
    types = sorted(set(d["types"]), key=lambda t: (t != "good", t))
    groups = [("val (норма)", None)] + [(f"test: {t}", t) for t in types]
    rng = np.random.default_rng(C.seed)
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 4.2), squeeze=False)
    for ax, r in zip(axes[0], results):
        for k, (g, t) in enumerate(groups):
            v = r["val_score"] if t is None else r["score"][d["types"] == t]
            ax.scatter(v, k + rng.uniform(-0.18, 0.18, len(v)), s=18, alpha=0.8,
                       color="tab:blue" if t is None else TYPE_COLORS.get(t, "tab:brown"))
        ax.axvline(r["threshold"], color="black", ls="--", lw=1.2, label=f"поріг q={C.threshold_q} за val")
        ax.set_yticks(range(len(groups)), [g for g, _ in groups])
        ax.set(title=label(r["model"], r["latent"]),
               xlabel=f"anomaly score (top-{C.score_topk:.0%} згладженої карти похибки)")
        ax.grid(alpha=0.3, axis="x")
        ax.legend(fontsize=8)
    save_fig(fig, "score_distributions.png")


def plot_confusion(results):
    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4), squeeze=False)
    for ax, r in zip(axes[0], results):
        cm = r["cm"]
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["норма", "дефект"])
        ax.set_yticks([0, 1], ["норма", "дефект"])
        ax.set(xlabel="Прогноз", ylabel="Істина")
        ax.set_title(f"{label(r['model'], r['latent'])}\nF1 {r['f1']:.3f}", fontsize=10)
    save_fig(fig, "confusion_matrices.png")


def plot_latent_study(df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = (("ROC-AUC ↑", "ROC-AUC (test)"), ("PR-AUC ↑", "PR-AUC (test)"),
             ("MSE ↓", "MSE реконструкції (test, норма)"), ("SSIM ↑", "SSIM реконструкції (test, норма)"))
    for ax, (col, title) in zip(axes.ravel(), specs):
        for m, sub in df.groupby("Модель", sort=False):
            ax.plot(sub["Latent dim"], sub[col], "o-", label=m)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["Latent dim"].unique()), [str(z) for z in sorted(df["Latent dim"].unique())])
        ax.set(title=title, xlabel="Розмірність латентного простору")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save_fig(fig, "latent_dim_study.png")


@torch.no_grad()
def encode(model, images):
    return np.concatenate([model.encode(to_tensor(images[i:i + 32]).float() / 255).float().cpu().numpy()
                           for i in range(0, len(images), 32)])


def plot_embeddings(named, d, methods, name, out=None, ncols=2):
    groups = np.array(["train (норма)"] * len(d["train"]) + ["val (норма)"] * len(d["val"]) +
                      [f"test: {t}" for t in d["types"]])
    order = ["train (норма)", "val (норма)"] + sorted({g for g in groups if g.startswith("test")},
                                                     key=lambda g: (g != "test: good", g))
    cells = []
    for key, model in named:
        z = np.concatenate([encode(model, d[k]) for k in ("train", "val", "test")])
        z = (z - z.mean(0)) / (z.std(0) + 1e-8)
        for m in methods:
            title = f"{label(key, model.latent_dim)}: {m}"
            if m == "PCA":
                pca = PCA(2, random_state=C.seed).fit(z)
                cells.append((f"{title} (дисперсія {pca.explained_variance_ratio_.sum():.0%})", pca.transform(z)))
            elif m == "t-SNE":
                tsne = TSNE(2, perplexity=min(30, (len(z) - 1) / 3), init="pca", random_state=C.seed)
                cells.append((title, tsne.fit_transform(z)))
            else:
                import umap
                cells.append((title, umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=C.seed).fit_transform(z)))
    nrows = -(-len(cells) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.6 * nrows), squeeze=False)
    for ax, (title, e) in zip(axes.ravel(), cells):
        for g in order:
            m, t = groups == g, g.split(": ")[-1]
            kw = (dict(color="lightgrey", s=12) if g.startswith("train") else dict(color="tab:blue", s=18)
                  if g.startswith("val") else dict(color=TYPE_COLORS.get(t, "tab:brown"), s=30, edgecolor="k", lw=0.3,
                                                   marker="o" if t == "good" else "^"))
            ax.scatter(e[m, 0], e[m, 1], label=f"{g} ({m.sum()})", alpha=0.85, **kw)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    save_fig(fig, name, out)


@torch.no_grad()
def plot_interpolation(named, d, name, out=None, steps=8):
    good, bad = np.where(d["types"] == "good")[0], np.where(d["types"] != "good")[0]
    pairs = ((d["val"][0], d["val"][1], "норма → норма"), (d["test"][good[0]], d["test"][bad[0]], "норма → дефект"))
    alphas = np.linspace(0, 1, steps)
    rows, labels = [], []
    for key, model in named:
        for a, b, title in pairs:
            za, zb = torch.from_numpy(encode(model, np.stack([a, b]))).to(DEV)
            dec = model.decode(torch.stack([(1 - t) * za + t * zb for t in alphas])).clamp(0, 1).permute(0, 2, 3, 1)
            head = not rows
            rows.append([panel(a, "A" if head else None)] + [panel(im, f"α = {t:.2f}" if head else None)
                                                             for im, t in zip(dec.cpu().numpy(), alphas)]
                        + [panel(b, "B" if head else None)])
            labels.append(f"{LABELS[key]}\n{title}")
    grid(rows, name, "Лінійна інтерполяція в латентному просторі: decode((1−α)·z_A + α·z_B), "
         f"z={named[0][1].latent_dim}", labels, size=1.9, out=out)


#10. Основний експеримент: ConvAE vs Residual ConvAE, z = 32 / 64 / 128
def run_main(d):
    stage(f"Навчання основних моделей: {len(MODELS)} архітектури × {len(C.latents)} розмірності z")
    models, results, infos = {}, [], {}
    for z in [C.main_latent] + [z for z in C.latents if z != C.main_latent]:
        for m in MODELS:
            model, infos[(m, z)] = train(m, z, d)
            results.append(evaluate(m, z, model, d))
            if z == C.main_latent:
                models[m] = model
    results.sort(key=lambda r: (MODELS.index(r["model"]), r["latent"]))
    main = [next(r for r in results if r["model"] == m and r["latent"] == C.main_latent) for m in MODELS]

    stage("Візуалізації основного експерименту")
    idx = pick_examples(d)
    plot_dataset(d)
    plot_training(infos)
    for r in main:
        plot_recon_grid(r, d, idx)
        plot_best_worst(r, d)
        plot_outcomes(r, d)
    plot_roc_pr(results, d)
    plot_scores(main, d)
    plot_confusion(main)
    plot_embeddings([(m, models[m]) for m in MODELS], d, ("PCA", "t-SNE"), "latent_space.png")
    plot_interpolation([(m, models[m]) for m in MODELS], d, "latent_interpolation.png")
    done("рисунки збережено")

    df = pd.DataFrame([result_row(r, infos[(r["model"], r["latent"])]) for r in results])
    save_csv(df, "comparison_table.csv")
    plot_latent_study(df)
    per = pd.DataFrame([{"Модель": LABELS[r["model"]], "Latent dim": r["latent"], **r["per_type"]} for r in results])
    save_csv(per, "per_defect_auc.csv")
    meta = dict(category=C.category, img_size=C.img, main_latent=C.main_latent, score_sigma=C.score_sigma,
                score_topk=C.score_topk, models=[dict(model=r["model"], latent=r["latent"], threshold=r["threshold"],
                                                      weights=f"weights/{r['tag']}.pt") for r in main])
    (OUT / "models.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    table(view(df), "Порівняння моделей (реконструкція – норма test; виявлення – увесь test; latency – batch 1)")
    table(per, "ROC-AUC за типами дефектів")
    return models, {r["model"]: r for r in main}, {m: infos[(m, C.main_latent)] for m in MODELS}


#11. Бонус: VAE, Denoising AE, шум, UMAP, способи score, поріг, параметри навчання, latency
def load_main(d):
    models, results, infos = {}, {}, {}
    for m in MODELS:
        models[m] = load_model(m, C.main_latent)
        info(f"Завантажено {label(m, C.main_latent)}")
        results[m] = evaluate(m, C.main_latent, models[m], d, save=False)
        infos[m] = history_info(pd.read_csv(OUT / f"history_{m}_z{C.main_latent}.csv"))
    return models, results, infos


def plot_vae(model, inf, n=24):
    h, be = inf["history"], inf["best_epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
    for ax, (col, title) in zip(axes, (("mse", "Reconstruction loss (MSE)"), ("kl", "KL-divergence (на зображення)"),
                                     ("loss", f"Загальна втрата: MSE + β·KL/D, β={C.vae_beta}"))):
        ax.plot(h.epoch, h[f"train_{col}"], "--", label="train")
        ax.plot(h.epoch, h[f"val_{col}"], label="val")
        ax.axvline(be, color="grey", ls=":", lw=1, label=f"найкраща епоха {be}")
        ax.set(title=title, xlabel="Епоха", yscale="log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save_fig(fig, "vae_losses.png", BONUS)
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim, generator=torch.Generator().manual_seed(C.seed)).to(DEV)
        imgs = model.decode(z).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    grid([[panel(im) for im in imgs[i:i + 8]] for i in range(0, n, 8)], "vae_generation.png",
         f"Conv VAE: нові зразки, z ~ N(0, I), z ∈ R^{model.latent_dim}", out=BONUS)


@torch.no_grad()
def to_clean(images, clean):
    a, b = to_tensor(images).float() / 255, to_tensor(clean).float() / 255
    mse = ((a - b) ** 2).mean((1, 2, 3)).clamp_min(1e-10)
    return float(mse.mean()), float((10 * torch.log10(1 / mse)).mean()), float(ssim(a, b)[1].mean())


def noise_study(named, d):
    rows, keep = [], {}
    good = d["y"] == 0
    show = [int(np.where(good)[0][0]), int(np.where(d["types"] == "broken_small")[0][0])
            if "broken_small" in set(d["types"]) else int(np.where(~good)[0][0])]
    for sigma in (0.0,) + tuple(C.noise_levels):
        val = d["val"] if sigma == 0 else add_noise(d["val"], sigma, C.seed + 10)
        test = d["test"] if sigma == 0 else add_noise(d["test"], sigma, C.seed + 11)
        in_mse, _, in_ssim = to_clean(test[good], d["test"][good])
        for key, model in named:
            vo, to = infer(model, val), infer(model, test)
            thr = float(np.quantile(vo["score"], C.threshold_q))
            pred = (to["score"] > thr).astype(int)
            mse, psnr, s = to_clean(to["recon"][good], d["test"][good])
            rows.append({"Модель": LABELS[key], "σ шуму": sigma, "ROC-AUC": skm.roc_auc_score(d["y"], to["score"]),
                         "PR-AUC": skm.average_precision_score(d["y"], to["score"]), "F1": skm.f1_score(d["y"], pred),
                         "MSE до чистого": mse, "PSNR до чистого (дБ)": psnr, "SSIM до чистого": s,
                         "PSNR входу (дБ)": 10 * math.log10(1 / in_mse) if sigma else np.inf, "SSIM входу": in_ssim})
            keep[(key, sigma)] = (test[show], to["recon"][show], to["smooth"][show])
    df = pd.DataFrame(rows)
    save_csv(df, "noise_robustness.csv", BONUS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
    for m, sub in df.groupby("Модель", sort=False):
        for ax, col in zip(axes, ("ROC-AUC", "F1", "PSNR до чистого (дБ)")):
            ax.plot(sub["σ шуму"], sub[col], "o-", label=m)
    sub = df[df["Модель"] == LABELS[named[0][0]]].iloc[1:]
    axes[2].plot(sub["σ шуму"], sub["PSNR входу (дБ)"], "k:", marker="x", label="зашумлений вхід")
    titles = ("ROC-AUC (test)", "F1 (поріг за зашумленим val)", "PSNR відновлення до чистого (test, норма)")
    for ax, title in zip(axes, titles):
        ax.set(title=title, xlabel="σ гаусівського шуму (шкала [0, 1])")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save_fig(fig, "noise_robustness.png", BONUS)
    keys = [k for k, _ in named]
    rows_, labels = [], []
    for j, i in enumerate(show):
        for sigma in C.noise_levels:
            hm = [keep[(k, sigma)] for k in keys]
            rows_.append([panel(hm[0][0][j], "Вхід (зашумлений)")] +
                         [panel(h[1][j], f"{LABELS[k]}: вихід") for k, h in zip(keys, hm)] +
                         [panel(d["test"][i], "Ціль (чисте)")] +
                         [panel(d["test"][i], f"{LABELS[k]}: heatmap", heat=h[2][j],
                                hmax=float(np.quantile(h[2], 0.999))) for k, h in zip(keys, hm)])
            labels.append(f"{d['names'][i]}\nσ = {sigma}")
    grid(rows_, "dae_denoising.png", "Denoising: Input (noisy) → Output → Target (clean) для трьох рівнів шуму", labels,
         size=2.3, out=BONUS, fs=7.5)
    return df


def score_types(named, results, d):
    rows = []
    for key, model in named:
        r = results[key]
        maha = LedoitWolf().fit(encode(model, d["train"])).mahalanobis(encode(model, d["test"]))
        row = {"Модель": LABELS[key]}
        for title, s in ((f"top-{C.score_topk:.0%} MSE (основний)", r["score"]), ("середній MSE", r["test"]["mse"]),
                         (f"top-{C.score_topk:.0%} (1−SSIM)", r["test"]["ssim_score"]),
                         ("Махаланобіс у латентному просторі", maha)):
            row[f"ROC-AUC: {title}"] = skm.roc_auc_score(d["y"], s)
            row[f"PR-AUC: {title}"] = skm.average_precision_score(d["y"], s)
        rows.append(row)
    df = pd.DataFrame(rows)
    save_csv(df, "score_types.csv", BONUS)
    return df


def threshold_study(results, d):
    rows = []
    for key, r in results.items():
        for q in C.threshold_qs:
            thr = float(np.quantile(r["val_score"], q))
            pred = (r["score"] > thr).astype(int)
            cm = skm.confusion_matrix(d["y"], pred, labels=[0, 1])
            rows.append({"Модель": LABELS[key], "Квантиль val": "max" if q == 1 else q, "Поріг": thr,
                         "F1": skm.f1_score(d["y"], pred),
                         "Precision": skm.precision_score(d["y"], pred, zero_division=0),
                         "Recall": skm.recall_score(d["y"], pred), "FP": int(cm[0, 1]), "FN": int(cm[1, 0])})
    df = pd.DataFrame(rows)
    save_csv(df, "threshold_sensitivity.csv", BONUS)
    return df


def param_study(base, base_info, d):
    row = lambda name, r, inf: {"Варіант": name, "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"], "F1": r["f1"],
                                "MSE (test, норма)": r["good"]["MSE"], "SSIM (test, норма)": r["good"]["SSIM"],
                                "Pixel ROC-AUC": r["pixel_auc"], "Найкраща епоха": inf["best_epoch"],
                                "Епох": inf["epochs"], "Час навчання (с)": inf["train_time"]}
    rows, curves = [row("База (MSE, lr 1e-3, batch 16, flip)", base, base_info)], [("База", base_info["history"])]
    for title, slug, kw in PARAM_STUDY:
        tag = f"ConvAE_z{C.main_latent}_{slug}"
        model, inf = train("ConvAE", C.main_latent, d, tag=tag, out=BONUS, **kw)
        rows.append(row(title, evaluate("ConvAE", C.main_latent, model, d, tag, BONUS, lat=False), inf))
        curves.append((title, inf["history"]))
    df = pd.DataFrame(rows)
    save_csv(df, "training_params_study.csv", BONUS)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for title, h in curves:
        ax.plot(h.epoch, h["val_mse" if "val_mse" in h else "val_loss"], label=title)
    ax.set(yscale="log", title=f"ConvAE (z={C.main_latent}): validation MSE за різних параметрів навчання",
           xlabel="Епоха", ylabel="val MSE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(fig, "training_params_curves.png", BONUS)
    return df


def latency_table():
    rows = []
    for m, z in [(m, z) for m in MODELS for z in C.latents] + [(m, C.main_latent) for m in BONUS_MODELS]:
        if (WEIGHTS / f"{m}_z{z}.pt").exists():
            model = load_model(m, z)
            lat = latency(model, trials=5)
            rows.append({"Модель": LABELS[m], "Latent dim": z, "Параметри (М)": n_params(model) / 1e6,
                         "GPU b1 (мс)": lat["gpu_b1"], "GPU b32 (мс/зобр.)": lat["gpu_b32"],
                         "CPU b1 (мс)": lat["cpu_b1"]})
    df = pd.DataFrame(rows)
    save_csv(df, "latency_all.csv", BONUS)
    return df


def run_bonus(d, models=None, results=None, infos=None):
    if models is None:
        stage("Бонус: завантаження збережених основних моделей")
        models, results, infos = load_main(d)
    stage(f"Бонус [1/5]: Conv VAE (β = {C.vae_beta}) і Denoising ConvAE (σ = {C.dae_sigma})")
    models, results, infos = dict(models), dict(results), dict(infos)
    for m in BONUS_MODELS:
        models[m], infos[m] = train(m, C.main_latent, d, out=BONUS)
        results[m] = evaluate(m, C.main_latent, models[m], d, out=BONUS)
    keys = MODELS + BONUS_MODELS
    named = [(k, models[k]) for k in keys]
    kind = lambda k: {"Тип": "основна" if k in MODELS else "бонус"}
    cmp = pd.DataFrame([result_row(results[k], infos[k], **kind(k)) for k in keys])
    save_csv(cmp, "bonus_comparison.csv", BONUS)
    idx = pick_examples(d)
    for k in BONUS_MODELS:
        plot_recon_grid(results[k], d, idx, BONUS)
    plot_roc_pr([results[k] for k in keys], d, "roc_pr_4models.png", BONUS, style_by_model=False)
    plot_vae(models["VAE"], infos["VAE"])
    plot_interpolation([("VAE", models["VAE"])], d, "vae_interpolation.png", BONUS)
    done("VAE: криві MSE / KL, генерація, інтерполяція")

    stage("Бонус [2/5]: стійкість до шуму – ConvAE vs Denoising ConvAE на однаково зашумлених даних")
    noise = noise_study([("ConvAE", models["ConvAE"]), ("DAE", models["DAE"])], d)
    table(noise[["Модель", "σ шуму", "ROC-AUC", "F1", "PSNR до чистого (дБ)", "SSIM до чистого", "PSNR входу (дБ)"]],
          "ROC-AUC / F1 на зашумленому test; PSNR / SSIM – відновлення норми відносно чистих зображень")

    stage("Бонус [3/5]: UMAP, способи оцінки аномальності, чутливість до порогу")
    if importlib.util.find_spec("umap"):
        plot_embeddings(named, d, ("UMAP",), "latent_umap.png", BONUS)
        done("UMAP латентного простору")
    else:
        warn("umap-learn недоступний – UMAP пропущено")
    st, th = score_types(named, results, d), threshold_study({k: results[k] for k in MODELS}, d)
    roc = [c for c in st.columns if c.startswith("ROC")]
    table(st[["Модель"] + roc].rename(columns={c: c.replace("ROC-AUC: ", "").replace(" у латентному просторі", " (z)")
                                               for c in roc}), "ROC-AUC різних способів score")
    table(th, "Чутливість F1 до порогу", 4)

    stage("Бонус [4/5]: вплив параметрів навчання (ConvAE)")
    ps = param_study(results["ConvAE"], infos["ConvAE"], d)
    table(ps.drop(columns=["Час навчання (с)", "Pixel ROC-AUC"]).rename(
        columns={"MSE (test, норма)": "MSE норми", "SSIM (test, норма)": "SSIM норми", "Найкраща епоха": "Краща епоха"}),
        "Параметри навчання ConvAE (test)")

    stage("Бонус [5/5]: повторний вимір latency усіх моделей (медіана з 5 серій)")
    lat = latency_table()
    table(lat, "Latency (batch 1 / 32, вхід 3×256×256, fp32)", 2)
    table(view(cmp, ("z", "GPU, мс", "CPU, мс")), f"Чотири архітектури (z = {C.main_latent})")


#12. Інференс збережених моделей
def run_inference(d, n=12):
    stage(f"Інференс збережених моделей на {n} тестових зображеннях")
    meta = json.loads((OUT / "models.json").read_text(encoding="utf-8"))
    idx = pick_examples(d, n, seed=1)
    rows, view, cells, labels = [], [], [], []
    loaded = [(load_model(m["model"], m["latent"]), m) for m in meta["models"]]
    outs = [infer(model, d["test"][idx]) for model, _ in loaded]
    for k, i in enumerate(idx):
        truth = "дефект" if d["y"][i] else "норма"
        row = {"Зображення": d["names"][i], "Істина": truth}
        cell = [panel(d["test"][i], "Оригінал" if k == 0 else None)]
        for (_, m), o in zip(loaded, outs):
            s = float(o["score"][k])
            pred = "дефект" if s > m["threshold"] else "норма"
            row[LABELS[m["model"]]] = f"{s:.4f} → {pred}" + ("" if pred == truth else " ✗")
            rows.append(dict(image=d["names"][i], truth=truth, model=LABELS[m["model"]], score=s,
                             threshold=m["threshold"], pred=pred, correct=pred == truth))
            cell += [panel(o["recon"][k], f"{LABELS[m['model']]}\nреконструкція" if k == 0 else None),
                     panel(d["test"][i], f"{s:.4f} → {pred}", heat=o["smooth"][k],
                           hmax=float(np.quantile(o["smooth"], 0.999)), tc="green" if pred == truth else "red")]
        cells.append(cell)
        view.append(row)
        labels.append(f"{d['names'][i]}\nістина: {truth}")
    grid(cells, "inference_demo.png", None, labels, size=2.2)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "inference_demo.csv", index=False, encoding="utf-8-sig")
    table(pd.DataFrame(view), "Інференс (score → прогноз; ✗ – помилка)")
    for m, sub in df.groupby("model", sort=False):
        done(f"{m}: правильно {int(sub['correct'].sum())} з {len(sub)}")


#13. Меню, запуск і журнал
MODES = {"1": ("all", "Повний експеримент: основні моделі → бонус → інференс", "~45 хв"),
         "2": ("main", "Лише основні моделі (ConvAE / Residual ConvAE, z = 32 / 64 / 128)", "~25 хв"),
         "3": ("bonus", "Лише бонус на збережених вагах (VAE, DAE, шум, UMAP, параметри)", "~15 хв"),
         "4": ("infer", "Інференс збережених моделей на 12 тестових зображеннях", "~1 хв"),
         "5": ("smoke", "Smoke-тест усього конвеєра на зображеннях 64×64", "~2 хв")}


def device_line():
    if DEV.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        hip = getattr(torch.version, "hip", None)
        backend = f"ROCm/HIP {hip}" if hip else f"CUDA {torch.version.cuda}"
        return f"GPU {p.name} ({p.total_memory / 2 ** 30:.0f} ГБ) · PyTorch {torch.__version__} · {backend} · AMP"
    return f"CPU ({platform.processor() or platform.machine()}) · PyTorch {torch.__version__} – GPU не знайдено"


def status_lines():
    lines = [paint("Дані: ", "1") + ("✓ " if (DATA_ROOT / C.category / "train").is_dir() else "✗ буде завантажено, ") +
             f"MVTec AD / {C.category}"]
    n = sum((WEIGHTS / f"{m}_z{z}.pt").exists() for m in MODELS for z in C.latents)
    lines.append(paint("Збережені моделі: ", "1") + f"{n} з {len(MODELS) * len(C.latents)} основних" +
                 (", бонус ✓" if (BONUS / "bonus_comparison.csv").exists() else ""))
    if (OUT / "comparison_table.csv").exists():
        t = pd.read_csv(OUT / "comparison_table.csv")
        t = t[t["Latent dim"] == C.main_latent]
        lines.append(paint("Останні результати: ", "1") + " · ".join(
            f"{r['Модель']} ROC-AUC {r['ROC-AUC ↑']:.3f}" for _, r in t.iterrows()))
    return lines


def ask(default, timeout=None):
    prompt = f"  Ваш вибір [{default}]"
    if not sys.stdin.isatty():
        return default
    if os.name == "nt" and timeout:
        import msvcrt
        end, shown = time.time() + timeout, None
        while (left := math.ceil(end - time.time())) > 0:
            if left != shown:
                print(f"\r{prompt} – автозапуск через {left:2d} с (будь-яка клавіша – вибрати вручну) ", end="",
                      flush=True)
                shown = left
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                print("\r" + " " * 100 + "\r", end="")
                if ch in "\r\n":
                    return default
                if ch in MODES or ch == "0":
                    print(f"{prompt}: {ch}")
                    return ch
                break
            time.sleep(0.05)
        else:
            print(f"\r{prompt}: {default}" + " " * 70)
            return default
    try:
        return input(f"{prompt}: ").strip() or default
    except EOFError:
        return "0"


def run(mode, cfg):
    configure(replace(cfg, **SMOKE) if mode == "smoke" else cfg)
    need = {"bonus": WEIGHTS / f"{MODELS[0]}_z{C.main_latent}.pt", "infer": OUT / "models.json"}.get(mode)
    if need and not need.exists():
        return warn(f"Немає {need.relative_to(ROOT)} – спочатку виконайте режим 1 або 2")
    if mode == "all" and OUT.exists():
        shutil.rmtree(OUT)
        configure(C)
    log = Tee(OUT / "log.txt")
    sys.stdout = log
    t0 = time.time()
    try:
        print(f"\n==== {datetime.now():%Y-%m-%d %H:%M} · режим {mode} · {device_line()} ====")
        stage("Дані")
        d = prepare_data()
        versions = {k: metadata.version(k) for k in ("torch", "torchvision", "numpy", "pandas", "scikit-learn",
                                                      "matplotlib", "pillow")}
        split = dict(train=len(d["train"]), val=len(d["val"]), test=len(d["test"]), test_defective=int(d["y"].sum()))
        cfg_json = dict(config=asdict(C), mode=mode, device=device_line(), python=platform.python_version(),
                        versions=versions, split=split,
                        score=f"mean top-{C.score_topk:.0%} пікселів (x - x̂)², Gaussian σ={C.score_sigma}",
                        threshold=f"квантиль {C.threshold_q} score на validation")
        (OUT / "run_config.json").write_text(json.dumps(cfg_json, ensure_ascii=False, indent=2), encoding="utf-8")
        models = results = infos = None
        if mode in ("all", "main", "smoke"):
            models, results, infos = run_main(d)
        if mode in ("all", "bonus", "smoke"):
            run_bonus(d, models, results, infos)
        if mode in ("all", "main", "infer", "smoke"):
            run_inference(d)
        stage(f"Готово за {(time.time() - t0) / 60:.1f} хв · результати: {OUT}")
    except KeyboardInterrupt:
        warn("перервано користувачем")
    except Exception:
        print(paint(traceback.format_exc(), "31"))
        warn("режим завершився з помилкою – повний текст у log.txt")
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser(description="ОМСМ, лабораторна робота 3: автоенкодери для виявлення дефектів")
    parser.add_argument("mode", nargs="?", choices=[m for m, _, _ in MODES.values()])
    parser.add_argument("--category", default=Config.category, choices=sorted(MVTEC_URL))
    args = parser.parse_args()
    cfg = Config(category=args.category)
    if args.mode:
        return run(args.mode, cfg)
    first = True
    while True:
        configure(cfg)
        print()
        box([paint("ОМСМ · Лабораторна робота 3 · Автоенкодери (Autoencoders)", "1"),
             f"Варіант 13 · MVTec AD / {cfg.category} · ConvAE vs Residual ConvAE",
             "КА-33 · Тодощенко Є., Сивенко К., Слюсар О.", "", device_line(), *status_lines()])
        width = max(len(text) for _, text, _ in MODES.values())
        for key, (_, text, eta) in MODES.items():
            print(f"   {paint(key, '1;33')}  {text.ljust(width)}  {paint(eta, '2')}")
        print(f"   {paint('0', '1;33')}  Вихід")
        choice = ask("1", 20) if first else ask("0")
        while choice not in MODES and choice != "0":
            choice = ask("0")
        first = False
        if choice == "0":
            return
        run(MODES[choice][0], cfg)


if __name__ == "__main__":
    main()
