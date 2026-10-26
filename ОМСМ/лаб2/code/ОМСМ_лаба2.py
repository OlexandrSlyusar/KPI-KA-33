"""Лабораторна робота 2. Детекція об'єктів з YOLO. Варіант 13 - VisDrone.

Класи: pedestrian / person / car / bus / truck. Обов'язкове порівняння: yolov8n vs yolov5s
(+ третя модель yolov7-tiny за бажанням), фокус - дрібні об'єкти.

Запуск у Visual Studio: Start (F5) або з терміналу:  python ОМСМ_лаба2.py
Скрипт можна переривати й запускати знову: готові етапи (дані, навчені моделі, оцінки)
пропускаються, перерване навчання продовжується з last.pt.
"""
#1. Налаштування
import contextlib
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from PIL import Image

from yolo_common import (CLASSES, REPOS, REPOS_DIR, RUNS_DIR, WEIGHTS_DIR, WORK_ROOT, Detector, draw, imread,
                         imwrite, letterbox, predict_sliced, to_tensor)

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # фреймворки не повинні самі змінювати пакети (torch ROCm)

SMOKE = os.environ.get("LAB2_SMOKE") == "1"  # швидка перевірка працездатності всього конвеєра

SEED = 0
IMG_SIZE = 640                 # однаковий imgsz для всіх моделей (вимога методички)
BATCH_SIZE = 16
EPOCHS = 20                    # >= 10 епох (мінімальна вимога)
PATIENCE = 10                  # рання зупинка (YOLOv8, YOLOv5)
TRAIN_FRACTION = 1.0           # частка train, однакова для всіх моделей (на CPU можна зменшити)
FREEZE_EPOCHS = 5              # бонус: заморожений backbone -> розморожування
TUNE_LR0 = [0.005, 0.01, 0.02] # бонус: легке тюнінгування lr0 (короткі запуски)
TUNE_EPOCHS = 5
BONUS_FRACTION = 0.3           # бонусні абляції YOLOv8n - на однаковій 30% підвибірці train (власний базовий запуск)
NMS_IOU_GRID = [0.45, 0.5, 0.6, 0.7]
CONF_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
EVAL_CONF, EVAL_IOU, MAX_DET = 0.001, 0.6, 1000   # однакові параметри оцінки для всіх моделей
DEPLOY_CONF = 0.25             # поріг для візуалізацій / confusion matrix / FP-FN
SAHI_MODELS = ["yolov8n", "yolov5s"]
SAHI_TILE, SAHI_OVERLAP = 640, 0.2
HIGH_RES = 960                 # бонус: інференс при більшій роздільній здатності (дрібні об'єкти)
RUN_THIRD_MODEL = True         # yolov7-tiny з офіційного репозиторію
RUN_BONUS = True
SHOW_PLOTS = False             # True - показувати вікна matplotlib (блокує виконання до закриття вікна)
if not SHOW_PLOTS:
    plt.switch_backend("Agg")
WORKERS = min(8, os.cpu_count() or 1)

DEVICE = "0" if torch.cuda.is_available() else "cpu"           # ROCm-збірка PyTorch теж дає torch.cuda
TORCH_DEVICE = torch.device("cuda:0" if DEVICE == "0" else "cpu")

# Профіль під обладнання: на CPU навчання YOLO на всьому VisDrone триває десятки годин, тому
# зберігаємо мінімум методички (10 епох), але навчаємо всі моделі на однаковій випадковій 30% підвибірці train.
# Val і test-dev використовуються повністю. Значення можна змінити вручну.
PROFILE = os.environ.get("LAB2_PROFILE", "auto")   # auto | gpu | cpu
if PROFILE == "cpu" or (PROFILE == "auto" and DEVICE == "cpu"):
    EPOCHS, TRAIN_FRACTION, FREEZE_EPOCHS, BONUS_FRACTION = 10, 0.3, 3, 0.15
EPOCHS = int(os.environ.get("LAB2_EPOCHS", EPOCHS))

EPOCHS = int(os.environ.get("LAB2_EPOCHS", EPOCHS))                  # ручне перевизначення
TRAIN_FRACTION = float(os.environ.get("LAB2_FRACTION", TRAIN_FRACTION))
BATCH_SIZE = int(os.environ.get("LAB2_BATCH", BATCH_SIZE))

if SMOKE:
    EPOCHS, TUNE_EPOCHS, FREEZE_EPOCHS, PATIENCE, BATCH_SIZE = 2, 1, 1, 2, 4
    TUNE_LR0, NMS_IOU_GRID, CONF_GRID = [0.005, 0.01], [0.5, 0.6], [0.2, 0.3]
    WORKERS = min(2, WORKERS)

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "outputs"
MY_IMAGES_DIR = PROJECT_DIR / "my_images"
MY_VIDEO_DIR = PROJECT_DIR / "my_video"
RAW_DIR = WORK_ROOT / "datasets" / "VisDrone"
DATA_DIR = WORK_ROOT / "datasets" / ("visdrone5_smoke" if SMOKE else "visdrone5")
DATA_YAML = DATA_DIR / "data.yaml"
BONUS_YAML = DATA_DIR / "data_bonus.yaml"
if SMOKE:
    RUNS_DIR = RUNS_DIR.parent / "runs_smoke"
    OUT_DIR = PROJECT_DIR / "outputs_smoke"
CACHE_DIR = OUT_DIR / "eval_cache"

# VisDrone: 0 ignored regions, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle,
# 8 awning-tricycle, 9 bus, 10 motor, 11 others.  Варіант 13: pedestrian / person(=people) / car / bus / truck.
VISDRONE_TO_OURS = {1: 0, 2: 1, 4: 2, 9: 3, 6: 4}
SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
VISDRONE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/{}.zip"
WEIGHT_URLS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "yolov5s.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
    "yolov7-tiny.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
}
SIZE_NAMES = ["small (<32²)", "medium (32²–96²)", "large (>96²)"]

# Однакова схема навчання й аугментацій для всіх моделей (значення YOLOv5 hyp.scratch-low = default YOLOv8)
HYP = dict(lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
           warmup_bias_lr=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5,
           shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0)
NO_AUG = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0)


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


def finish_fig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def download(url, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Завантаження {url}")
    last = [0]

    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total > 0 and done - last[0] > total / 20:
            last[0] = done
            print(f"  {min(100, 100 * done / total):5.1f}%  ({done / 2**20:.0f} / {total / 2**20:.0f} МБ)")

    urllib.request.urlretrieve(url, tmp, hook)
    tmp.rename(dest)
    return dest


def save_versions():
    import ultralytics
    info = {
        "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__, "numpy": np.__version__,
        "opencv": cv2.__version__, "pandas": pd.__version__,
        "device": torch.cuda.get_device_name(0) if DEVICE == "0" else platform.processor() or "CPU",
        "repos": {k: v[1] for k, v in REPOS.items()}, "seed": SEED, "imgsz": IMG_SIZE, "batch": BATCH_SIZE,
        "epochs": EPOCHS, "train_fraction": TRAIN_FRACTION, "bonus_fraction": BONUS_FRACTION, "smoke": SMOKE,
    }
    (OUT_DIR / "versions.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


#2. Дані: VisDrone2019-DET -> формат YOLO (5 класів), сплити train / val / test-dev
def parse_annotations(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = [p for p in line.strip().split(",") if p != ""]
        if len(parts) >= 8:
            rows.append([int(float(v)) for v in parts[:8]])
    return np.array(rows, dtype=np.int64).reshape(-1, 8)


def convert_image(args):
    """Зафарбовує ignored regions (сірий 114) і пише YOLO-мітки. Повертає статистику об'єктів."""
    img_path, ann_path, out_img, out_lbl = args
    ann = parse_annotations(ann_path)
    with Image.open(img_path) as im:
        w, h = im.size
    ignored = ann[(ann[:, 5] == 0) | (ann[:, 4] == 0)]
    if len(ignored):
        img = imread(img_path)
        for x, y, bw, bh in ignored[:, :4]:
            img[max(y, 0):max(y + bh, 0), max(x, 0):max(x + bw, 0)] = 114
        imwrite(out_img, img)
    else:
        shutil.copy2(img_path, out_img)
    lines, stats = [], []
    for x, y, bw, bh, score, cat, _, _ in ann:
        if score != 1 or cat not in VISDRONE_TO_OURS:
            continue
        x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        c = VISDRONE_TO_OURS[cat]
        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        stats.append((c, (x2 - x1) * (y2 - y1)))
    Path(out_lbl).write_text("\n".join(lines) + ("\n" if lines else ""))
    return stats


def prepare_data():
    marker = DATA_DIR / ".prepared.json"
    if marker.exists():
        info = json.loads(marker.read_text(encoding="utf-8"))
        if info.get("train_fraction") != TRAIN_FRACTION or info.get("bonus_fraction") != BONUS_FRACTION:  # змінили частку train - перебудовуємо лише список
            all_train = sorted(str(p) for p in (DATA_DIR / "images" / "train").glob("*.jpg"))
            write_train_list(all_train, info)
            marker.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        return info
    rng = random.Random(SEED)
    stats = {}
    for split, folder in SPLITS.items():
        src = RAW_DIR / folder
        if not (src / "images").is_dir():
            zip_path = download(VISDRONE_URL.format(folder), RAW_DIR / f"{folder}.zip")
            print(f"Розпакування {zip_path.name} ...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(RAW_DIR)
        images = sorted((src / "images").glob("*.jpg"))
        if SMOKE:  # маленька підвибірка лише для перевірки коду
            images = rng.sample(images, {"train": 48, "val": 16, "test": 24}[split])
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        jobs = [(p, src / "annotations" / f"{p.stem}.txt", DATA_DIR / "images" / split / p.name,
                 DATA_DIR / "labels" / split / f"{p.stem}.txt") for p in images]
        print(f"Конвертація {split}: {len(jobs)} зображень ...")
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            per_image = list(pool.map(convert_image, jobs))
        stats[split] = [s for img_stats in per_image for s in img_stats]
        files = sorted(str(j[2]) for j in jobs)
        if split == "train":
            all_train = files
        else:
            (DATA_DIR / f"{split}_list.txt").write_text("\n".join(files) + "\n", encoding="utf-8")

    data = {"path": str(DATA_DIR), "train": str(DATA_DIR / "train_list.txt"), "val": str(DATA_DIR / "val_list.txt"),
            "test": str(DATA_DIR / "test_list.txt"), "nc": len(CLASSES), "names": CLASSES}
    DATA_YAML.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    BONUS_YAML.write_text(yaml.safe_dump(dict(data, train=str(DATA_DIR / "train_bonus_list.txt")), sort_keys=False,
                                         allow_unicode=True), encoding="utf-8")

    rows = []
    for split, objs in stats.items():
        for c, name in enumerate(CLASSES):
            areas = np.array([a for cc, a in objs if cc == c])
            rows.append(dict(split=split, cls=name, objects=len(areas), small=int((areas < 32 ** 2).sum()),
                             medium=int(((areas >= 32 ** 2) & (areas < 96 ** 2)).sum()), large=int((areas >= 96 ** 2).sum()),
                             median_side_px=float(np.sqrt(np.median(areas))) if len(areas) else 0.0))
    info = {"counts": {s: len(Path(DATA_DIR / f"{s}_list.txt").read_text().split("\n")) - 1 for s in ("val", "test")},
            "stats": rows}
    write_train_list(all_train, info)
    marker.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def write_train_list(all_train, info):
    """Список train (однаковий для всіх фреймворків); при TRAIN_FRACTION < 1 - випадкова підвибірка з SEED."""
    files = sorted(all_train)
    if TRAIN_FRACTION < 1:
        files = sorted(random.Random(SEED).sample(files, max(1, round(len(files) * TRAIN_FRACTION))))
    (DATA_DIR / "train_list.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    bonus = sorted(random.Random(SEED + 1).sample(sorted(all_train), max(1, round(len(all_train) * BONUS_FRACTION))))
    (DATA_DIR / "train_bonus_list.txt").write_text("\n".join(bonus) + "\n", encoding="utf-8")
    info["counts"]["train_bonus"] = len(bonus)
    info["bonus_fraction"] = BONUS_FRACTION
    info["train_fraction"] = TRAIN_FRACTION
    info["counts"]["train"] = len(files)
    info["counts"]["train_full"] = len(all_train)


def report_data(info):
    df = pd.DataFrame(info["stats"])
    df.to_csv(OUT_DIR / "dataset_stats.csv", index=False)
    c = info["counts"]
    tot = df.groupby("split")[["objects", "small", "medium", "large"]].sum()
    print(f"Класів: {len(CLASSES)} {CLASSES} | train/val/test: {c['train']}/{c['val']}/{c['test']} зображень "
          f"(train = {100 * info['train_fraction']:.0f}% від {c['train_full']})")
    for split, r in tot.iterrows():
        print(f"  {split}: об'єктів {r.objects}, дрібних {100 * r.small / max(r.objects, 1):.1f}%, "
              f"середніх {100 * r.medium / max(r.objects, 1):.1f}%, великих {100 * r.large / max(r.objects, 1):.1f}%")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    tr = df[df.split == "train"].set_index("cls")
    tr[["objects"]].plot.bar(ax=axes[0], legend=False, color="#4C72B0")
    axes[0].set_title("Кількість об'єктів за класами (train)")
    axes[0].set_xlabel("")
    (tr[["small", "medium", "large"]].div(tr["objects"].clip(lower=1), axis=0) * 100).plot.bar(ax=axes[1], stacked=True)
    axes[1].set_title("Розподіл розмірів об'єктів, % (COCO: small < 32², large > 96² пікс.)")
    axes[1].set_xlabel("")
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis="y", alpha=0.3)
    finish_fig(fig, "dataset_stats.png")


#3. Моделі та сторонні репозиторії (офіційні YOLOv5 і YOLOv7, зафіксовані коміти)
def ensure_repo(name):
    dest = REPOS_DIR / name
    owner_repo, sha = REPOS[name]
    if (dest / ".ready").exists():
        patch_repo(name, dest)
        return dest
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("git"):
        try:
            for cmd in (["git", "init", "-q", str(dest)],
                        ["git", "-C", str(dest), "fetch", "-q", "--depth", "1", f"https://github.com/{owner_repo}", sha],
                        ["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"]):
                subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, OSError):
            shutil.rmtree(dest, ignore_errors=True)
    if not (dest / "train.py").exists():
        zip_path = download(f"https://github.com/{owner_repo}/archive/{sha}.zip", REPOS_DIR / f"{name}-{sha[:7]}.zip")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(REPOS_DIR)
        shutil.rmtree(dest, ignore_errors=True)
        (REPOS_DIR / f"{owner_repo.split('/')[1]}-{sha}").rename(dest)
    patch_repo(name, dest)
    (dest / ".ready").write_text(sha)
    return dest


def patch_repo(name, dest):
    """Сумісність коду YOLOv7 (2022 р.) з NumPy 2.x; ідемпотентно, логіку моделі не змінює."""
    if name != "yolov7":
        return
    fixes = {"utils/datasets.py": [(".astype(np.int)", ".astype(int)")],
             "utils/metrics.py": [("np.trapz(", "(np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(")],
             # AMD ROCm (Windows): cudnn.benchmark запускає вичерпний пошук ядер MIOpen, який зависає
             # з помилкою EvaluateInvokers; вимикаємо лише бенчмарк (на результати навчання не впливає).
             "utils/torch_utils.py": [("cudnn.benchmark, cudnn.deterministic = True, False",
                                       "cudnn.benchmark, cudnn.deterministic = False, False")]}
    for rel, pairs in fixes.items():
        p = dest / rel
        text = new = p.read_text(encoding="utf-8")
        for old, repl in pairs:
            if repl not in new:
                new = new.replace(old, repl)
        if new != text:
            p.write_text(new, encoding="utf-8")


def weights(name):
    return download(WEIGHT_URLS[name], WEIGHTS_DIR / name)


#4. Навчання
def run_dir(run_id):
    return RUNS_DIR / run_id


def is_done(run_id):
    return (run_dir(run_id) / "done.json").exists()


def done_info(run_id):
    return json.loads((run_dir(run_id) / "done.json").read_text(encoding="utf-8"))


def mark_done(run_id, run, train_time, epochs):
    d = run_dir(run_id)
    info = dict(id=run_id, framework=run["framework"], label=run["label"], best=str(d / "weights" / "best.pt"),
                train_time_s=train_time, epochs=epochs)
    (d / "done.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


def timed(run_id, fn):
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    t_file = d / "timing.json"
    spent = json.loads(t_file.read_text()) if t_file.exists() else 0.0
    start = time.perf_counter()
    try:
        fn()
    finally:
        spent += time.perf_counter() - start
        t_file.write_text(json.dumps(spent))
    return spent


def train_ultralytics(run_id, run, epochs, model_path, extra=None):
    from ultralytics import YOLO
    last = run_dir(run_id) / "weights" / "last.pt"
    if last.exists():
        print(f"[{run_id}] продовження з {last}")
        YOLO(str(last)).train(resume=True)
        return
    kw = dict(data=str(run.get("data", DATA_YAML)), epochs=epochs, imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE, workers=WORKERS,
              seed=SEED, deterministic=True, optimizer="SGD", close_mosaic=0, patience=PATIENCE,
              project=str(RUNS_DIR), name=run_id, exist_ok=True, plots=True, verbose=False)
    kw.update(HYP)
    kw.update(run.get("hyp", {}))
    kw.update(extra or {})
    YOLO(str(model_path)).train(**kw)


def repo_env():
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
               WANDB_MODE="disabled", WANDB_DISABLED="true", GIT_PYTHON_REFRESH="quiet",
               YOLO_AUTOINSTALL="false")
    # AMD ROCm: YOLOv7 вмикає cudnn.benchmark, і MIOpen починає вичерпний пошук ядер для кожної згортки
    # (на Windows він зависає з помилкою EvaluateInvokers). FAST - готові евристики без бенчмарку.
    env.setdefault("MIOPEN_FIND_MODE", "FAST")
    return env


def write_hyp(run_id, base_file, overrides):
    hyp = yaml.safe_load(Path(base_file).read_text(encoding="utf-8"))
    hyp.update(overrides)
    path = run_dir(run_id) / "hyp_used.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(hyp, sort_keys=False), encoding="utf-8")
    return path


def clear_label_caches():
    """У трьох фреймворків різні формати *.cache з однаковою назвою - видаляємо перед запуском."""
    for p in (DATA_DIR / "labels").glob("*.cache"):
        p.unlink()
    for p in DATA_DIR.glob("*.cache"):
        p.unlink()


def train_yolov5(run_id, run, epochs):
    repo = ensure_repo("yolov5")
    last = run_dir(run_id) / "weights" / "last.pt"
    if last.exists():
        cmd = [sys.executable, "train.py", "--resume", str(last)]
    else:
        hyp_keys = {k: v for k, v in HYP.items() if k not in ("warmup_epochs",)}
        hyp = write_hyp(run_id, repo / "data" / "hyps" / "hyp.scratch-low.yaml", dict(hyp_keys, warmup_epochs=HYP["warmup_epochs"]))
        cmd = [sys.executable, "train.py", "--data", str(DATA_YAML), "--weights", str(weights("yolov5s.pt")),
               "--hyp", str(hyp), "--img", str(IMG_SIZE), "--batch-size", str(BATCH_SIZE), "--epochs", str(epochs),
               "--device", DEVICE, "--workers", str(WORKERS), "--project", str(RUNS_DIR), "--name", run_id,
               "--exist-ok", "--seed", str(SEED), "--optimizer", "SGD", "--patience", str(PATIENCE)]
    clear_label_caches()
    subprocess.run(cmd, cwd=repo, env=repo_env(), check=True)


def train_yolov7(run_id, run, epochs):
    repo = ensure_repo("yolov7")
    last = run_dir(run_id) / "weights" / "last.pt"
    if last.exists():
        cmd = [sys.executable, "train.py", "--resume", str(last)]
    else:
        # Ті самі аугментації, що й у YOLOv5/YOLOv8 (вимикаємо mixup/paste_in, яких немає в інших моделях)
        hyp = write_hyp(run_id, repo / "data" / "hyp.scratch.tiny.yaml",
                        dict(HYP, mixup=0.0, copy_paste=0.0, paste_in=0.0))
        cmd = [sys.executable, "train.py", "--weights", str(weights("yolov7-tiny.pt")),
               "--cfg", "cfg/training/yolov7-tiny.yaml", "--data", str(DATA_YAML), "--hyp", str(hyp),
               "--epochs", str(epochs), "--batch-size", str(BATCH_SIZE), "--img-size", str(IMG_SIZE), str(IMG_SIZE),
               "--device", DEVICE, "--workers", str(WORKERS), "--project", str(RUNS_DIR), "--name", run_id,
               "--exist-ok", "--linear-lr"]
    clear_label_caches()
    try:
        subprocess.run(cmd, cwd=repo, env=repo_env(), check=True)
    finally:
        clear_label_caches()


MAIN_RUNS = [
    dict(id="yolov8n", label="YOLOv8n", framework="ultralytics", weights="yolov8n.pt"),
    dict(id="yolov5s", label="YOLOv5s", framework="yolov5", weights="yolov5s.pt"),
    dict(id="yolov7_tiny", label="YOLOv7-tiny", framework="yolov7", weights="yolov7-tiny.pt", optional=True),
]
BONUS_RUNS = [
    dict(id="yolov8n_ab", label="YOLOv8n (база абляцій)", framework="ultralytics", weights="yolov8n.pt"),
    dict(id="yolov8n_noaug", label="YOLOv8n (без аугментацій)", framework="ultralytics", weights="yolov8n.pt", hyp=NO_AUG),
    dict(id="yolov8n_cos", label="YOLOv8n (Cosine LR)", framework="ultralytics", weights="yolov8n.pt", hyp=dict(cos_lr=True)),
    dict(id="yolov8n_mixcut", label="YOLOv8n (MixUp + CutMix)", framework="ultralytics", weights="yolov8n.pt",
         hyp=dict(mixup=0.15, cutmix=0.15)),
    dict(id="yolov8n_freeze", label="YOLOv8n (freeze → unfreeze)", framework="ultralytics", weights="yolov8n.pt",
         two_stage=True),
]
for _r in BONUS_RUNS:  # абляції - на однаковій підвибірці train
    _r["data"] = BONUS_YAML


def train_run(run):
    run_id = run["id"]
    if is_done(run_id):
        print(f"[{run_id}] вже навчено -> {run_dir(run_id)}")
        return done_info(run_id)
    print(f"\n===== Навчання {run['label']} ({run_id}) | device={DEVICE} | epochs={run.get('epochs', EPOCHS)} =====")
    set_seed(SEED)
    if run.get("two_stage"):  # бонус: спершу заморожений backbone (10 шарів YOLOv8), потім донавчання всієї мережі
        s1 = dict(run, id=f"{run_id}_stage1", label=run["label"] + " stage1")
        if not is_done(s1["id"]):
            t = timed(s1["id"], lambda: train_ultralytics(s1["id"], s1, FREEZE_EPOCHS, weights(run["weights"]), dict(freeze=10)))
            mark_done(s1["id"], s1, t, FREEZE_EPOCHS)
        t = timed(run_id, lambda: train_ultralytics(run_id, run, EPOCHS - FREEZE_EPOCHS,
                                                    run_dir(s1["id"]) / "weights" / "last.pt", dict(warmup_epochs=0.0)))
        mark_done(run_id, run, t + done_info(s1["id"])["train_time_s"], EPOCHS)
    elif run["framework"] == "ultralytics":
        t = timed(run_id, lambda: train_ultralytics(run_id, run, run.get("epochs", EPOCHS), weights(run["weights"]),
                                                    run.get("extra")))
        mark_done(run_id, run, t, run.get("epochs", EPOCHS))
    elif run["framework"] == "yolov5":
        t = timed(run_id, lambda: train_yolov5(run_id, run, EPOCHS))
        mark_done(run_id, run, t, EPOCHS)
    elif run["framework"] == "yolov7":
        t = timed(run_id, lambda: train_yolov7(run_id, run, EPOCHS))
        mark_done(run_id, run, t, EPOCHS)
    return done_info(run_id)


def read_history(run_id, framework):
    """Уніфіковані криві: train/val box loss, val mAP@0.5, val mAP@0.5:0.95 по епохах."""
    d = run_dir(run_id)
    if framework == "yolov7":
        rows = []
        for line in (d / "results.txt").read_text().splitlines():
            p = line.split()
            if len(p) >= 15:
                rows.append(dict(epoch=int(p[0].split("/")[0]) + 1, train_box=float(p[2]), map50=float(p[10]),
                                 map5095=float(p[11]), val_box=float(p[12])))
        return pd.DataFrame(rows)
    df = pd.read_csv(d / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    if framework == "ultralytics":
        out = pd.DataFrame(dict(epoch=df["epoch"], train_box=df["train/box_loss"], val_box=df["val/box_loss"],
                                map50=df["metrics/mAP50(B)"], map5095=df["metrics/mAP50-95(B)"]))
        if df["epoch"].min() == 0:
            out["epoch"] += 1
        return out
    return pd.DataFrame(dict(epoch=df["epoch"] + 1, train_box=df["train/box_loss"], val_box=df["val/box_loss"],
                             map50=df["metrics/mAP_0.5"], map5095=df["metrics/mAP_0.5:0.95"]))


def history_of(run):
    if run.get("two_stage"):
        h1 = read_history(f"{run['id']}_stage1", "ultralytics")
        h2 = read_history(run["id"], "ultralytics")
        h2["epoch"] += h1["epoch"].max()
        return pd.concat([h1, h2], ignore_index=True)
    return read_history(run["id"], run["framework"])


#5. Оцінка (однаковий оцінювач COCOeval для всіх моделей)
def coco_backend():
    """faster-coco-eval - C++-реалізація COCOeval з ідентичними результатами. Еталонний pycocotools
    рахує зіставлення в циклі Python і на VisDrone (сотні об'єктів і до 1000 детекцій на кадр) працює годинами."""
    try:
        from faster_coco_eval import COCO, COCOeval_faster as COCOeval
    except ImportError:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    return COCO, COCOeval


def build_gt(split, limit=None):
    COCO, _ = coco_backend()
    files = Path(DATA_DIR / f"{split}_list.txt").read_text(encoding="utf-8").split("\n")
    files = [Path(f) for f in files if f.strip()][:limit]
    images, anns = [], []
    for i, f in enumerate(files):
        with Image.open(f) as im:
            w, h = im.size
        images.append(dict(id=i, file_name=str(f), width=w, height=h))
        lbl = DATA_DIR / "labels" / split / f"{f.stem}.txt"
        for line in lbl.read_text().splitlines():
            c, cx, cy, bw, bh = map(float, line.split())
            bw, bh = bw * w, bh * h
            anns.append(dict(id=len(anns) + 1, image_id=i, category_id=int(c), bbox=[cx * w - bw / 2, cy * h - bh / 2, bw, bh],
                             area=bw * bh, iscrowd=0))
    coco = COCO()
    coco.dataset = dict(images=images, annotations=anns, categories=[dict(id=i, name=n) for i, n in enumerate(CLASSES)])
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco, files


def predict_files(det, files, size=IMG_SIZE, sliced=False, iou=EVAL_IOU, batch=8):
    preds = []
    for k in range(0, len(files), 1 if sliced else batch):
        imgs = [imread(f) for f in files[k:k + (1 if sliced else batch)]]
        if sliced:
            preds.append(predict_sliced(det, imgs[0], SAHI_TILE, SAHI_OVERLAP, EVAL_CONF, iou, MAX_DET))
        else:
            preds += det.predict(imgs, size, EVAL_CONF, iou, MAX_DET)
    return preds


def coco_metrics(coco, preds):
    _, COCOeval = coco_backend()
    rows = [np.column_stack([np.full(len(p), i), p[:, 0], p[:, 1], p[:, 2] - p[:, 0], p[:, 3] - p[:, 1], p[:, 4], p[:, 5]])
            for i, p in enumerate(preds) if len(p)]
    if not rows:
        return dict(map=0, map50=0, map75=0, ap_s=0, ap_m=0, ap_l=0, ar=0, ar_s=0, per_class={}, pr={})
    dets = np.concatenate(rows).astype(np.float64)  # [image_id, x, y, w, h, score, class]
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(dets)
        ev = COCOeval(coco, dt, "bbox")
        ev.params.maxDets = [1, 100, MAX_DET]   # VisDrone: сотні об'єктів на кадр
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval["precision"]  # [T, R, K, A, M]; A: all, small, medium, large; M: maxDets
    rec = ev.eval["recall"]      # [T, K, A, M]

    def mean_valid(x):
        x = np.asarray(x)
        return float(x[x > -1].mean()) if (x > -1).any() else 0.0

    per_class, pr = {}, {}
    for k, name in enumerate(CLASSES):
        p_all, p50 = prec[:, :, k, 0, -1], prec[0, :, k, 0, -1]
        per_class[name] = dict(ap50=mean_valid(p50) if (p50 > -1).any() else float("nan"),
                               ap=mean_valid(p_all) if (p_all > -1).any() else float("nan"))
        pr[name] = np.where(p50 > -1, p50, np.nan).tolist()
    t50, t75 = 0, int(np.argmin(np.abs(ev.params.iouThrs - 0.75)))
    return dict(map=mean_valid(prec[:, :, :, 0, -1]), map50=mean_valid(prec[t50, :, :, 0, -1]),
                map75=mean_valid(prec[t75, :, :, 0, -1]), ap_s=mean_valid(prec[:, :, :, 1, -1]),
                ap_m=mean_valid(prec[:, :, :, 2, -1]), ap_l=mean_valid(prec[:, :, :, 3, -1]),
                ar=mean_valid(rec[:, :, 0, -1]), ar_s=mean_valid(rec[:, :, 1, -1]), per_class=per_class, pr=pr)


def match_counts(coco, preds, conf=DEPLOY_CONF, iou_thr=0.5):
    """Confusion matrix (nc+1)x(nc+1) з фоном, як у валідаторах YOLO: рядки - істина, стовпці - передбачення."""
    nc = len(CLASSES)
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    for i, p in enumerate(preds):
        anns = coco.imgToAnns.get(i, [])
        gt = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]).reshape(-1, 4)
        gc = np.array([a["category_id"] for a in anns], dtype=int)
        p = p[p[:, 4] >= conf]
        if not len(gt):
            for c in p[:, 5].astype(int):
                cm[nc, c] += 1
            continue
        if not len(p):
            for c in gc:
                cm[c, nc] += 1
            continue
        ious = torchvision.ops.box_iou(torch.tensor(gt, dtype=torch.float32), torch.tensor(p[:, :4], dtype=torch.float32)).numpy()
        gi, di = np.nonzero(ious > iou_thr)
        matches = np.stack([gi, di, ious[gi, di]], 1) if len(gi) else np.zeros((0, 3))
        if len(matches):
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        mg, md = set(matches[:, 0].astype(int)), set(matches[:, 1].astype(int))
        for g, d, _ in matches:
            cm[gc[int(g)], int(p[int(d), 5])] += 1
        for g in range(len(gt)):
            if g not in mg:
                cm[gc[g], nc] += 1
        for d in range(len(p)):
            if d not in md:
                cm[nc, int(p[d, 5])] += 1
    tp = np.diag(cm)[:nc]
    fp = cm[:, :nc].sum(0) - tp
    fn = cm[:nc, :].sum(1) - tp
    return cm, tp, fp, fn


def model_complexity(det):
    from torch.utils.flop_counter import FlopCounterMode
    net = det.net
    params = sum(p.numel() for p in net.parameters()) / 1e6
    dev = det.device
    det.to("cpu")
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        det.raw(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
    det.to(dev)
    return params, counter.get_total_flops() / 1e9  # GFLOPs (1 MAC = 2 FLOPs, як у звітах YOLO)


def measure_speed(det, device, imgs, size=IMG_SIZE, warmup=5, runs=50):
    """Медіанна latency (мс) forward-проходу та повного конвеєра (препроцесинг + forward + NMS), batch=1."""
    det.to(device)
    for im in imgs[:warmup]:
        det.predict([im], size, DEPLOY_CONF, 0.6)
    inf, total = [], []
    for k in range(runs):
        _, t = det.predict([imgs[k % len(imgs)]], size, DEPLOY_CONF, 0.6, return_time=True)
        inf.append(t["infer"])
        total.append(t["pre"] + t["infer"] + t["post"])
    det.to(TORCH_DEVICE)
    return float(np.median(inf)), float(np.median(total))


def cached(name, fn):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = fn()
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def evaluate_run(run, info, coco_test, test_files, coco_val, val_files, speed_imgs):
    det = Detector(run["framework"], info["best"], TORCH_DEVICE)
    preds_test = predict_files(det, test_files)
    preds_val = predict_files(det, val_files)
    m_test, m_val = coco_metrics(coco_test, preds_test), coco_metrics(coco_val, preds_val)
    cm, tp, fp, fn = match_counts(coco_test, preds_test)
    params, gflops = model_complexity(det)
    lat_cpu, e2e_cpu = measure_speed(det, torch.device("cpu"), speed_imgs)
    lat_gpu, e2e_gpu = measure_speed(det, torch.device("cuda:0"), speed_imgs) if DEVICE == "0" else (float("nan"),) * 2
    res = dict(test=m_test, val=m_val, cm=cm.tolist(), tp=tp.tolist(), fp=fp.tolist(), fn=fn.tolist(),
               params=params, gflops=gflops, lat_cpu=lat_cpu, fps_cpu=1000 / e2e_cpu, lat_gpu=lat_gpu,
               fps_gpu=1000 / e2e_gpu if e2e_gpu == e2e_gpu else float("nan"),
               weights_mb=Path(info["best"]).stat().st_size / 2**20, epoch_time=info["train_time_s"] / info["epochs"])
    del det
    return res


def plot_curves(runs, histories, name, title_suffix=""):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # Box loss YOLOv8 (CIoU + DFL) і YOLOv5/v7 (CIoU з іншим коефіцієнтом) мають різний масштаб,
    # тому для порівняння динаміки loss нормуємо на значення 1-ї епохи.
    panels = [("train_box", "Train box loss (відносно 1-ї епохи)"), ("val_box", "Val box loss (відносно 1-ї епохи)"),
              ("map50", "Val mAP@0.5"), ("map5095", "Val mAP@0.5:0.95")]
    for ax, (col, title) in zip(axes.ravel(), panels):
        for run in runs:
            h = histories[run["id"]]
            y = h[col] / h[col].iloc[0] if "box" in col else h[col]
            ax.plot(h["epoch"], y, marker="o", ms=3, label=run["label"])
        ax.set_title(title + title_suffix)
        ax.set_xlabel("Епоха")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    finish_fig(fig, name)


def plot_pr(runs, evals):
    recall = np.linspace(0, 1, 101)
    fig, axes = make_grid(len(runs) + 1, 2, 6.5, 5)
    for ax, run in zip(axes, runs):
        pr = evals[run["id"]]["test"]["pr"]
        for name in CLASSES:
            ax.plot(recall, pr[name], lw=1.2, label=f"{name} AP50={evals[run['id']]['test']['per_class'][name]['ap50']:.3f}")
        ax.set_title(f"{run['label']}: PR-криві (IoU=0.5, test)")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes[len(runs)]
    for run in runs:
        mean_p = np.nanmean(np.array([evals[run["id"]]["test"]["pr"][n] for n in CLASSES], dtype=float), axis=0)
        ax.plot(recall, mean_p, lw=2, label=f"{run['label']} mAP50={evals[run['id']]['test']['map50']:.3f}")
    ax.set_title("Усереднені PR-криві (IoU=0.5, test)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish_fig(fig, "pr_curves.png")


def plot_confusion(runs, evals):
    labels = CLASSES + ["background"]
    fig, axes = make_grid(len(runs), len(runs), 6.5, 5.5)
    for ax, run in zip(axes, runs):
        cm = np.array(evals[run["id"]]["cm"], dtype=float)
        norm = cm / np.clip(cm.sum(0, keepdims=True), 1, None)  # нормування по стовпцях (передбаченнях), як в Ultralytics
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(f"{run['label']} | conf≥{DEPLOY_CONF}, IoU≥0.5")
        ax.set_xlabel("Передбачений клас")
        ax.set_ylabel("Істинний клас")
    finish_fig(fig, "confusion_matrices.png")


#6. Бонуси: EigenCAM (аналог Grad-CAM для детекторів), SAHI, роздільна здатність, тюнінг
def eigencam(det, img, size=IMG_SIZE):
    """EigenCAM: перша головна компонента активацій шару P3 (stride 8), що подає ознаки в голову детекції."""
    layer = det.net.model[det.detect_layer.f[0]]
    store = {}
    hook = layer.register_forward_hook(lambda m, i, o: store.update(a=o))
    boxed, r, (px, py) = letterbox(img, size)
    with torch.no_grad():
        det.raw(to_tensor([boxed], det.device))
    hook.remove()
    a = store["a"][0].float().cpu()
    c, h, w = a.shape
    m = a.reshape(c, -1).T
    m = m - m.mean(0)
    _, _, vt = torch.linalg.svd(m, full_matrices=False)
    proj = m @ vt[0]
    if torch.dot(proj, a.abs().mean(0).flatten()) < 0:  # SVD визначає знак компоненти з точністю до ±1
        proj = -proj
    cam = proj.reshape(h, w).clamp(min=0).numpy()
    cam = cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR)
    nh, nw = round(img.shape[0] * r), round(img.shape[1] * r)
    cam = cv2.resize(cam[py:py + nh, px:px + nw], (img.shape[1], img.shape[0]))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def plot_eigencam(runs, infos, coco_test, test_files):
    counts = [sum(1 for a in coco_test.imgToAnns.get(i, []) if a["area"] < 32 ** 2) for i in range(len(test_files))]
    pool = [i for i in np.argsort(counts)[::-1][:max(10, len(counts) // 10)]]
    picks = np.random.RandomState(SEED).choice(pool, 2, replace=False)
    fig, axes = make_grid((1 + len(runs)) * 2, 2, 7, 4.6)
    imgs = [imread(test_files[i]) for i in picks]
    for col, (i, img) in enumerate(zip(picks, imgs)):
        axes[col].imshow(img[:, :, ::-1])
        axes[col].set_title(f"Оригінал (test): {test_files[i].name}\nдрібних об'єктів: {counts[i]}", fontsize=9)
        axes[col].axis("off")
    for row, run in enumerate(runs, start=1):
        det = Detector(run["framework"], infos[run["id"]]["best"], TORCH_DEVICE)
        for col, img in enumerate(imgs):
            cam = eigencam(det, img)
            pred = det.predict([img], IMG_SIZE, DEPLOY_CONF, 0.6)[0]
            ax = axes[row * 2 + col]
            ax.imshow(draw(img, pred)[:, :, ::-1])
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.set_title(f"{run['label']}: EigenCAM (P3) + детекції conf≥{DEPLOY_CONF} ({int((pred[:, 4] >= DEPLOY_CONF).sum())})", fontsize=9)
            ax.axis("off")
        del det
    finish_fig(fig, "eigencam.png")


def tune_lr0():
    """Бонус: легке тюнінгування lr0 короткими запусками YOLOv8n на частині train; критерій - val mAP@0.5:0.95."""
    rows = []
    for lr in TUNE_LR0:
        run = dict(id=f"tune_lr0_{lr}", label=f"YOLOv8n lr0={lr}", framework="ultralytics", weights="yolov8n.pt",
                   epochs=TUNE_EPOCHS, hyp=dict(lr0=lr), data=BONUS_YAML, extra=dict(patience=TUNE_EPOCHS, plots=False))
        info = train_run(run)
        h = read_history(run["id"], "ultralytics")
        rows.append({"lr0": lr, "епох": TUNE_EPOCHS, "частка train": BONUS_FRACTION, "Val mAP@0.5": h["map50"].max(),
                     "Val mAP@0.5:0.95": h["map5095"].max(), "Час (с)": info["train_time_s"]})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(OUT_DIR / "tuning_lr0.csv", index=False)
    return df


def tune_nms(run, info, coco_val, val_files):
    """Бонус: підбір IoU-порогу NMS (за val mAP@0.5:0.95) і порогу впевненості (за val F1)."""
    det = Detector(run["framework"], info["best"], TORCH_DEVICE)
    rows = []
    for iou in NMS_IOU_GRID:
        preds = predict_files(det, val_files, iou=iou)
        m = coco_metrics(coco_val, preds)
        rows.append({"NMS IoU": iou, "Val mAP@0.5": m["map50"], "Val mAP@0.5:0.95": m["map"]})
    nms_df = pd.DataFrame(rows).round(4)
    best_iou = float(nms_df.loc[nms_df["Val mAP@0.5:0.95"].idxmax(), "NMS IoU"])
    preds = predict_files(det, val_files, iou=best_iou)
    rows = []
    for conf in CONF_GRID:
        _, tp, fp, fn = match_counts(coco_val, preds, conf=conf)
        p, r = tp.sum() / max(tp.sum() + fp.sum(), 1), tp.sum() / max(tp.sum() + fn.sum(), 1)
        rows.append({"conf": conf, "Precision": p, "Recall": r, "F1": 2 * p * r / max(p + r, 1e-9)})
    conf_df = pd.DataFrame(rows).round(4)
    del det
    nms_df.to_csv(OUT_DIR / "tuning_nms_iou.csv", index=False)
    conf_df.to_csv(OUT_DIR / "tuning_conf.csv", index=False)
    return nms_df, conf_df, best_iou, float(conf_df.loc[conf_df["F1"].idxmax(), "conf"])


#7. Інференс на власних зображеннях / відео, GIF
def collect_media(folder, exts):
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in exts) if Path(folder).is_dir() else []


def save_gif(frames_bgr, path, fps=2, width=720):
    import imageio.v2 as imageio
    scaled = [cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1]))) for f in frames_bgr]
    height = max(f.shape[0] for f in scaled)
    frames = []
    for f in scaled:  # кадри різних пропорцій вирівнюємо до одного розміру (вимога формату GIF)
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        top = (height - f.shape[0]) // 2
        canvas[top:top + f.shape[0]] = f
        frames.append(canvas[:, :, ::-1])
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)  # duration у мс (imageio >= 2.28)


def process_video(det, src, size, max_frames=450):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():  # кирилиця в шляху - копіюємо у тимчасовий ASCII-файл
        tmp = WORK_ROOT / f"tmp_video{src.suffix}"
        shutil.copy2(src, tmp)
        cap = cv2.VideoCapture(str(tmp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames, out_frames = 0, []
    out_tmp = WORK_ROOT / f"det_{src.stem.encode('ascii', 'ignore').decode() or 'video'}.mp4"
    writer = None
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        vis = draw(frame, det.predict([frame], size, DEPLOY_CONF, 0.6)[0])
        if writer is None:
            writer = cv2.VideoWriter(str(out_tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        if frames % max(1, round(fps / 6)) == 0:
            out_frames.append(vis)
        frames += 1
    cap.release()
    if writer is not None:
        writer.release()
        shutil.move(str(out_tmp), OUT_DIR / f"video_{src.stem}_detections.mp4")
    if out_frames:
        save_gif(out_frames[:90], OUT_DIR / f"video_{src.stem}_detections.gif", fps=6)
    return frames


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    versions = save_versions()
    print("Device:", DEVICE, "|", versions["device"], "| torch", torch.__version__, "| ultralytics", versions["ultralytics"])
    print("Робочий каталог (дані, репозиторії, запуски):", WORK_ROOT)
    assert SMOKE or EPOCHS >= 10, "Методичка: не менше 10 епох"

    # 2. Дані
    data_info = prepare_data()
    report_data(data_info)

    # 4. Навчання
    main_runs = [r for r in MAIN_RUNS if RUN_THIRD_MODEL or not r.get("optional")]
    infos = {}
    for run in main_runs + (BONUS_RUNS if RUN_BONUS else []):
        try:
            infos[run["id"]] = train_run(run)
        except Exception as e:  # третя (необов'язкова) модель не повинна зупиняти лабораторну
            if not run.get("optional"):
                raise
            print(f"[{run['id']}] пропущено через помилку: {e}")
    main_runs = [r for r in main_runs if r["id"] in infos]
    bonus_runs = [r for r in BONUS_RUNS if r["id"] in infos]
    histories = {r["id"]: history_of(r) for r in main_runs + bonus_runs}
    plot_curves(main_runs, histories, "training_curves.png")
    if bonus_runs:
        plot_curves(bonus_runs, histories, "bonus_training_curves.png", " (YOLOv8n, абляції)")

    # 5. Оцінка
    coco_test, test_files = build_gt("test")
    coco_val, val_files = build_gt("val")
    speed_imgs = [imread(f) for f in test_files[:20]]
    evals = {}
    for run in main_runs + bonus_runs:
        print(f"Оцінка {run['label']} ...")
        evals[run["id"]] = cached(f"eval_{run['id']}", lambda run=run: evaluate_run(
            run, infos[run["id"]], coco_test, test_files, coco_val, val_files, speed_imgs))

    def row(run, e):
        return {"Модель": run["label"], "imgsz": IMG_SIZE, "Параметри (М)": e["params"], "FLOPs (Г)": e["gflops"],
                "mAP@0.5": e["test"]["map50"], "mAP@0.5:0.95": e["test"]["map"], "AP_small": e["test"]["ap_s"],
                "AP_medium": e["test"]["ap_m"], "AP_large": e["test"]["ap_l"], "FPS (GPU)": e["fps_gpu"],
                "FPS (CPU)": e["fps_cpu"], "Latency CPU (мс)": e["lat_cpu"], "Розмір ваг (МБ)": e["weights_mb"],
                "Час епохи (с)": e["epoch_time"], "Val mAP@0.5:0.95": e["val"]["map"]}

    table = pd.DataFrame([row(r, evals[r["id"]]) for r in main_runs]).round(3)
    table.to_csv(OUT_DIR / "comparison_table.csv", index=False)
    print("\nТаблиця порівняння (test-dev, однаковий оцінювач COCOeval):")
    print(table.to_string(index=False))
    if bonus_runs:
        bonus = pd.DataFrame([row(r, evals[r["id"]]) for r in bonus_runs]).round(3)
        bonus.to_csv(OUT_DIR / "bonus_runs_table.csv", index=False)
        print("\nБонусні запуски YOLOv8n:")
        print(bonus[["Модель", "mAP@0.5", "mAP@0.5:0.95", "AP_small", "Val mAP@0.5:0.95", "Час епохи (с)"]].to_string(index=False))

    per_class = pd.DataFrame([{"Модель": r["label"], "Клас": n, "AP@0.5": evals[r["id"]]["test"]["per_class"][n]["ap50"],
                               "AP@0.5:0.95": evals[r["id"]]["test"]["per_class"][n]["ap"],
                               "TP": evals[r["id"]]["tp"][k], "FP": evals[r["id"]]["fp"][k], "FN": evals[r["id"]]["fn"][k]}
                              for r in main_runs for k, n in enumerate(CLASSES)]).round(3)
    per_class.to_csv(OUT_DIR / "per_class_ap.csv", index=False)
    print("\nAP по класах та FP/FN (conf≥0.25, IoU≥0.5):")
    print(per_class.to_string(index=False))
    plot_pr(main_runs, evals)
    plot_confusion(main_runs, evals)

    # 6. Бонуси для дрібних об'єктів: більша роздільна здатність інференсу та SAHI
    small_rows = []
    for run in main_runs:
        e = evals[run["id"]]
        small_rows.append({"Модель": run["label"], "Режим": f"imgsz {IMG_SIZE}", "mAP@0.5": e["test"]["map50"],
                           "mAP@0.5:0.95": e["test"]["map"], "AP_small": e["test"]["ap_s"], "AR_small": e["test"]["ar_s"],
                           "FPS (CPU)": e["fps_cpu"]})

        def hi_res(run=run):
            det = Detector(run["framework"], infos[run["id"]]["best"], TORCH_DEVICE)
            m = coco_metrics(coco_test, predict_files(det, test_files, size=HIGH_RES))
            _, e2e = measure_speed(det, torch.device("cpu"), speed_imgs, size=HIGH_RES, runs=20)
            return dict(m=m, fps=1000 / e2e)
        hr = cached(f"hires_{run['id']}", hi_res)
        small_rows.append({"Модель": run["label"], "Режим": f"imgsz {HIGH_RES} (лише інференс)", "mAP@0.5": hr["m"]["map50"],
                           "mAP@0.5:0.95": hr["m"]["map"], "AP_small": hr["m"]["ap_s"], "AR_small": hr["m"]["ar_s"],
                           "FPS (CPU)": hr["fps"]})
        if run["id"] in SAHI_MODELS:
            def sahi(run=run):
                det = Detector(run["framework"], infos[run["id"]]["best"], TORCH_DEVICE)
                m = coco_metrics(coco_test, predict_files(det, test_files, sliced=True))
                det.to("cpu")
                t0 = time.perf_counter()
                for im in speed_imgs[:5]:
                    predict_sliced(det, im, SAHI_TILE, SAHI_OVERLAP, DEPLOY_CONF, 0.6)
                return dict(m=m, fps=5 / (time.perf_counter() - t0))
            sh = cached(f"sahi_{run['id']}", sahi)
            small_rows.append({"Модель": run["label"], "Режим": f"SAHI (тайли {SAHI_TILE}, перекриття {SAHI_OVERLAP})",
                               "mAP@0.5": sh["m"]["map50"], "mAP@0.5:0.95": sh["m"]["map"], "AP_small": sh["m"]["ap_s"],
                               "AR_small": sh["m"]["ar_s"], "FPS (CPU)": sh["fps"]})
    small = pd.DataFrame(small_rows).round(3)
    small.to_csv(OUT_DIR / "small_objects_table.csv", index=False)
    print("\nДрібні об'єкти: роздільна здатність і SAHI (test-dev):")
    print(small.to_string(index=False))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    piv = small.pivot(index="Модель", columns="Режим", values="AP_small").reindex([r["label"] for r in main_runs])
    piv.plot.bar(ax=ax)
    ax.set_title("AP_small (COCO, площа < 32² пікс.) на test-dev")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", alpha=0.3)
    finish_fig(fig, "small_objects.png")

    lr_df = tune_lr0() if RUN_BONUS else None
    if lr_df is not None:
        print("\nТюнінг lr0 (YOLOv8n, короткі запуски):")
        print(lr_df.to_string(index=False))
    best_id = max([r["id"] for r in main_runs + bonus_runs], key=lambda k: evals[k]["val"]["map"])  # вибір за val, не за test
    best_run = next(r for r in main_runs + bonus_runs if r["id"] == best_id)
    nms_df, conf_df, best_iou, best_conf = tune_nms(best_run, infos[best_id], coco_val, val_files)
    print(f"\nТюнінг NMS для {best_run['label']} (val):")
    print(nms_df.to_string(index=False))
    print(conf_df.to_string(index=False))
    det = Detector(best_run["framework"], infos[best_id]["best"], TORCH_DEVICE)
    tuned = cached(f"tuned_nms_{best_id}", lambda: coco_metrics(coco_test, predict_files(det, test_files, iou=best_iou)))

    plot_eigencam(main_runs, infos, coco_test, test_files)

    # 7. Збереження найкращої моделі та інференс на власних зображеннях / відео
    best_path = OUT_DIR / f"best_model_{best_id}.pt"
    shutil.copy2(infos[best_id]["best"], best_path)
    meta = dict(run_id=best_id, label=best_run["label"], framework=best_run["framework"], weights=best_path.name,
                imgsz=IMG_SIZE, conf=best_conf, iou=best_iou, classes=CLASSES)
    (OUT_DIR / "best_model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nНайкраща модель (за val mAP@0.5:0.95): {best_run['label']} -> {best_path}")

    test_pick = np.random.RandomState(SEED).choice(len(test_files), min(8, len(test_files)), replace=False)
    test_vis = [draw(imread(test_files[i]), det.predict([imread(test_files[i])], IMG_SIZE, best_conf, best_iou)[0], conf=best_conf)
                for i in test_pick]
    fig, axes = make_grid(len(test_vis), 2, 8, 4.8)
    for ax, i, vis in zip(axes, test_pick, test_vis):
        ax.imshow(vis[:, :, ::-1])
        ax.set_title(f"{best_run['label']} | {test_files[i].name}", fontsize=9)
        ax.axis("off")
    finish_fig(fig, "test_detections.png")
    save_gif(test_vis, OUT_DIR / "test_detections.gif", fps=1)

    own = collect_media(MY_IMAGES_DIR, {".jpg", ".jpeg", ".png", ".webp", ".bmp"})[:10]
    own_vis = []
    if not own:
        MY_IMAGES_DIR.mkdir(exist_ok=True)
        print(f"Папка {MY_IMAGES_DIR.name}/ порожня: додайте 5-10 зображень (вид згори: вулиця, парковка) і запустіть run_inference.py.")
    else:
        (OUT_DIR / "own_detections").mkdir(exist_ok=True)
        fig, axes = make_grid(len(own), 2, 8, 5.5)
        for ax, path in zip(axes, own):
            img = imread(path)
            pred = det.predict([img], IMG_SIZE, best_conf, best_iou)[0]
            vis = draw(img, pred, conf=best_conf)
            own_vis.append(vis)
            imwrite(OUT_DIR / "own_detections" / f"{path.stem}.jpg", vis)
            counts = {CLASSES[c]: int((pred[:, 5] == c).sum()) for c in range(len(CLASSES)) if (pred[:, 5] == c).any()}
            ax.imshow(vis[:, :, ::-1])
            ax.set_title(f"{path.name}: {counts}", fontsize=9)
            ax.axis("off")
            print(path.name, "->", counts)
        finish_fig(fig, "own_images.png")
        save_gif(own_vis, OUT_DIR / "own_detections.gif", fps=1)
    for video in collect_media(MY_VIDEO_DIR, {".mp4", ".avi", ".mov", ".mkv"}):
        n = process_video(det, video, IMG_SIZE)
        print(f"Відео {video.name}: оброблено {n} кадрів -> outputs/video_{video.stem}_detections.mp4/.gif")

    # 8. Висновки
    lines = []
    pct = lambda v: f"{100 * v:.1f}%"
    t = {k: v["test"] for k, v in evals.items()}
    top = max(main_runs, key=lambda r: t[r["id"]]["map"])
    lines.append(f"Найкраща з основних моделей за test mAP@0.5:0.95: {top['label']} (mAP@0.5 {pct(t[top['id']]['map50'])}, "
                 f"mAP@0.5:0.95 {pct(t[top['id']]['map'])}, AP_small {pct(t[top['id']]['ap_s'])}).")
    if "yolov8n" in evals and "yolov5s" in evals:
        a, b = evals["yolov8n"], evals["yolov5s"]
        lines.append(f"YOLOv8n vs YOLOv5s: параметри {a['params']:.2f}M vs {b['params']:.2f}M, FLOPs {a['gflops']:.1f}G vs {b['gflops']:.1f}G, "
                     f"ваги {a['weights_mb']:.1f} vs {b['weights_mb']:.1f} МБ, FPS CPU {a['fps_cpu']:.1f} vs {b['fps_cpu']:.1f}, "
                     f"mAP@0.5 {pct(a['test']['map50'])} vs {pct(b['test']['map50'])}, mAP@0.5:0.95 {pct(a['test']['map'])} vs "
                     f"{pct(b['test']['map'])}, AP_small {pct(a['test']['ap_s'])} vs {pct(b['test']['ap_s'])}.")
    if "yolov7_tiny" in evals:
        c = evals["yolov7_tiny"]
        lines.append(f"YOLOv7-tiny: параметри {c['params']:.2f}M, FLOPs {c['gflops']:.1f}G, FPS CPU {c['fps_cpu']:.1f}, "
                     f"mAP@0.5 {pct(c['test']['map50'])}, mAP@0.5:0.95 {pct(c['test']['map'])}, AP_small {pct(c['test']['ap_s'])}.")
    for rid, what in (("yolov8n_noaug", "Вплив аугментацій"), ("yolov8n_cos", "Cosine LR vs лінійний"),
                      ("yolov8n_mixcut", "MixUp + CutMix"), ("yolov8n_freeze", "Freeze → unfreeze")):
        if rid in evals and "yolov8n_ab" in evals:
            a, b = t["yolov8n_ab"], t[rid]
            lines.append(f"{what} (YOLOv8n): mAP@0.5:0.95 {pct(a['map'])} (базовий) vs {pct(b['map'])}, "
                         f"різниця {100 * (b['map'] - a['map']):+.1f} п.п.; AP_small {pct(a['ap_s'])} vs {pct(b['ap_s'])}.")
    for run in main_runs:
        base = small[(small["Модель"] == run["label"]) & (small["Режим"] == f"imgsz {IMG_SIZE}")].iloc[0]
        for _, r in small[(small["Модель"] == run["label"]) & (small["Режим"] != f"imgsz {IMG_SIZE}")].iterrows():
            lines.append(f"{run['label']}, {r['Режим']}: AP_small {pct(base['AP_small'])} -> {pct(r['AP_small'])} "
                         f"({100 * (r['AP_small'] - base['AP_small']):+.1f} п.п.), mAP@0.5:0.95 {pct(base['mAP@0.5:0.95'])} -> "
                         f"{pct(r['mAP@0.5:0.95'])}, FPS CPU {base['FPS (CPU)']:.1f} -> {r['FPS (CPU)']:.1f}.")
    if lr_df is not None:
        best_lr = lr_df.loc[lr_df["Val mAP@0.5:0.95"].idxmax()]
        lines.append(f"Тюнінг lr0: найкраще lr0={best_lr['lr0']} (val mAP@0.5:0.95 {pct(best_lr['Val mAP@0.5:0.95'])} за {TUNE_EPOCHS} епох).")
    lines.append(f"NMS для {best_run['label']}: оптимальний IoU={best_iou} (test mAP@0.5:0.95 {pct(t[best_id]['map'])} -> "
                 f"{pct(tuned['map'])}), оптимальний conf за F1 на val = {best_conf}.")
    worst = min(CLASSES, key=lambda n: t[top["id"]]["per_class"][n]["ap"])
    lines.append(f"Найскладніший клас для {top['label']}: {worst} (AP@0.5:0.95 {pct(t[top['id']]['per_class'][worst]['ap'])}).")
    print("\nВисновки:")
    for ln in lines:
        print(ln)
    (OUT_DIR / "conclusions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nУсі результати збережено в {OUT_DIR}")


if __name__ == "__main__":
    main()
