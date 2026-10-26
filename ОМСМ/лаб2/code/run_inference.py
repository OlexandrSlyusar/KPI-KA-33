"""Інференс найкращої навченої моделі на власних зображеннях (my_images/) і відео (my_video/).

Запускати після ОМСМ_лаба2.py (потрібні outputs/best_model.json та outputs/best_model_*.pt).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ОМСМ_лаба2 import (DEPLOY_CONF, IMG_SIZE, MY_IMAGES_DIR, MY_VIDEO_DIR, OUT_DIR, collect_media, make_grid,
                        process_video, save_gif)
from yolo_common import CLASSES, Detector, draw, imread, imwrite

if __name__ == "__main__":
    meta = json.loads((OUT_DIR / "best_model.json").read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Завантаження навченої моделі {meta['label']} ({meta['framework']}) ...")
    det = Detector(meta["framework"], OUT_DIR / meta["weights"], device)
    conf, iou = meta.get("conf", DEPLOY_CONF), meta.get("iou", 0.6)

    own = collect_media(MY_IMAGES_DIR, {".jpg", ".jpeg", ".png", ".webp", ".bmp"})[:10]
    if not own:
        print(f"Помилка: папка {MY_IMAGES_DIR} порожня!")
    else:
        (OUT_DIR / "own_detections").mkdir(parents=True, exist_ok=True)
        fig, axes = make_grid(len(own), 2, 8, 5.5)
        frames = []
        for ax, path in zip(axes, own):
            img = imread(path)
            pred = det.predict([img], IMG_SIZE, conf, iou)[0]
            vis = draw(img, pred, conf=conf)
            frames.append(vis)
            imwrite(OUT_DIR / "own_detections" / f"{path.stem}.jpg", vis)
            counts = {CLASSES[c]: int((pred[:, 5] == c).sum()) for c in range(len(CLASSES)) if (pred[:, 5] == c).any()}
            ax.imshow(vis[:, :, ::-1])
            ax.set_title(f"{path.name}: {counts}", fontsize=9)
            ax.axis("off")
            print(f"{path.name} -> {counts}")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "own_images.png", dpi=150)
        save_gif(frames, OUT_DIR / "own_detections.gif", fps=1)
        plt.close(fig)
        print("\nГотово! Картинку збережено: outputs/own_images.png, GIF: outputs/own_detections.gif")

    for video in collect_media(MY_VIDEO_DIR, {".mp4", ".avi", ".mov", ".mkv"}):
        n = process_video(det, video, IMG_SIZE)
        print(f"Відео {video.name}: {n} кадрів -> outputs/video_{video.stem}_detections.mp4 та .gif")
