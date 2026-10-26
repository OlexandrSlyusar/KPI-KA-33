import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

CLASSES = ["pedestrian", "person", "car", "bus", "truck"]
COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207)]

WORK_ROOT = Path(os.environ.get("LAB2_WORK_ROOT", Path.home() / "omsm_lab2"))
if not str(WORK_ROOT).isascii():
    WORK_ROOT = Path("C:/omsm_lab2") if os.name == "nt" else Path("/tmp/omsm_lab2")
REPOS_DIR = WORK_ROOT / "third_party"
WEIGHTS_DIR = WORK_ROOT / "weights"
RUNS_DIR = WORK_ROOT / "runs"

REPOS = {
    "yolov5": ("ultralytics/yolov5", "af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce"),
    "yolov7": ("WongKinYiu/yolov7", "a207844b1ce82d204ab36d87d496728d3d2348e7"),
}


#1. Зображення
def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img, quality=95):
    ext = Path(path).suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"Не вдалося закодувати {path}")
    buf.tofile(str(path))


def letterbox(img, size, color=114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, (left, top)


def to_tensor(imgs_bgr, device):
    x = np.stack([im[:, :, ::-1].transpose(2, 0, 1) for im in imgs_bgr])
    return torch.from_numpy(np.ascontiguousarray(x)).to(device).float().div_(255)


#2. Моделі
def _purge_repo_modules():
    for name in list(sys.modules):
        if name in ("models", "utils") or name.startswith(("models.", "utils.")):
            del sys.modules[name]


class Detector:

    def __init__(self, framework, weights, device="cpu"):
        self.framework, self.weights, self.device = framework, Path(weights), torch.device(device)
        if framework == "ultralytics":
            from ultralytics import YOLO
            net = YOLO(str(weights)).model
            net = net.fuse() if hasattr(net, "fuse") else net
        elif framework in ("yolov5", "yolov7"):
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            repo = str(REPOS_DIR / framework)
            _purge_repo_modules()
            sys.path.insert(0, repo)
            try:
                from models.experimental import attempt_load
                if framework == "yolov5":
                    net = attempt_load(str(weights), device=torch.device("cpu"), fuse=True)
                else:
                    net = attempt_load(str(weights), map_location=torch.device("cpu"))
            finally:
                sys.path.remove(repo)
                _purge_repo_modules()
        else:
            raise ValueError(framework)
        self.net = net.float().eval().to(self.device)
        self._reset_grids()
        for p in self.net.parameters():
            if p.is_leaf:
                p.requires_grad_(False)

    def to(self, device):
        self.device = torch.device(device)
        self.net.to(self.device)
        self._reset_grids()
        return self

    def _reset_grids(self):
        m = self.detect_layer
        for attr in ("grid", "anchor_grid"):
            g = getattr(m, attr, None)
            if isinstance(g, list):
                setattr(m, attr, [torch.empty(0)] * len(g))

    @property
    def detect_layer(self):
        return self.net.model[-1]

    def raw(self, x):
        out = self.net(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    def decode(self, out):
        if self.framework == "ultralytics":
            out = out.transpose(1, 2)
            return out[..., :4], out[..., 4:]
        return out[..., :4], out[..., 5:] * out[..., 4:5]

    @torch.no_grad()
    def predict(self, imgs_bgr, size=640, conf=0.001, iou=0.6, max_det=1000, return_time=False):
        t0 = time.perf_counter()
        boxed = [letterbox(im, size) for im in imgs_bgr]
        x = to_tensor([b[0] for b in boxed], self.device)
        t1 = time.perf_counter()
        out = self.raw(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        xywh, scores = self.decode(out.float())
        results = []
        for i, (im, (_, r, (px, py))) in enumerate(zip(imgs_bgr, boxed)):
            det = nms(xywh[i], scores[i], conf, iou, max_det)
            det[:, [0, 2]] = ((det[:, [0, 2]] - px) / r).clamp(0, im.shape[1])
            det[:, [1, 3]] = ((det[:, [1, 3]] - py) / r).clamp(0, im.shape[0])
            results.append(det.cpu().numpy())
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        if return_time:
            return results, dict(pre=(t1 - t0) * 1e3, infer=(t2 - t1) * 1e3, post=(t3 - t2) * 1e3)
        return results


#3. NMS, SAHI, візуалізація
def xywh2xyxy(b):
    out = b.clone()
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def nms(xywh, scores, conf=0.001, iou=0.6, max_det=1000, max_candidates=30000):
    i, j = (scores > conf).nonzero(as_tuple=True)
    s = scores[i, j]
    if s.numel() > max_candidates:
        top = s.topk(max_candidates).indices
        i, j, s = i[top], j[top], s[top]
    boxes = xywh2xyxy(xywh[i])
    keep = torchvision.ops.batched_nms(boxes, s, j, iou)[:max_det]
    return torch.cat([boxes[keep], s[keep, None], j[keep, None].float()], 1)


def merge_nms(dets, iou=0.6, max_det=1000):
    dets = [d for d in dets if len(d)]
    if not dets:
        return np.zeros((0, 6), dtype=np.float32)
    d = torch.from_numpy(np.concatenate(dets).astype(np.float32))
    keep = torchvision.ops.batched_nms(d[:, :4], d[:, 4], d[:, 5].long(), iou)[:max_det]
    return d[keep].numpy()


def predict_sliced(det, img, size=640, overlap=0.2, conf=0.001, iou=0.6, max_det=1000):
    h, w = img.shape[:2]
    step = int(size * (1 - overlap))
    xs = list(range(0, max(w - size, 0) + 1, step)) or [0]
    ys = list(range(0, max(h - size, 0) + 1, step)) or [0]
    if xs[-1] + size < w:
        xs.append(w - size)
    if ys[-1] + size < h:
        ys.append(h - size)
    tiles, offsets = [], []
    for y0 in ys:
        for x0 in xs:
            tiles.append(img[y0:y0 + size, x0:x0 + size])
            offsets.append((x0, y0))
    outs = []
    for k in range(0, len(tiles), 8):
        for d, (x0, y0) in zip(det.predict(tiles[k:k + 8], size, conf, iou, max_det), offsets[k:k + 8]):
            d = d.copy()
            d[:, [0, 2]] += x0
            d[:, [1, 3]] += y0
            outs.append(d)
    outs.append(det.predict([img], size, conf, iou, max_det)[0])
    return merge_nms(outs, iou, max_det)


def draw(img, det, conf=0.25, names=CLASSES, thickness=None):
    img = img.copy()
    t = thickness or max(1, round(sum(img.shape[:2]) / 1200))
    for x1, y1, x2, y2, s, c in det[det[:, 4] >= conf] if len(det) else []:
        c = int(c)
        color = COLORS[c % len(COLORS)]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, t, cv2.LINE_AA)
        label = f"{names[c]} {s:.2f}"
        fs = 0.35 * t
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y_text = max(int(y1), th + 2)
        cv2.rectangle(img, (int(x1), y_text - th - 2), (int(x1) + tw, y_text), color, -1)
        cv2.putText(img, label, (int(x1), y_text - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img
