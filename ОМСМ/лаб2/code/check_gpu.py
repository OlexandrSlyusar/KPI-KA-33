"""Перевірка середовища: PyTorch ROCm бачить Radeon і виконує згортку та NMS на GPU."""
import platform
import time

import torch
import torchvision

print("Python", platform.python_version(), "| torch", torch.__version__, "| torchvision", torchvision.__version__)
print("HIP (ROCm):", getattr(torch.version, "hip", None), "| GPU доступний:", torch.cuda.is_available())
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | пам'ять {p.total_memory / 2**30:.1f} ГБ")
    x = torch.randn(16, 3, 640, 640, device=dev)
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(dev)
    with torch.autocast("cuda"):
        y = conv(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        with torch.autocast("cuda"):
            y = conv(x)
    torch.cuda.synchronize()
    print(f"Conv2d 16x3x640x640 (AMP): {(time.perf_counter() - t) / 20 * 1000:.1f} мс")
    boxes = torch.rand(5000, 4, device=dev) * 600
    boxes[:, 2:] += boxes[:, :2]
    keep = torchvision.ops.nms(boxes, torch.rand(5000, device=dev), 0.5)
    print("torchvision NMS на GPU: OK,", keep.numel(), "рамок")
else:
    print("УВАГА: GPU не знайдено - навчання піде на CPU (повільно). Перевірте драйвер AMD Adrenalin.")
