import torch
import torch.nn.functional as F
from torchvision import transforms as T, models
import torch.nn as nn
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Налаштування
try:
    import torch_directml
    DEVICE = torch_directml.device() if torch_directml.is_available() else torch.device("cpu")
except ImportError:
    DEVICE = torch.device("cpu")

OUT_DIR = Path("outputs")
MY_IMAGES_DIR = Path("my_images")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_SIZE = 224

eval_tf = T.Compose([T.Resize(IMG_SIZE * 8 // 7), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])

def build_model(arch, num_classes):
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def make_grid(n, cols, cell_w, cell_h):
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_w, rows * cell_h), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat

# 1. Завантаження моделі
print("Завантаження навченої моделі...")
ckpt = torch.load(OUT_DIR / "best_model.pt", map_location=DEVICE)
best_model = build_model(ckpt["arch"], len(ckpt["classes"]))
best_model.load_state_dict(ckpt["state_dict"])
best_model.to(DEVICE).eval()

@torch.no_grad()
def predict_top_k(model, path, k=5):
    with Image.open(path) as img:
        x = eval_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs, idx = F.softmax(model(x), dim=1)[0].topk(k)
    return [(ckpt["classes"][i], p) for p, i in zip(probs.tolist(), idx.tolist())]

# 2. Обробка картинок
own_files = sorted(p for p in MY_IMAGES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:10]

if not own_files:
    print(f"Помилка: Папка {MY_IMAGES_DIR.absolute()} порожня!")
else:
    fig, axes = make_grid(len(own_files), 5, 4, 4.5)
    for ax, path in zip(axes, own_files):
        top = predict_top_k(best_model, path)
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{top[0][0]} ({top[0][1]:.2f})", fontsize=10)
        ax.axis("off")
        print(f"{path.name} -> {top[0][0]} ({top[0][1]:.2f})")
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "own_images.png", dpi=150)
    print("\nГотово! Картинку збережено за адресою: outputs/own_images.png")
    plt.show()