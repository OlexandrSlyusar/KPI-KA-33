#1. Середовище та залежності
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import platform
import random
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from importlib import metadata

REQUIRED = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "tqdm": "tqdm",
            "transformers": "transformers", "datasets": "datasets", "rouge_score": "rouge-score",
            "sentencepiece": "sentencepiece", "safetensors": "safetensors"}


def ensure_packages():
    if importlib.util.find_spec("torch") is None:
        sys.exit("PyTorch не знайдено: оберіть у Visual Studio середовище omsm_lab2 venv (PyTorch з ROCm).")
    missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    pins = []
    for pkg in ("torch", "torchvision", "numpy"):
        try:
            pins.append(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            pass
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(pins))
    print(f"Встановлення пакетів: {' '.join(missing)} ...", flush=True)
    if subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing, "-c", f.name]).returncode:
        sys.exit("Не вдалося встановити залежності - див. повідомлення pip вище.")


ensure_packages()

import matplotlib

matplotlib.use("Agg")
import datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from huggingface_hub.utils import logging as hub_logging
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")
hub_logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
datasets.logging.set_verbosity_error()
datasets.disable_progress_bars()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


#2. Конфігурація експерименту
@dataclass(frozen=True)
class Config:
    dataset: str = "abisee/cnn_dailymail"
    version: str = "3.0.0"
    train_n: int = 20000
    val_n: int = 1000
    val_gen_n: int = 300
    test_n: int = 2000
    bonus_n: int = 500
    max_source: int = 512
    max_target: int = 128
    epochs: int = 3
    batch: int = 16
    eval_batch: int = 32
    lrs: tuple = (("bart", 5e-5), ("t5", 3e-4))
    same_lr: float = 5e-5
    weight_decay: float = 0.01
    warmup: float = 0.05
    clip: float = 1.0
    seed: int = 0
    beams: int = 4
    min_new: int = 56
    max_new: int = 142
    length_penalty: float = 2.0
    no_repeat: int = 3
    boot: int = 1000
    lat_gpu: int = 10
    lat_cpu: int = 3
    n_human: int = 5
    log_every: int = 25
    ckpt_minutes: float = 5.0
    out: str = "outputs"


SMOKE = dict(train_n=48, val_n=16, val_gen_n=8, test_n=24, bonus_n=8, max_source=128, max_target=48, epochs=2,
             batch=8, eval_batch=8, min_new=8, max_new=32, boot=100, lat_gpu=2, lat_cpu=1, n_human=2, log_every=2,
             ckpt_minutes=0.25, out="outputs_smoke")
MODELS = {"bart": ("facebook/bart-base", "BART-base", ""),
          "t5": ("google-t5/t5-small", "T5-small", "summarize: ")}
HF_NAMES = {"bart": "facebook/bart-base", "t5": "t5-small"}
COLORS = {"bart": "tab:blue", "t5": "tab:orange", "lead3": "tab:gray", "t5_lr": "tab:red"}
METRICS = ("rouge1", "rouge2", "rougeL", "rougeLsum")
MNAMES = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rougeLsum": "ROUGE-Lsum"}
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16 if DEV.type == "cuda" and torch.cuda.is_bf16_supported() else None
C = Config()
OUT = BONUS = WEIGHTS = ROOT / C.out


def configure(cfg):
    global C, OUT, BONUS, WEIGHTS
    C, OUT = cfg, ROOT / cfg.out
    BONUS, WEIGHTS = OUT / "bonus", OUT / "weights"
    for p in (OUT, BONUS, WEIGHTS):
        p.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def autocast(device=DEV):
    return torch.autocast("cuda", dtype=AMP) if AMP and device.type == "cuda" else contextlib.nullcontext()


def free():
    gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()


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


def fmt(v, digits=2):
    if isinstance(v, (float, np.floating)):
        return "-" if not np.isfinite(v) else f"{v:.{digits}f}"
    return str(v)


def table(df, title=None, digits=2):
    rows = [[str(c) for c in df.columns]] + [[fmt(v, digits) for v in r] for r in df.itertuples(index=False)]
    wid = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    num = [all(re.fullmatch(r"[-+0-9.e±]+", r[i]) for r in rows[1:]) for i in range(len(wid))]
    line = lambda a, b, c: "  " + a + b.join("─" * (w + 2) for w in wid) + c
    if title:
        print(paint(f"\n  {title}", "1"))
    print(line("┌", "┬", "┐"))
    for k, r in enumerate(rows):
        print("  │" + "│".join(f" {v.rjust(w) if k and n else v.ljust(w)} " for v, w, n in zip(r, wid, num)) + "│")
        if k == 0:
            print(line("├", "┼", "┤"))
    print(line("└", "┴", "┘"), flush=True)


def bar(it, desc, total=None):
    return tqdm(it, desc=f"  {desc}", total=total, leave=False, dynamic_ncols=True,
                bar_format="{desc} {percentage:3.0f}%|{bar:18}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")


class StopRequested(Exception):
    pass


STOP_KEYS = ("\x1b", "q", "Q", "й", "Й")


def stop_requested():
    flag = OUT / "STOP"
    if flag.exists():
        flag.unlink(missing_ok=True)
        return True
    if os.name == "nt" and sys.stdin.isatty():
        import msvcrt
        hit = False
        while msvcrt.kbhit():
            hit |= msvcrt.getwch() in STOP_KEYS
        return hit
    return False


#4. Дані CNN/DailyMail
WORD = re.compile(r"\w+")
SENT = re.compile(r"(?<=[.!?])\s+")


def words(text):
    return WORD.findall(text.lower())


def load_data():
    ds = load_dataset(C.dataset, C.version)
    full = {k: len(v) for k, v in ds.items()}

    def pick(split, n):
        return ds[split].shuffle(seed=C.seed).select(range(min(n, len(ds[split])))).flatten_indices()

    sets = dict(train=pick("train", C.train_n), val=pick("validation", C.val_n), test=pick("test", C.test_n))
    d = {k: {c: list(v[c]) for c in ("id", "article", "highlights")} for k, v in sets.items()}
    d.update(ds=sets, full=full)
    (OUT / "split.json").write_text(json.dumps({k: d[k]["id"] for k in sets}), encoding="utf-8")
    info(f"{C.dataset} {C.version}: повний датасет train {full['train']} / validation {full['validation']} / "
         f"test {full['test']}")
    n = {k: len(d[k]["id"]) for k in sets}
    info(f"підвибірки (seed = {C.seed}): train {n['train']} | validation {n['val']} | test {n['test']}")
    return d


def data_stats(d, toks):
    rows = []
    for split in ("train", "val", "test"):
        art = [len(words(a)) for a in d[split]["article"]]
        hl = [len(words(h)) for h in d[split]["highlights"]]
        row = {"Вибірка": split, "Статей": len(art), "Слів у статті (медіана)": float(np.median(art)),
               "Слів у summary": float(np.mean(hl)),
               "Речень у summary": float(np.mean([h.count("\n") + 1 for h in d[split]["highlights"]]))}
        for key, tok in toks.items():
            lens = np.array([len(x) for x in tok([MODELS[key][2] + a for a in d[split]["article"]])["input_ids"]])
            row[f"> {C.max_source} ток. {MODELS[key][1]}, %"] = float((lens > C.max_source).mean() * 100)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "data_stats.csv", index=False, encoding="utf-8-sig")
    table(df, "Статистика підвибірок", 1)
    return df


def encode(tok, prefix, dset):
    def fn(b):
        enc = tok([prefix + a for a in b["article"]], max_length=C.max_source, truncation=True)
        enc["labels"] = tok(text_target=b["highlights"], max_length=C.max_target, truncation=True)["input_ids"]
        return enc

    return dset.map(fn, batched=True, remove_columns=dset.column_names)


#5. Моделі
def load_model(key, path=None, device=DEV):
    src = str(path) if path else MODELS[key][0]
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSeq2SeqLM.from_pretrained(src).to(device)
    return tok, model


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def weights_mb(path):
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 2 ** 20


def gen_args(**over):
    args = dict(num_beams=C.beams, min_new_tokens=C.min_new, max_new_tokens=C.max_new,
                length_penalty=C.length_penalty, no_repeat_ngram_size=C.no_repeat, early_stopping=True)
    args.update(over)
    return args


GREEDY = dict(num_beams=1, length_penalty=1.0, early_stopping=False)


@torch.no_grad()
def summarize(model, tok, prefix, texts, gen=None, desc=None, batch=None, device=DEV):
    model.eval()
    batch, gen = batch or C.eval_batch, gen or gen_args()
    lens = [len(x) for x in tok([prefix + t for t in texts], max_length=C.max_source, truncation=True)["input_ids"]]
    order = np.argsort(-np.array(lens), kind="stable")
    out = [""] * len(texts)
    steps = range(0, len(texts), batch)
    sync()
    t0 = time.perf_counter()
    for i in (bar(steps, desc, len(steps)) if desc else steps):
        if stop_requested():
            raise StopRequested("генерацію перервано")
        idx = order[i:i + batch]
        enc = tok([prefix + texts[j] for j in idx], max_length=C.max_source, truncation=True, padding=True,
                  return_tensors="pt").to(device)
        with autocast(device):
            ids = model.generate(**enc, **gen)
        for j, s in zip(idx, tok.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)):
            out[j] = " ".join(s.split())
    sync()
    return out, time.perf_counter() - t0


#6. Метрики ROUGE та статистики тексту
SCORER = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=True)


def as_lines(text):
    return "\n".join(s.strip() for s in SENT.split(text.strip()) if s.strip())


def rouge_frame(preds, refs):
    rows = []
    for p, r in zip(preds, refs):
        s = SCORER.score(r, as_lines(p))
        rows.append({m: s[m].fmeasure * 100 for m in METRICS})
    return pd.DataFrame(rows)


def text_stats(preds, articles):
    rows = []
    for p, a in zip(preds, articles):
        w, src = words(p), words(a)
        bi = list(zip(w, w[1:]))
        tri = list(zip(w, w[1:], w[2:]))
        src_bi = set(zip(src, src[1:]))
        rows.append(dict(pred_words=len(w), pred_sents=len(as_lines(p).splitlines()),
                         novel_bigrams=100 * sum(b not in src_bi for b in bi) / max(1, len(bi)),
                         repeated_trigrams=100 * (1 - len(set(tri)) / max(1, len(tri)))))
    return pd.DataFrame(rows)


def boot_ci(values, n=None, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n or C.boot, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_boot(a, b, n=None, seed=0):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m, lo, hi = boot_ci(diff, n, seed)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n or C.boot, len(diff)))].mean(1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return dict(diff=m, lo=lo, hi=hi, p=float(min(1.0, p)))


def score_preds(name, preds, d, extra=None):
    df = pd.DataFrame({"id": d["test"]["id"][:len(preds)], "article_words": [len(words(a)) for a in
                                                                            d["test"]["article"][:len(preds)]],
                       "reference": d["test"]["highlights"][:len(preds)], "prediction": preds})
    df = pd.concat([df, rouge_frame(preds, df["reference"]), text_stats(preds, d["test"]["article"][:len(preds)])],
                   axis=1)
    for k, v in (extra or {}).items():
        df[k] = v
    return df


def lead3(article):
    return " ".join(SENT.split(" ".join(article.split()))[:3])


#7. Навчання
@torch.no_grad()
def val_loss(model, loader):
    model.eval()
    tot = cnt = 0.0
    for b in loader:
        if stop_requested():
            raise StopRequested("валідацію перервано")
        b = {k: v.to(DEV) for k, v in b.items()}
        with autocast():
            loss = model(**b).loss
        n = (b["labels"] != -100).sum().item()
        tot, cnt = tot + loss.item() * n, cnt + n
    return tot / cnt


def train_info(h, lr, train_time=float("nan")):
    b = int(h["val_loss"].idxmin())
    return dict(best_epoch=int(h["epoch"][b]), val_loss=float(h["val_loss"][b]),
                epoch_time=float(h["epoch_time"].median()), samples_per_s=float(h["samples_per_s"].median()),
                train_time=train_time, lr=lr)


def signature(lr):
    return dict(lr=lr, epochs=C.epochs, batch=C.batch, train_n=C.train_n, val_n=C.val_n, max_source=C.max_source,
                max_target=C.max_target, seed=C.seed, warmup=C.warmup, weight_decay=C.weight_decay)


def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_model(key, d, lr=None, tag=None):
    tag, lr = tag or key, lr or dict(C.lrs)[key]
    hf, name, prefix = MODELS[key]
    hist_path, ckpt = OUT / f"history_{tag}.csv", WEIGHTS / f"{tag}_resume.pt"
    if hist_path.exists() and not ckpt.exists() and (WEIGHTS / tag / "config.json").exists():
        h = pd.read_csv(hist_path)
        if len(h) == C.epochs:
            done(f"{name}: навчання завершено раніше ({C.epochs} епох) - завантажую збережені ваги")
            tok, model = load_model(key, WEIGHTS / tag)
            return tok, model, train_info(h, lr)
    set_seed(C.seed)
    tok, model = load_model(key)
    tr, va = encode(tok, prefix, d["ds"]["train"]), encode(tok, prefix, d["ds"]["val"])
    summarize(model, tok, prefix, d["val"]["article"][:2], dict(GREEDY, max_new_tokens=4, min_new_tokens=1))
    coll = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
    gen = torch.Generator().manual_seed(C.seed)
    loader = DataLoader(tr, C.batch, shuffle=True, collate_fn=coll, generator=gen)
    vloader = DataLoader(va, C.eval_batch, collate_fn=coll)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": C.weight_decay}, {"params": rest, "weight_decay": 0.0}],
                            lr=lr)
    total = len(loader) * C.epochs
    sched = get_linear_schedule_with_warmup(opt, int(C.warmup * total), total)
    gv = gen_args(**GREEDY, min_new_tokens=0)
    refs = d["val"]["highlights"][:C.val_gen_n]
    hist, steps, best, best_ep, step, start, part = [], [], math.inf, 0, 0, 1, None
    info(f"{name} ({hf}): {n_params(model) / 1e6:.1f} M параметрів · lr {lr:g} · {len(loader)} кроків на епоху · "
         f"{'bf16 AMP' if AMP else 'fp32'}")
    info(paint("Esc або Q - зупинити зі збереженням прогресу; продовжити - режим 1", "33"))
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        if st["sig"] == signature(lr):
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            gen.set_state(st["gen"])
            torch.set_rng_state(st["rng"])
            if DEV.type == "cuda" and st.get("cuda_rng"):
                torch.cuda.set_rng_state_all(st["cuda_rng"])
            hist, steps, best, best_ep, step = st["hist"], st["steps"], st["best"], st["best_ep"], st["step"]
            start, part = st["epoch"] + 1, st.get("part")
            where = f"епохи {start}/{C.epochs}" + (f", крок {part['batch']}/{len(loader)}" if part else "")
            done(f"{name}: продовжую перерваний запуск з {where}")
        del st

    def checkpoint(epoch, part_state=None):
        atomic_save(dict(sig=signature(lr), epoch=epoch, part=part_state, model=model.state_dict(),
                         opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if DEV.type == "cuda"
                         else None, hist=hist, steps=steps, best=best, best_ep=best_ep, step=step), ckpt)

    t_all = time.perf_counter()
    for ep in range(start, C.epochs + 1):
        if part:
            gen.set_state(part["ep_gen"])
            skip, losses, window, elapsed = part["batch"], part["losses"], part["window"], part["elapsed"]
        else:
            skip, losses, window, elapsed = 0, [], [], 0.0
        ep_gen, part = gen.get_state(), None
        model.train()
        sync()
        t0, last_ck = time.perf_counter(), time.perf_counter()
        pb = bar(loader, f"{name} · епоха {ep}/{C.epochs}")
        for i, b in enumerate(pb):
            if i < skip:
                t0 = last_ck = time.perf_counter()
                continue
            b = {k: v.to(DEV, non_blocking=True) for k, v in b.items()}
            with autocast():
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            lv = loss.item()
            if not math.isfinite(lv):
                raise FloatingPointError(f"{name}: loss = {lv} на кроці {step + 1}")
            losses.append(lv)
            window.append(lv)
            step += 1
            if step % C.log_every == 0:
                steps.append(dict(step=step, epoch=step / len(loader), loss=float(np.mean(window)),
                                  lr=sched.get_last_lr()[0]))
                pb.set_postfix_str(f"loss {np.mean(window):.3f}")
                window = []
            if stop_requested():
                sync()
                pb.close()
                checkpoint(ep - 1, dict(batch=i + 1, ep_gen=ep_gen, losses=losses, window=window,
                                        elapsed=elapsed + time.perf_counter() - t0))
                raise StopRequested(f"{name}: епоха {ep}, крок {i + 1}/{len(loader)}")
            if time.perf_counter() - last_ck > C.ckpt_minutes * 60 and i + 1 < len(loader):
                sync()
                checkpoint(ep - 1, dict(batch=i + 1, ep_gen=ep_gen, losses=losses, window=window,
                                        elapsed=elapsed + time.perf_counter() - t0))
                last_ck = time.perf_counter()
        pb.close()
        sync()
        ep_time = elapsed + time.perf_counter() - t0
        try:
            vl = val_loss(model, vloader)
            preds, _ = summarize(model, tok, prefix, d["val"]["article"][:C.val_gen_n], gv)
        except StopRequested:
            checkpoint(ep - 1, dict(batch=len(loader), ep_gen=ep_gen, losses=losses, window=window, elapsed=ep_time))
            raise StopRequested(f"{name}: валідація після епохи {ep}")
        vr = rouge_frame(preds, refs).mean()
        hist.append(dict(epoch=ep, train_loss=float(np.mean(losses)), val_loss=vl,
                         **{f"val_{m}": float(vr[m]) for m in METRICS}, lr=sched.get_last_lr()[0],
                         epoch_time=ep_time, samples_per_s=len(tr) / ep_time))
        mark = ""
        if vl < best:
            best, best_ep, mark = vl, ep, paint(" ★", "33")
            model.save_pretrained(WEIGHTS / tag)
            tok.save_pretrained(WEIGHTS / tag)
        pd.DataFrame(hist).to_csv(hist_path, index=False)
        pd.DataFrame(steps).to_csv(OUT / f"steps_{tag}.csv", index=False)
        if ep < C.epochs:
            checkpoint(ep)
        done(f"{name} · епоха {ep}: train loss {np.mean(losses):.3f} · val loss {vl:.3f} · val ROUGE-1/2/L "
             f"(greedy, {len(refs)}) {vr['rouge1']:.1f}/{vr['rouge2']:.1f}/{vr['rougeL']:.1f} · "
             f"{ep_time:.0f} с ({len(tr) / ep_time:.1f} зразків/с){mark}")
    ckpt.unlink(missing_ok=True)
    del opt, sched
    if best_ep != C.epochs:
        del model
        free()
        tok, model = load_model(key, WEIGHTS / tag)
    return tok, model, train_info(pd.DataFrame(hist), lr, time.perf_counter() - t_all)


#8. Оцінювання
def evaluate(key, tok, model, d, tag=None):
    tag = tag or key
    name, prefix = MODELS[key][1], MODELS[key][2]
    preds, sec = summarize(model, tok, prefix, d["test"]["article"], desc=f"{name} · генерація test")
    df = score_preds(tag, preds, d)
    df.to_csv(OUT / f"test_predictions_{tag}.csv", index=False, encoding="utf-8-sig")
    return df, sec


@torch.no_grad()
def latency(model, tok, prefix, texts, n, device):
    times, toks = [], []
    enc_all = [tok(prefix + t, max_length=C.max_source, truncation=True, return_tensors="pt") for t in texts[:n + 1]]
    for k, enc in enumerate(enc_all):
        if stop_requested():
            raise StopRequested("вимір latency перервано")
        enc = enc.to(device)
        sync()
        t0 = time.perf_counter()
        with autocast(device):
            ids = model.generate(**enc, **gen_args())
        sync()
        if k:
            times.append(time.perf_counter() - t0)
            toks.append(ids.shape[1])
    return float(np.median(times)), float(np.mean(toks))


def summary_row(key, df, info_, model, gen_sec, lat, tag=None):
    tag = tag or key
    row = {"Модель": HF_NAMES[key], "Завдання": "Сумаризація CNN/DM", "Параметри (М)": n_params(model) / 1e6,
           "Макс. довжина входу": C.max_source, "lr": info_["lr"], "Найкраща епоха": info_["best_epoch"],
           "Час епохи (с)": info_["epoch_time"], "Навчання (зразків/с)": info_["samples_per_s"],
           "Val loss": info_["val_loss"]}
    for m in METRICS:
        mean, lo, hi = boot_ci(df[m])
        row[MNAMES[m]] = mean
        row[f"{MNAMES[m]} ±"] = (hi - lo) / 2
    row.update({"Довжина summary (слів)": df["pred_words"].mean(), "Нові біграми (%)": df["novel_bigrams"].mean(),
                "Повтори триграм (%)": df["repeated_trigrams"].mean(), "Розмір ваг (МБ)": weights_mb(WEIGHTS / tag),
                "Генерація test (статей/с)": len(df) / gen_sec, "Latency GPU (с)": lat[0], "Latency CPU (с)": lat[1]})
    return row


#9. Візуалізації
def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training(tags):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for tag, label, color in tags:
        st, h = pd.read_csv(OUT / f"steps_{tag}.csv"), pd.read_csv(OUT / f"history_{tag}.csv")
        ax[0].plot(st["epoch"], st["loss"], color=color, alpha=0.8, lw=1, label=f"{label}: train")
        ax[0].plot(h["epoch"], h["val_loss"], "o--", color=color, lw=2, label=f"{label}: validation")
        ax[1].plot(h["epoch"], h["val_rouge1"], "o-", color=color, label=f"{label}: ROUGE-1")
        ax[1].plot(h["epoch"], h["val_rougeL"], "s--", color=color, label=f"{label}: ROUGE-L")
        ax[2].plot(st["epoch"], st["lr"], color=color, label=label)
    ax[0].set(title="Cross-entropy loss", xlabel="Епоха", ylabel="loss")
    ax[1].set(title=f"ROUGE на validation (greedy, {C.val_gen_n} статей)", xlabel="Епоха", ylabel="F1, %")
    ax[2].set(title="Learning rate (warmup + linear decay)", xlabel="Епоха", ylabel="lr")
    ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    save(fig, OUT / "training_curves.png")


def plot_rouge(rows):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(rows)
    for k, r in enumerate(rows):
        x = np.arange(len(METRICS)) + (k - (len(rows) - 1) / 2) * w
        ax.bar(x, [r[MNAMES[m]] for m in METRICS], w, yerr=[r.get(f"{MNAMES[m]} ±", 0) for m in METRICS],
               capsize=3, color=r["color"], label=r["Модель"])
        for xi, m in zip(x, METRICS):
            ax.text(xi, r[MNAMES[m]] + r.get(f"{MNAMES[m]} ±", 0) + 0.5, f"{r[MNAMES[m]]:.1f}", ha="center",
                    fontsize=8)
    ax.set_xticks(range(len(METRICS)), [MNAMES[m] for m in METRICS])
    ax.set(ylabel="F1, %", title=f"ROUGE на test ({C.test_n} статей), 95 % bootstrap CI")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, OUT / "rouge_comparison.png")


def plot_lengths(dfs, test):
    ref = [len(words(h)) for h in test["highlights"]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    bins = np.arange(0, 121, 5)
    ax[0].hist(ref, bins, alpha=0.45, color="tab:green", label=f"Еталон (середнє {np.mean(ref):.0f})")
    for key, df in dfs.items():
        ax[0].hist(df["pred_words"], bins, histtype="step", lw=2, color=COLORS[key],
                   label=f"{HF_NAMES[key]} (середнє {df['pred_words'].mean():.0f})")
    ax[0].set(title="Довжина summary", xlabel="Слів", ylabel="Статей")
    ax[0].legend()
    a, b = dfs["bart"]["rougeL"], dfs["t5"]["rougeL"]
    ax[1].scatter(b, a, s=8, alpha=0.4)
    lim = max(a.max(), b.max()) + 2
    ax[1].plot([0, lim], [0, lim], "k--", lw=1)
    ax[1].set(title=f"ROUGE-L по статтях (BART краще - {np.mean(a > b) * 100:.0f} %)", xlabel="t5-small",
              ylabel="facebook/bart-base", xlim=(0, lim), ylim=(0, lim))
    for x in ax:
        x.grid(alpha=0.3)
    save(fig, OUT / "length_and_agreement.png")


def plot_speed(cmp):
    cols = [("Параметри (М)", "Параметри, М"), ("Розмір ваг (МБ)", "Розмір ваг, МБ"), ("Час епохи (с)", "Час епохи, с"),
            ("Latency GPU (с)", "Latency GPU, с/статтю"), ("Latency CPU (с)", "Latency CPU, с/статтю")]
    fig, ax = plt.subplots(1, len(cols), figsize=(16, 3.6))
    for a, (c, t) in zip(ax, cols):
        v = cmp[c].to_numpy()
        a.bar(cmp["Модель"], v, color=[COLORS["bart"], COLORS["t5"]])
        for i, x in enumerate(v):
            a.text(i, x, f"{x:.1f}" if x >= 10 else f"{x:.2f}", ha="center", va="bottom", fontsize=9)
        a.set_title(t, fontsize=10)
        a.tick_params(axis="x", labelsize=8)
        a.grid(axis="y", alpha=0.3)
    save(fig, OUT / "speed_size.png")


#10. Основний експеримент: facebook/bart-base vs t5-small
def human_eval(d, dfs):
    idx = sorted(np.random.default_rng(1).choice(len(dfs["bart"]), C.n_human, replace=False).tolist())
    items, md = [], [f"# Оцінка «людським оком»: {C.n_human} статей test", ""]
    for k, i in enumerate(idx, 1):
        art = " ".join(d["test"]["article"][i].split())
        it = dict(n=k, id=d["test"]["id"][i], article=art, reference=d["test"]["highlights"][i].replace(" .\n", ". "),
                  **{key: dfs[key]["prediction"][i] for key in dfs},
                  **{f"rougeL_{key}": float(dfs[key]["rougeL"][i]) for key in dfs})
        items.append(it)
        md += [f"## Стаття {k} · id {it['id']} · {len(words(art))} слів", "", f"**Стаття:** {art}", "",
               f"**Еталон:** {it['reference']}", ""]
        md += [f"**{HF_NAMES[key]}** (ROUGE-L {it[f'rougeL_{key}']:.1f}): {it[key]}" + "\n" for key in dfs]
    (OUT / "human_eval.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "human_eval.md").write_text("\n".join(md), encoding="utf-8")
    done(f"{C.n_human} статей для оцінки людським оком: {(OUT / 'human_eval.md').relative_to(ROOT)}")


def run_main(d):
    stage("Навчання основних моделей: facebook/bart-base vs t5-small")
    toks = {k: AutoTokenizer.from_pretrained(MODELS[k][0]) for k in MODELS}
    data_stats(d, toks)
    rows, dfs, meta = [], {}, []
    for key in MODELS:
        saved = OUT / f"summary_{key}.json"
        if saved.exists() and (OUT / f"test_predictions_{key}.csv").exists():
            st = json.loads(saved.read_text(encoding="utf-8"))
            rows.append(st["row"])
            meta.append(st["meta"])
            dfs[key] = load_preds(key)
            done(f"{HF_NAMES[key]}: навчання й оцінювання завершено раніше - беру збережені результати")
            continue
        tok, model, inf = train_model(key, d)
        df, sec = evaluate(key, tok, model, d)
        lat_gpu = latency(model, tok, MODELS[key][2], d["test"]["article"], C.lat_gpu, DEV)[0] if DEV.type == "cuda" \
            else float("nan")
        del model
        free()
        cpu_tok, cpu_model = load_model(key, WEIGHTS / key, torch.device("cpu"))
        lat_cpu = latency(cpu_model, cpu_tok, MODELS[key][2], d["test"]["article"], C.lat_cpu, torch.device("cpu"))[0]
        rows.append(summary_row(key, df, inf, cpu_model, sec, (lat_gpu, lat_cpu)))
        del cpu_model
        free()
        dfs[key] = df
        meta.append(dict(key=key, model=HF_NAMES[key], hf=MODELS[key][0], path=f"weights/{key}", prefix=MODELS[key][2],
                         **inf))
        saved.write_text(json.dumps(dict(row=rows[-1], meta=meta[-1]), ensure_ascii=False, default=float),
                         encoding="utf-8")
        r = rows[-1]
        done(f"{HF_NAMES[key]}: ROUGE-1/2/L/Lsum {r['ROUGE-1']:.2f} / {r['ROUGE-2']:.2f} / {r['ROUGE-L']:.2f} / "
             f"{r['ROUGE-Lsum']:.2f} · {r['Довжина summary (слів)']:.0f} слів · {r['Генерація test (статей/с)']:.1f} "
             f"статей/с · ваги {r['Розмір ваг (МБ)']:.0f} МБ")
    l3 = [lead3(a) for a in d["test"]["article"]]
    dl = score_preds("lead3", l3, d)
    dl.to_csv(OUT / "test_predictions_lead3.csv", index=False, encoding="utf-8-sig")
    lead = {"Модель": "Lead-3 (перші 3 речення)", **{MNAMES[m]: boot_ci(dl[m])[0] for m in METRICS},
            **{f"{MNAMES[m]} ±": (boot_ci(dl[m])[2] - boot_ci(dl[m])[1]) / 2 for m in METRICS},
            "Довжина summary (слів)": dl["pred_words"].mean(), "Нові біграми (%)": dl["novel_bigrams"].mean(),
            "Повтори триграм (%)": dl["repeated_trigrams"].mean()}
    cmp = pd.DataFrame(rows)
    cmp.to_csv(OUT / "comparison_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([*rows, lead]).to_csv(OUT / "rouge_with_baseline.csv", index=False, encoding="utf-8-sig")
    sig = {MNAMES[m]: paired_boot(dfs["bart"][m], dfs["t5"][m]) for m in METRICS}
    (OUT / "significance.json").write_text(json.dumps(sig, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "models.json").write_text(json.dumps(dict(models=meta, generation=gen_args(), max_source=C.max_source),
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    table(cmp[["Модель", "Параметри (М)", "Макс. довжина входу", "Час епохи (с)", "ROUGE-1", "ROUGE-2", "ROUGE-L",
               "ROUGE-Lsum", "Розмір ваг (МБ)", "Latency GPU (с)", "Latency CPU (с)"]],
          f"Порівняння моделей (test, {len(d['test']['id'])} статей; latency - batch 1, beam {C.beams})")
    table(pd.DataFrame([{"Модель": r["Модель"], **{MNAMES[m]: r[MNAMES[m]] for m in METRICS},
                         "Слів": r["Довжина summary (слів)"], "Нові біграми, %": r["Нові біграми (%)"]}
                        for r in [*rows, lead]]), "ROUGE з базовою лінією Lead-3")
    for m, s in sig.items():
        info(f"{m}: BART − T5 = {s['diff']:+.2f} (95 % CI {s['lo']:+.2f}…{s['hi']:+.2f}, p = {s['p']:.3f})")
    plot_training([(k, HF_NAMES[k], COLORS[k]) for k in MODELS])
    plot_rouge([dict(r, color=COLORS[k]) for r, k in zip(rows, MODELS)] + [dict(lead, color=COLORS["lead3"])])
    plot_lengths(dfs, d["test"])
    plot_speed(cmp)
    human_eval(d, dfs)
    done("рисунки та таблиці збережено")
    return dfs


#11. Бонус: zero-shot, декодування, однаковий lr, довжина статей
def load_preds(tag):
    return pd.read_csv(OUT / f"test_predictions_{tag}.csv", keep_default_na=False)


def bonus_zero_shot(d, dfs):
    if (BONUS / "zero_shot.csv").exists():
        return table(pd.read_csv(BONUS / "zero_shot.csv").drop(columns="Приклад"), "Zero-shot vs fine-tuning (раніше)")
    rows, n = [], min(C.bonus_n, len(d["test"]["id"]))
    sub = {c: v[:n] for c, v in d["test"].items()}
    for key in MODELS:
        tok, model = load_model(key)
        preds, sec = summarize(model, tok, MODELS[key][2], sub["article"], desc=f"zero-shot {MODELS[key][1]}")
        z = pd.concat([rouge_frame(preds, sub["highlights"]), text_stats(preds, sub["article"])], axis=1)
        for kind, df in (("без fine-tuning (zero-shot)", z), ("після fine-tuning", dfs[key].iloc[:n])):
            rows.append({"Модель": HF_NAMES[key], "Варіант": kind, **{MNAMES[m]: df[m].mean() for m in METRICS},
                         "Слів": df["pred_words"].mean(), "Нові біграми, %": df["novel_bigrams"].mean()})
        rows[-2]["Приклад"] = preds[0][:300]
        del model
        free()
    out = pd.DataFrame(rows)
    out.to_csv(BONUS / "zero_shot.csv", index=False, encoding="utf-8-sig")
    table(out.drop(columns="Приклад"), f"Zero-shot vs fine-tuning (test, {n} статей)")


DECODING = (("greedy", dict(GREEDY, min_new_tokens=0, no_repeat_ngram_size=0)),
            ("beam 4", dict(length_penalty=1.0, min_new_tokens=0, no_repeat_ngram_size=0)),
            ("beam 4 + no-repeat 3", dict(length_penalty=1.0, min_new_tokens=0)),
            ("beam 4 + no-repeat 3 + lp 2 + min 56 (основний)", {}))


def bonus_decoding(d):
    if (BONUS / "decoding.csv").exists():
        return table(pd.read_csv(BONUS / "decoding.csv"), "Стратегії декодування (раніше)")
    rows, n = [], min(C.bonus_n, len(d["test"]["id"]))
    sub = {c: v[:n] for c, v in d["test"].items()}
    for key in MODELS:
        tok, model = load_model(key, WEIGHTS / key)
        for label, over in DECODING:
            preds, sec = summarize(model, tok, MODELS[key][2], sub["article"], gen_args(**over),
                                   desc=f"{MODELS[key][1]} · {label}")
            df = pd.concat([rouge_frame(preds, sub["highlights"]), text_stats(preds, sub["article"])], axis=1)
            rows.append({"Модель": HF_NAMES[key], "Декодування": label, **{MNAMES[m]: df[m].mean() for m in METRICS},
                         "Слів": df["pred_words"].mean(), "Повтори триграм, %": df["repeated_trigrams"].mean(),
                         "Нові біграми, %": df["novel_bigrams"].mean(), "мс/статтю": 1000 * sec / n})
        del model
        free()
    out = pd.DataFrame(rows)
    out.to_csv(BONUS / "decoding.csv", index=False, encoding="utf-8-sig")
    table(out, f"Стратегії декодування (test, {n} статей)")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for key in MODELS:
        s = out[out["Модель"] == HF_NAMES[key]]
        ax[0].plot(range(len(s)), s["ROUGE-L"], "o-", color=COLORS[key], label=HF_NAMES[key])
        ax[1].plot(range(len(s)), s["Повтори триграм, %"], "o-", color=COLORS[key], label=HF_NAMES[key])
    for a, t in zip(ax, ("ROUGE-L, %", "Повтори триграм, %")):
        a.set_xticks(range(len(DECODING)), ["greedy", "beam 4", "+ no-repeat 3", "+ lp 2, min 56"])
        a.set(title=t)
        a.grid(alpha=0.3)
        a.legend()
    save(fig, BONUS / "decoding.png")


def bonus_same_lr(d, dfs):
    tag = "t5_lr5e-5"
    if (OUT / f"test_predictions_{tag}.csv").exists() and not (WEIGHTS / f"{tag}_resume.pt").exists():
        df = load_preds(tag)
    else:
        tok, model, _ = train_model("t5", d, lr=C.same_lr, tag=tag)
        df, _ = evaluate("t5", tok, model, d, tag=tag)
        del model
        free()
    runs = (("bart", dict(C.lrs)["bart"], dfs["bart"], "bart", "facebook/bart-base, lr 5e-5", COLORS["bart"]),
            ("t5", dict(C.lrs)["t5"], dfs["t5"], "t5", "t5-small, lr 3e-4", COLORS["t5"]),
            ("t5", C.same_lr, df, tag, "t5-small, lr 5e-5", COLORS["t5_lr"]))
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for key, lr, frame, h, label, color in runs:
        hist = pd.read_csv(OUT / f"history_{h}.csv")
        rows.append({"Модель": HF_NAMES[key], "lr": f"{lr:g}", **{MNAMES[m]: frame[m].mean() for m in METRICS},
                     "Val loss": hist["val_loss"].min(), "Слів": frame["pred_words"].mean()})
        ax[0].plot(hist["epoch"], hist["val_loss"], "o-", color=color, label=label)
        ax[1].plot(hist["epoch"], hist["val_rougeL"], "o-", color=color, label=label)
    out = pd.DataFrame(rows)
    out.to_csv(BONUS / "same_lr.csv", index=False, encoding="utf-8-sig")
    table(out, f"Однаковий learning rate (t5-small з lr = {C.same_lr:g}, як у BART)", 3)
    ax[0].set(title="Validation loss", xlabel="Епоха")
    ax[1].set(title="Validation ROUGE-L (greedy)", xlabel="Епоха")
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    save(fig, BONUS / "same_lr.png")


def bonus_length(d, dfs):
    lead = load_preds("lead3")
    frames = {**dfs, "lead3": lead}
    q = pd.qcut(dfs["bart"]["article_words"], 4, duplicates="drop")
    nq = len(q.cat.categories)
    rows = []
    for k, interval in enumerate(q.cat.categories):
        m = (q == interval).to_numpy()
        rows.append({"Квартиль довжини статті": f"Q{k + 1}: {int(interval.left) + 1}-{int(interval.right)} слів",
                     "Статей": int(m.sum()),
                     **{f"ROUGE-L {HF_NAMES.get(t, 'Lead-3')}": f["rougeL"].to_numpy()[m].mean()
                        for t, f in frames.items()}})
    toks = {k: AutoTokenizer.from_pretrained(MODELS[k][0]) for k in MODELS}
    lens = np.array([len(x) for x in toks["bart"](d["test"]["article"])["input_ids"]])
    for trunc, m in (("≤ 512 токенів (без обрізання)", lens <= C.max_source), ("> 512 токенів (обрізано)",
                                                                                lens > C.max_source)):
        rows.append({"Квартиль довжини статті": trunc, "Статей": int(m.sum()),
                     **{f"ROUGE-L {HF_NAMES.get(t, 'Lead-3')}": f["rougeL"].to_numpy()[m].mean() if m.any()
                        else float("nan") for t, f in frames.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(BONUS / "rouge_by_length.csv", index=False, encoding="utf-8-sig")
    table(out, "ROUGE-L залежно від довжини статті (test)")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(nq)
    for k, (t, f) in enumerate(frames.items()):
        ax.bar(x + (k - 1) * 0.27, out[f"ROUGE-L {HF_NAMES.get(t, 'Lead-3')}"][:nq], 0.27,
               color=COLORS.get(t, "tab:gray"), label=HF_NAMES.get(t, "Lead-3"))
    ax.set_xticks(x, out["Квартиль довжини статті"][:nq], fontsize=8)
    ax.set(ylabel="ROUGE-L, %", title="ROUGE-L за квартилями довжини статті")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, BONUS / "rouge_by_length.png")


def run_bonus(d, dfs=None):
    stage("Бонус: zero-shot, стратегії декодування, однаковий lr, довжина статей")
    dfs = dfs or {k: load_preds(k) for k in MODELS}
    bonus_zero_shot(d, dfs)
    bonus_decoding(d)
    bonus_length(d, dfs)
    bonus_same_lr(d, dfs)
    done("бонусні експерименти завершено")


#12. Інференс на власних прикладах
OWN_TEXTS = [
    ("Лабораторна робота (EN)",
     "Students of the Institute for Applied System Analysis at Igor Sikorsky Kyiv Polytechnic Institute have "
     "compared two transformer models that write short summaries of news articles. The team fine-tuned BART and T5 "
     "on twenty thousand stories from the CNN and Daily Mail websites using a single graphics card at home. Each "
     "model read the first five hundred tokens of an article and learned to produce three or four sentences that "
     "describe the main events. The larger BART model needed more than twice as much time per epoch, but its "
     "summaries were closer to the reference highlights written by journalists. The smaller T5 model was faster "
     "and lighter, which makes it more convenient for laptops and phones. The students also checked the summaries "
     "by hand and noticed that both models mostly copy whole sentences from the beginning of the text. They plan "
     "to repeat the experiment with Ukrainian news next semester."),
    ("Погода / повінь (EN, вигадані назви)",
     "Heavy rain caused the river Silverbrook to burst its banks on Tuesday, flooding dozens of homes in the small "
     "town of Eastmere. Local officials said that more than 300 residents were moved to a school gymnasium and "
     "two hotels overnight. No one was seriously injured, but a 70-year-old man was treated for hypothermia after "
     "spending several hours on the roof of his car. The town mayor, Linda Hartley, said the water level had risen "
     "by almost two metres in less than six hours. Forecasters warned that another 40 millimetres of rain could fall "
     "on Thursday, and the regional council has asked people living near the river to prepare emergency bags. "
     "Volunteers have already filled more than 5,000 sandbags. Insurance companies estimate the damage at several "
     "million pounds."),
    ("Спорт (EN, вигадані клуби)",
     "Northbridge Rovers came from behind to beat Eastfield United 3-1 on Saturday and move to the top of the "
     "league table for the first time in eleven years. Eastfield took the lead after just six minutes when "
     "striker Tom Walsh headed in a corner. Northbridge equalised before half-time through a long-range shot by "
     "18-year-old midfielder Adam Kerr, who was making only his third senior appearance. Two goals in the last "
     "fifteen minutes from substitute Marco Silva sealed the victory and sent the home crowd of 24,000 into "
     "celebration. Northbridge manager Paul Grant praised his young players and said the team still had a lot of "
     "hard work ahead. Eastfield have now lost three matches in a row and their coach admitted that the defence "
     "had been too slow."),
    ("Технології (EN, вигадана компанія)",
     "A start-up called Voltera has opened a factory that recycles old electric car batteries, recovering lithium, "
     "nickel and cobalt that can be used again in new cells. The plant, built on the site of a former steelworks, "
     "will process up to 10,000 tonnes of batteries a year and employ about 250 people. Company director Sarah "
     "Mills said the process uses water-based chemistry instead of high-temperature furnaces, cutting energy use "
     "by almost half. Car makers face new rules that require a growing share of recycled metals in batteries sold "
     "from 2030. Environmental groups welcomed the project but said more collection points were needed, because "
     "many small batteries from phones and scooters still end up in household waste."),
    ("Українською (поза мовою навчання)",
     "У Києві студенти КПІ ім. Ігоря Сікорського представили систему, яка автоматично складає короткий зміст "
     "новинних статей. Розробники навчили дві нейронні мережі на десятках тисяч англомовних новин і порівняли їх "
     "за якістю та швидкістю. За словами авторів, більша модель пише точніші резюме, а менша працює майже вдвічі "
     "швидше. Наступним кроком команда планує адаптувати систему до українських новин."),
]


def own_texts():
    path = ROOT / "my_articles.txt"
    extra = []
    if path.exists():
        parts = [p.strip() for p in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")) if p.strip()]
        extra = [(f"my_articles.txt #{k}", p) for k, p in enumerate(parts, 1)]
    return OWN_TEXTS + extra


def run_inference():
    stage("Інференс збережених моделей на власних прикладах")
    texts = own_texts()
    rows, md = [], ["# Інференс на власних прикладах", ""]
    outs = {}
    for key in MODELS:
        path = WEIGHTS / key
        tok, model = load_model(key, path)
        outs[key], sec = summarize(model, tok, MODELS[key][2], [t for _, t in texts])
        done(f"{HF_NAMES[key]}: {len(texts)} текстів за {sec:.1f} с")
        del model
        free()
    for k, (title, text) in enumerate(texts):
        md += [f"## {k + 1}. {title} ({len(words(text))} слів)", "", text, ""]
        print(paint(f"\n  {k + 1}. {title}", "1"))
        for key in MODELS:
            s = outs[key][k]
            stat = text_stats([s], [text]).iloc[0]
            rows.append(dict(n=k + 1, title=title, model=HF_NAMES[key], text_words=len(words(text)), summary=s,
                             summary_words=int(stat["pred_words"]), novel_bigrams=float(stat["novel_bigrams"])))
            md += [f"**{HF_NAMES[key]}:** {s}", ""]
            print(f"  {paint(HF_NAMES[key].ljust(19), '36')} {s}")
    pd.DataFrame(rows).to_csv(OUT / "inference_own.csv", index=False, encoding="utf-8-sig")
    (OUT / "inference_own.md").write_text("\n".join(md), encoding="utf-8")
    done(f"{(OUT / 'inference_own.md').relative_to(ROOT)}, {(OUT / 'inference_own.csv').relative_to(ROOT)}")


#13. Меню, запуск і журнал
MODES = {"1": ("all", "Повний експеримент: BART і T5 → бонус → інференс", "~1,5 год"),
         "2": ("main", "Лише основні моделі (fine-tuning, ROUGE, швидкодія, 5 статей)", "~55 хв"),
         "3": ("bonus", "Лише бонус на збережених вагах (zero-shot, декодування, lr, довжина)", "~30 хв"),
         "4": ("infer", "Інференс збережених моделей на власних прикладах", "~1 хв"),
         "5": ("smoke", "Smoke-тест усього конвеєра на малих підвибірках", "~5 хв"),
         "6": ("fresh", "Почати повний експеримент заново (видалити попередні результати)", "~1,5 год")}


def device_line():
    if DEV.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        hip = getattr(torch.version, "hip", None)
        backend = f"ROCm/HIP {hip}" if hip else f"CUDA {torch.version.cuda}"
        return (f"GPU {p.name} ({p.total_memory / 2 ** 30:.0f} ГБ) · PyTorch {torch.__version__} · {backend} · "
                f"{'bf16 AMP' if AMP else 'fp32'}")
    return f"CPU ({platform.processor() or platform.machine()}) · PyTorch {torch.__version__} - GPU не знайдено"


def unfinished():
    p = OUT / "run_config.json"
    if not p.exists() or ((BONUS / "same_lr.csv").exists() and (OUT / "inference_own.csv").exists()):
        return False
    try:
        old = json.loads(p.read_text(encoding="utf-8"))["config"]
    except (ValueError, KeyError):
        return False
    new = json.loads(json.dumps(asdict(C)))
    return all(old.get(k) == v for k, v in new.items() if k not in ("ckpt_minutes", "log_every"))


def progress_text():
    parts = []
    for tag, label in (("bart", "BART"), ("t5", "T5"), ("t5_lr5e-5", "T5 lr 5e-5")):
        h = OUT / f"history_{tag}.csv"
        if (OUT / f"test_predictions_{tag}.csv").exists():
            parts.append(f"{label} ✓")
        elif h.exists():
            parts.append(f"{label} епоха {len(pd.read_csv(h))}/{C.epochs}")
    return ", ".join(parts) or "моделі ще не навчено"


def status_lines():
    hub = Path(os.environ["HF_HOME"]) / "hub"
    have = (hub / "datasets--abisee--cnn_dailymail").is_dir()
    lines = [paint("Дані: ", "1") + ("✓ " if have else "✗ буде завантажено (~840 МБ), ") + f"{C.dataset} {C.version}"]
    n = sum((WEIGHTS / k / "config.json").exists() for k in MODELS)
    lines.append(paint("Збережені моделі: ", "1") + f"{n} з {len(MODELS)}" +
                 (", бонус ✓" if (BONUS / "same_lr.csv").exists() else ""))
    if unfinished():
        lines.append(paint("Перерваний запуск: ", "1;33") + f"{progress_text()} - режим 1 продовжить з місця зупинки")
    if (OUT / "comparison_table.csv").exists():
        t = pd.read_csv(OUT / "comparison_table.csv")
        lines.append(paint("Останні результати: ", "1") + " · ".join(
            f"{r['Модель']} ROUGE-L {r['ROUGE-L']:.2f}" for _, r in t.iterrows()))
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
                print(f"\r{prompt} - автозапуск через {left:2d} с (будь-яка клавіша - вибрати вручну) ", end="",
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
    if mode == "fresh":
        if OUT.exists():
            shutil.rmtree(OUT)
        configure(C)
        mode = "all"
    need = {"bonus": OUT / "test_predictions_t5.csv", "infer": OUT / "models.json"}.get(mode)
    if need and not need.exists():
        return warn(f"Немає {need.relative_to(ROOT)} - спочатку виконайте режим 1 або 2")
    resume = mode == "all" and unfinished()
    if mode in ("all", "smoke") and OUT.exists() and not resume:
        shutil.rmtree(OUT)
        configure(C)
    log = Tee(OUT / "log.txt")
    sys.stdout = log
    t0 = time.time()
    try:
        print(f"\n==== {datetime.now():%Y-%m-%d %H:%M} · режим {mode} · {device_line()} ====")
        if resume:
            done(f"продовжую перерваний запуск: {progress_text()}")
        dfs = None
        if mode != "infer":
            stage("Дані")
            d = load_data()
            versions = {k: metadata.version(k) for k in ("torch", "transformers", "datasets", "tokenizers",
                                                          "rouge-score", "numpy", "pandas")}
            cfg_json = dict(config=asdict(C), mode=mode, device=device_line(), python=platform.python_version(),
                            versions=versions, models={k: MODELS[k][0] for k in MODELS},
                            split={**{k: len(d[k]["id"]) for k in ("train", "val", "test")}, "full": d["full"]},
                            generation=gen_args())
            (OUT / "run_config.json").write_text(json.dumps(cfg_json, ensure_ascii=False, indent=2), encoding="utf-8")
            if mode in ("all", "main", "smoke"):
                dfs = run_main(d)
            if mode in ("all", "bonus", "smoke"):
                run_bonus(d, dfs)
        if mode in ("all", "main", "infer", "smoke"):
            run_inference()
        stage(f"Готово за {(time.time() - t0) / 60:.1f} хв · результати: {OUT}")
    except StopRequested as e:
        warn(f"зупинено ({e}) - прогрес збережено, режим 1 продовжить з цього місця")
    except KeyboardInterrupt:
        warn(f"перервано Ctrl+C - режим 1 продовжить з останньої контрольної точки (не старшої за "
             f"{C.ckpt_minutes:g} хв)")
    except Exception:
        print(paint(traceback.format_exc(), "31"))
        warn("режим завершився з помилкою - повний текст у log.txt")
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser(description="ОМСМ, лабораторна робота 4: трансформери для сумаризації новин")
    parser.add_argument("mode", nargs="?", choices=[m for m, _, _ in MODES.values()])
    args = parser.parse_args()
    cfg = Config()
    if args.mode:
        return run(args.mode, cfg)
    first = True
    while True:
        configure(cfg)
        print()
        box([paint("ОМСМ · Лабораторна робота 4 · Transformers (NLP)", "1"),
             "Варіант 13 · CNN/DailyMail · абстрактивна сумаризація · facebook/bart-base vs t5-small",
             "КА-33 · Тодощенко Є., Сивенко К., Слюсар О.", "", device_line(), *status_lines(), "",
             paint("Під час роботи: Esc або Q - пауза зі збереженням; режим 1 - продовжити з місця зупинки", "2")])
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
