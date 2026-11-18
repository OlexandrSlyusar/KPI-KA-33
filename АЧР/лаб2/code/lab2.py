# ЛР-2. Аналіз часових рядів. Ковзні середні та ЧАКФ (реалізація на Python)
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf, acf

DATA = sys.argv[1] if len(sys.argv) > 1 else "Data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3})

# ---------- Частина 1: ряди v та y (бригада 1) ----------
A0, A1, A2, A3 = 0.0, 0.22, -0.18, 0.08
B1, B2, B3 = 0.5, 0.25, 0.25

def generate_vy(n=100, seed=2026):
    v = np.random.default_rng(seed).standard_normal(n)
    y = np.zeros(n)
    y[:3] = v[:3]
    for k in range(3, n):
        y[k] = (A0 + A1*y[k-1] + A2*y[k-2] + A3*y[k-3]
                + v[k] + B1*v[k-1] + B2*v[k-2] + B3*v[k-3])
    return v, y

# ---------- Частина 2: ковзні середні (формули 11.1, 11.2) ----------
def ema_weights(N):
    alpha = 2/(N + 1)
    return np.array([(1 - alpha)**(N - i + 1) for i in range(1, N + 1)])

def sma(x, N):
    out = np.full(len(x), np.nan)
    for k in range(N - 1, len(x)):
        out[k] = x[k - N + 1:k + 1].sum() / N
    return out

def ema(x, N):
    w = ema_weights(N)
    out = np.full(len(x), np.nan)
    for k in range(N - 1, len(x)):
        out[k] = (w * x[k - N + 1:k + 1]).sum() / w.sum()
    return out

# ---------- АКФ та ЧАКФ (формули 13.1, 13.2) ----------
def acf_r(x, L):
    m = x.mean()
    den = ((x - m)**2).sum()
    return np.array([((x[s:] - m)*(x[:-s] - m)).sum()/den
                     for s in range(1, L + 1)])

def pacf_prog(x, L):
    r = acf_r(x, L)
    phi = np.zeros((L + 1, L + 1))
    phi[1, 1] = r[0]
    for k in range(2, L + 1):
        num = r[k-1] - sum(phi[k-1, j]*r[k-j-1] for j in range(1, k))
        den = 1 - sum(phi[k-1, j]*r[j-1] for j in range(1, k))
        phi[k, k] = num/den
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k]*phi[k-1, k-j]
    return np.array([phi[k, k] for k in range(1, L + 1)])

def ljung_box(r, N, s):
    k = np.arange(1, s + 1)
    return N*(N + 2)*np.sum(r[:s]**2/(N - k))

# ---------- Графіки ----------
def plot_pair(x, ma, name, label, fname):
    t = np.arange(1, 51)
    plt.figure(figsize=(6.6, 3.3))
    plt.plot(t, x[:50], "k-o", ms=2, label=name)
    plt.plot(t, ma[:50], "k--", label=label)
    plt.xlim(1, 50); plt.xlabel("time"); plt.ylabel("value"); plt.legend()
    plt.savefig(os.path.join(OUT, fname), dpi=200, bbox_inches="tight")
    plt.close()

def plot_full(x, name, fname):
    t = np.arange(1, len(x) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True)
    for ax, N in zip(axs, (5, 10)):
        ax.plot(t, x, "k-", lw=1, label=name)
        ax.plot(t, sma(x, N), "k--", lw=1, label="_SMA (N=%d)" % N)
        ax.plot(t, ema(x, N), color="0.4", ls="-.", lw=1,
                label="EMA (N=%d)" % N)
        ax.set_ylabel("value"); ax.legend(fontsize=7)
    axs[1].set_xlabel("time"); axs[1].set_xlim(1, len(x))
    fig.savefig(os.path.join(OUT, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_correlogram(x, name, fname, L=12):
    N = len(x); lim = 2/np.sqrt(N)
    vals = (acf(x, nlags=L, fft=False)[1:], pacf_prog(x, L))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, v, ttl in zip(axs, vals, ("АКФ (AC)", "ЧАКФ (PAC)")):
        ax.bar(range(1, L + 1), v, width=0.6, fill=False, hatch="////",
               edgecolor="k")
        ax.axhline(lim, color="k", ls="--", lw=0.8)
        ax.axhline(-lim, color="k", ls="--", lw=0.8)
        ax.set_title(ttl, fontsize=9); ax.set_xticks(range(1, L + 1))
        ax.set_xlabel("лаг s")
    fig.savefig(os.path.join(OUT, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    # --- Частина 1 ---
    v, y = generate_vy()
    print("v: mean=%.4f std=%.4f;  y: mean=%.4f std=%.4f"
          % (v.mean(), v.std(ddof=1), y.mean(), y.std(ddof=1)))
    print("corr(v,y)=%.3f  max y=%.2f (k=%d)  max v=%.2f"
          % (np.corrcoef(v, y)[0, 1], y.max(), y.argmax() + 1, v.max()))
    t = np.arange(1, 101)
    plt.figure(figsize=(6.6, 3.3))
    plt.plot(t, y, "k-", lw=1.2, label="Y")
    plt.plot(t, v, "k--", lw=0.8, label="V")
    plt.xlim(1, 100); plt.xlabel("time"); plt.ylabel("value"); plt.legend()
    plt.savefig(os.path.join(OUT, "fig1_v_y.png"), dpi=200,
                bbox_inches="tight")
    plt.close()

    # --- Таблиця 4: ваги ---
    for N in (5, 10):
        w = ema_weights(N)
        print("N=%d, alpha=%.4f, w =" % (N, 2/(N + 1)), np.round(w, 6),
              " sum=%.6f" % w.sum())

    # --- Частина 2 ---
    for name, fn in (("rts1", "rts1.txt"), ("rts1996", "1996rts1.txt")):
        x = np.loadtxt(os.path.join(DATA, fn))
        print("\n%s: n=%d mean=%.2f std=%.2f min=%.2f max=%.2f"
              % (name, len(x), x.mean(), x.std(ddof=1), x.min(), x.max()))
        for N in (5, 10):
            plot_pair(x, sma(x, N), name, "_SMA", f"{name}_sma{N}.png")
            plot_pair(x, ema(x, N), name, "EMA", f"{name}_ema{N}.png")
            print("  N=%2d  mean|y-MA|: _SMA=%.3f  EMA=%.3f"
                  % (N, np.nanmean(abs(x - sma(x, N))),
                     np.nanmean(abs(x - ema(x, N)))))
        plot_full(x, name, f"{name}_full.png")
        plot_correlogram(x, name, f"{name}_correlogram.png")

        p1 = pacf_prog(x, 12)
        p2 = pacf(x, nlags=12, method="ldb")[1:]
        r1 = acf_r(x, 12)
        r2 = acf(x, nlags=12, fft=False)[1:]
        print("  max|pacf_prog - pacf| = %.1e;  max|r - acf| = %.1e"
              % (abs(p1 - p2).max(), abs(r1 - r2).max()))
        for s in range(12):
            print("  %2d  %10.6f  %7.3f" % (s + 1, p1[s], p2[s]))
        print("  Таблиця 7 (лаг, AC, Q):")
        for s in (1, 2, 3, 6, 12):
            print("  %2d  %.3f  %8.2f" % (s, r1[s-1],
                                          ljung_box(r1, len(x), s)))
