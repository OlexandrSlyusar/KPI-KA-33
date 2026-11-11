# ЛР-3. Оцінювання параметрів регресійних рівнянь (відтворення процедур EViews)
# Запуск: python lab3.py <папка ATS_lab_03_new/Data> [папка для результатів]
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ats import (ls, lag, movav, ema, ema_weights, ma_coefs, durbin_watson, describe,
                 correlogram, eviews_text)

DATA = sys.argv[1] if len(sys.argv) > 1 else "../data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../out"
os.makedirs(OUT, exist_ok=True)
SERIES = [("rts1", "rts1.txt"), ("rts1996", "1996rts1.txt")]
SECTORS = ["RTScr", "RTSeu", "RTSfn", "RTSin", "RTSmm", "RTSog", "RTStl"]
LAGS = 12


def load(f):
    for d in (DATA, os.path.dirname(os.path.abspath(DATA))):
        if os.path.exists(os.path.join(d, f)):
            return np.loadtxt(os.path.join(d, f))
    raise FileNotFoundError(f)


def order(c):
    sig = [i + 1 for i in range(len(c["PAC"])) if abs(c["PAC"][i]) > c["band"]]
    return (max(sig) if sig else 1), sig


def summary(e):
    d = {"name": e.name, "dep": e.dep, "names": e.names, "coef": e.coef.tolist(), "se": e.se.tolist(),
         "t": e.tstat.tolist(), "prob": e.prob.tolist(), "first": e.first, "last": e.last, "T": e.T,
         "k": e.k, "stats": {k: float(v) for k, v in e.st.items()}, "window": eviews_text(e),
         "resid": e.e.tolist()}
    c = correlogram(e.e, LAGS)
    d.update(Q12=float(c["Q"][-1]), Q12p=float(c["Prob"][-1]), rpac1=float(c["PAC"][0]))
    if hasattr(e, "intercept"):
        d.update(intercept=float(e.intercept), p=e.p, q=e.q,
                 ar_roots=np.abs(e.ar_roots).tolist(), ma_roots=np.abs(e.ma_roots).tolist())
    return d


def corr_dict(c):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}


def ylags(y, p):
    return [("Y(-%d)" % i, lag(y, i)) for i in range(1, p + 1)]


def mvlags(mv, q, nm="MV"):
    return [("%s(-%d)" % (nm, j), lag(mv, j)) for j in range(1, q + 1)]


# ---------- Частина 1: статистика Дарбіна-Уотсона ----------
def part1():
    e = load("example_for_DW.txt")
    return {"n": len(e), "num": float(np.sum(np.diff(e) ** 2)), "den": float(np.sum(e ** 2)),
            "dw": durbin_watson(e)}


# ---------- Частина 2: побудова адекватного рівняння АРКС ----------
def part2(name, y):
    R = {"name": name, "N": len(y), "desc": describe(y)}
    cy = correlogram(y, LAGS)
    p, sig = order(cy)
    R.update(corr_y=corr_dict(cy), p=p, p_sig=sig)
    ar = ls("eq_ar", y, [("C", None)] + ylags(y, p))
    cr = correlogram(ar.resid, LAGS)
    q, qsig = order(cr)
    R.update(ar=summary(ar), corr_res=corr_dict(cr), q=q, q_sig=qsig)

    seq, prev = [], None
    for j in range(1, q + 1):
        st = [] if prev is None else [np.concatenate([prev.coef, [0.0]])]
        prev = ls("eq_arma_%d_%d" % (p, j), y, [("C", None)], ar=p, ma=j, starts=st)
        seq.append(summary(prev))
    R["ev_seq"] = seq
    S1 = {"ev": seq[-1]}
    mas = {"sma5": lambda x: movav(x, 5), "sma10": lambda x: movav(x, 10),
           "ema5": lambda x: ema(x, 5), "ema10": lambda x: ema(x, 10)}
    for key, f in mas.items():
        mv = f(ar.resid)
        e = ls("eq_r_" + key, y - mv, [("C", None)] + ylags(y, p) + mvlags(mv, q), dep="Y1")
        e.R2y = 1 - e["Sum squared resid"] / np.sum((y[e.first - 1:e.last] - y[e.first - 1:e.last].mean()) ** 2)
        S1[key] = summary(e)
        S1[key]["R2y"] = float(e.R2y)
    R["s1"] = S1

    S2 = {}
    for key, f in mas.items():
        mv = f(y)
        cm = correlogram(mv, LAGS)
        qm, qms = order(cm)
        b = ma_coefs(qm)
        y1 = y - mv - sum(b[j - 1] * lag(mv, j) for j in range(1, qm + 1))
        e1 = ls("eq_y1_" + key, y1, [("C", None)] + ylags(y, p), dep="Y1")
        e2 = ls("eq_y2_" + key, y - mv, [("C", None)] + ylags(y, p) + mvlags(mv, qm), dep="Y1")
        d = {"corr": corr_dict(cm), "q": qm, "q_sig": qms, "b": b.tolist(), "a1": summary(e1), "a2": summary(e2)}
        for tag, e in (("a1", e1), ("a2", e2)):
            yy = y[e.first - 1:e.last]
            d[tag]["R2y"] = float(1 - e["Sum squared resid"] / np.sum((yy - yy.mean()) ** 2))
        S2[key] = d
    R["s2"] = S2
    R["ar"]["R2y"] = R["ar"]["stats"]["R-squared"]
    for d in seq:
        d["R2y"] = d["stats"]["R-squared"]
    return R


# ---------- Частина 3: множинна регресія rts1 на галузеві індекси ----------
def part3():
    D = {n: load(n + ".txt") for n in ["rts1"] + SECTORS}
    names = ["rts1"] + SECTORS
    M = np.corrcoef(np.vstack([D[n] for n in names]))
    y = D["rts1"]
    incl = [n for n in SECTORS if abs(M[0, names.index(n)]) >= 0.5]
    models, regs = [], [("C", None)] + [(n.upper(), D[n]) for n in incl]
    while True:
        e = ls("eq_mr%d" % (len(models) + 1), y, regs, dep="RTS1")
        models.append(summary(e))
        i = int(np.argmax(e.prob))
        if e.prob[i] <= 0.05:
            break
        regs = [r for r in regs if r[0] != e.names[i]]
    cr = correlogram(ls("x", y, [("C", None)] + [(n.upper(), D[n]) for n in incl]).resid, LAGS)
    return {"names": names, "corr": M.tolist(), "included": incl, "models": models, "corr_res": corr_dict(cr)}


# ---------- Частина 4: ЧКФ та модель ARMAX ----------
def pccf(y, x, K=LAGS):
    out = []
    for k in range(0, K + 1):
        xk = lag(x, k)
        ctrl = [lag(x, j) for j in range(0, k)]
        ok = ~np.isnan(xk) & ~np.isnan(y)
        for c in ctrl:
            ok &= ~np.isnan(c)
        Z = np.column_stack([np.ones(ok.sum())] + [c[ok] for c in ctrl])
        ry = y[ok] - Z @ np.linalg.lstsq(Z, y[ok], rcond=None)[0]
        rx = xk[ok] - Z @ np.linalg.lstsq(Z, xk[ok], rcond=None)[0]
        out.append((float(np.sum(ry * rx) / np.sqrt(np.sum(ry ** 2) * np.sum(rx ** 2))), int(ok.sum())))
    return out


def part4(p):
    D = {n: load(n + ".txt") for n in ["rts1"] + SECTORS}
    y = D["rts1"]
    P = {n: pccf(y, D[n]) for n in SECTORS}
    band = 2 / np.sqrt(min(v[-1][1] for v in P.values()))
    lag1 = [n for n in SECTORS if abs(P[n][1][0]) > band]
    base = [("C", None)] + [(n.upper(), D[n]) for n in SECTORS] + \
           [("RTS1(-%d)" % i, lag(y, i)) for i in range(1, p + 1)]
    m1 = ls("eq_armax1", y, base + [(n.upper() + "(-1)", lag(D[n], 1)) for n in lag1], dep="RTS1")
    regs = base + [(n.upper() + "(-1)", lag(D[n], 1)) for n in SECTORS]
    steps = []
    while True:
        e = ls("eq_armax2", y, regs, dep="RTS1")
        i = int(np.argmax(e.prob))
        steps.append({"k": e.k, "drop": e.names[i], "p": float(e.prob[i]),
                      "aic": e["Akaike info criterion"], "sc": e["Schwarz criterion"], "dw": e["Durbin-Watson stat"]})
        if len(steps) == 1:
            m2 = e
        if e.prob[i] <= 0.05:
            break
        regs = [r for r in regs if r[0] != e.names[i]]
    m3 = e
    m3.name = "eq_armax3"
    m4 = ls("eq_armax4", y, [("C", None)] + [(n.upper(), D[n]) for n in SECTORS], dep="RTS1", ar=1)
    c3 = correlogram(m3.resid, LAGS)
    c1 = correlogram(m1.resid, LAGS)
    return {"pccf": P, "band": float(band), "lag1": lag1, "m1": summary(m1), "m2": summary(m2),
            "m3": summary(m3), "m4": summary(m4), "steps": steps, "corr_m3": corr_dict(c3), "corr_m1": corr_dict(c1)}


# ---------- Рисунки для протоколу ----------
def hist_fig(name, y, fname):
    d = describe(y)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [2.1, 1]})
    a.hist(y, bins=int(np.ceil(np.sqrt(len(y)))) + 4, color="0.55", edgecolor="k", lw=0.5)
    a.set_xlabel("значення"); a.set_ylabel("частота"); a.grid(alpha=0.3)
    txt = "Series: %s\nSample 1 %d\nObservations %d\n\n" % (name.upper(), len(y), len(y))
    for k in ["Mean", "Median", "Maximum", "Minimum", "Std. Dev.", "Skewness", "Kurtosis"]:
        txt += "%-10s %10.4f\n" % (k, d[k])
    txt += "\n%-10s %10.4f\n%-10s %10.6f" % ("Jarque-Bera", d["Jarque-Bera"], "Probability", d["Probability"])
    b.axis("off")
    b.text(0, 1, txt, va="top", family="monospace", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close(fig)


def fit_fig(name, y, eq, fname, label):
    t = np.arange(1, len(y) + 1)
    fig, a = plt.subplots(figsize=(6.6, 2.8))
    a.plot(t, y, "k-", lw=1, label=name)
    f = np.full(len(y), np.nan)
    f[eq["first"] - 1:eq["last"]] = y[eq["first"] - 1:eq["last"]] - np.array(eq["resid"])
    a.plot(t, f, color="0.5", ls="--", lw=1, label=label)
    a.set_xlim(1, len(y)); a.set_xlabel("k"); a.set_ylabel("значення"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=200); plt.close(fig)


if __name__ == "__main__":
    res = {"part1": part1(), "series": [], "w5": ema_weights(5).tolist(), "w10": ema_weights(10).tolist()}
    print("DW(example_for_DW) = %.5f" % res["part1"]["dw"])
    for name, fn in SERIES:
        y = load(fn)
        R = part2(name, y)
        res["series"].append(R)
        hist_fig(name, y, "hist_%s.png" % name)
        print("\n%s: p=%d (%s), q=%d (%s)" % (name, R["p"], R["p_sig"], R["q"], R["q_sig"]))
        for k, v in R["s2"].items():
            print("  КС по y %-5s q=%d b=%s" % (k, v["q"], np.round(v["b"], 4)))
    res["part3"] = part3()
    res["part4"] = part4(res["series"][0]["p"])
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print("\nрезультати збережено: %s" % os.path.join(OUT, "results.json"))
