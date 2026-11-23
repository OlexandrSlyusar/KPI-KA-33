# ЛР-4. Моделі з трендом, АРІКС, АР, АРКС та прогнози на їх основі (відтворення процедур EViews)
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ats import ls, lag, correlogram, describe, forecast_arma, forecast_lags, forecast_eval, eviews_text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
SERIES = [("RSAFSNA", "RSAFSNA.txt", 1992, 1, "обсяг роздрібної торгівлі та послуг харчування, млн дол. США"),
          ("CURRNS", "CURRNS.txt", 1947, 1, "кількість доларів США в обігу, млрд дол.")]
H = 8
LAGS = 24


def load(f):
    return np.loadtxt(os.path.join(DATA, f))


def labels(y0, m0, n):
    return ["%d:%02d" % (y0 + (m0 - 1 + i) // 12, (m0 - 1 + i) % 12 + 1) for i in range(n)]


def corr_dict(c):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}


def sig_pac(c, L):
    return [i + 1 for i in range(L) if abs(c["PAC"][i]) > c["band"]]


def sig_ac(c, L):
    return [i + 1 for i in range(L) if abs(c["AC"][i]) > c["band"]]


def summary(e, lab):
    d = {"name": e.name, "dep": e.dep, "names": e.names, "coef": e.coef.tolist(), "se": e.se.tolist(),
         "t": e.tstat.tolist(), "prob": e.prob.tolist(), "first": lab(e.first), "last": lab(e.last), "T": e.T,
         "k": e.k, "stats": {k: float(v) for k, v in e.st.items()}, "window": eviews_text(e, lab)}
    c = correlogram(e.e, LAGS)
    d.update(Q24=float(c["Q"][-1]), Q24p=float(c["Prob"][-1]))
    if hasattr(e, "intercept"):
        d.update(intercept=float(e.intercept), ar_lags=e.ar_lags, ma_lags=e.ma_lags,
                 ar_roots=np.abs(e.ar_roots).tolist(), ma_roots=np.abs(e.ma_roots).tolist())
    return d


def lag_num(name):
    return int(name[name.index("(") + 1:-1].replace("-", ""))


def prune(fit, arl, mal):
    """Послідовне вилучення незначущих (Prob > 0.05) складових, по одній - з найбільшим Prob."""
    steps = []
    while True:
        e = fit(arl, mal)
        cand = [(e.prob[i], e.names[i]) for i in range(len(e.names)) if e.names[i] != "C"]
        pmax, nmax = max(cand)
        steps.append({"ar": list(arl), "ma": list(mal), "aic": e["Akaike info criterion"],
                      "sc": e["Schwarz criterion"], "pmax": float(pmax), "worst": nmax})
        if pmax <= 0.05:
            return e, arl, mal, steps
        l = lag_num(nmax)
        if nmax.startswith("MA"):
            mal = [x for x in mal if x != l]
        else:
            arl = [x for x in arl if x != l]


def analyse(name, fn, y0, m0, desc):
    y = load(fn)
    n = len(y)
    NT = n - H
    lab_list = labels(y0, m0, n)
    lab = lambda i: lab_list[i - 1]
    ly = np.log(y)
    dly = ly - lag(ly, 1)
    k = np.arange(1, n + 1, dtype=float)
    F0, F1 = NT, n - 1
    act = ly[F0:F1 + 1]
    R = {"name": name, "desc": desc, "N": n, "NT": NT, "H": H, "first": lab_list[0], "last": lab_list[-1],
         "train": [lab_list[0], lab_list[NT - 1]], "test": [lab_list[F0], lab_list[F1]],
         "dstat": describe(y), "dstat_log": describe(ly), "labels_test": lab_list[F0:], "actual_log": act.tolist(),
         "actual": y[F0:].tolist()}

    # ---------- Тренди 1-3 порядку для logy ----------
    R["trends"], fc_static = [], {}
    for o in (1, 2, 3):
        regs = [("C", None)] + [("K" if j == 1 else "K^%d" % j, (k ** j)[:NT]) for j in range(1, o + 1)]
        e = ls("trend%d" % o, ly[:NT], regs, dep="LOGY")
        f = sum(e.beta[j] * k ** j for j in range(o + 1))
        d = summary(e, lab)
        d["order"] = o
        R["trends"].append(d)
        fc_static["trend%d" % o] = f[F0:F1 + 1]

    # ---------- Сезонність: АКФ відхилень від тренду ----------
    resid_tr = ly[:NT] - (R["trends"][1]["coef"][0] + R["trends"][1]["coef"][1] * k[:NT] + R["trends"][1]["coef"][2] * k[:NT] ** 2)
    R["corr_detr"] = corr_dict(correlogram(resid_tr, 36))

    # ---------- АРІКС: модель для перших різниць dlogy ----------
    cd = correlogram(dly[:NT], 36)
    R["corr_dlogy"] = corr_dict(cd)
    ar0 = sig_pac(cd, 12)
    fit_d = lambda a, m: ls("arima", dly[:NT], [("C", None)], dep="DLOGY", ar=a, ma=m)
    e_ar, arl, _, st1 = prune(fit_d, ar0, [])
    cr = correlogram(e_ar.resid[~np.isnan(e_ar.resid)], LAGS)
    ma0 = sig_ac(cr, 12)
    e_ari, arl, mal, st2 = prune(fit_d, arl, ma0)
    R["arima"] = {"ar0": ar0, "ma0": ma0, "steps_ar": st1, "steps_arma": st2, "corr_ar_res": corr_dict(cr),
                  "ar_only": summary(e_ar, lab), "model": summary(e_ari, lab),
                  "corr_res": corr_dict(correlogram(e_ari.e, LAGS))}
    X1 = np.ones((n, 1))
    ds = forecast_arma(e_ari, dly, X1, F0, F1, False)
    dd = forecast_arma(e_ari, dly, X1, F0, F1, True)
    fc_static["arima"] = ds[F0:F1 + 1] + ly[F0 - 1:F1]
    fc_dyn = {"arima": np.cumsum(dd[F0:F1 + 1]) + ly[F0 - 1]}
    R["arima"]["dlogy_static"], R["arima"]["dlogy_dynamic"] = ds[F0:F1 + 1].tolist(), dd[F0:F1 + 1].tolist()
    m = R["arima"]["model"]
    t0 = lab_list.index(m["first"])
    lvl = ly[t0:NT]
    R["arima"]["R2_logy"] = float(1 - m["stats"]["Sum squared resid"] / np.sum((lvl - lvl.mean()) ** 2))

    # ---------- АР-модель для logy (явні лаги) ----------
    cl = correlogram(ly[:NT], LAGS)
    R["corr_logy"] = corr_dict(cl)
    la0 = sig_pac(cl, 13)
    fit_ar = lambda a, mm: ls("ar", ly[:NT], [("C", None)] + [("LOGY(-%d)" % l, lag(ly, l)[:NT]) for l in a], dep="LOGY")
    e_lag, la, _, st3 = prune(fit_ar, la0, [])
    fc_static["ar"] = forecast_lags(e_lag, ly, F0, F1, False, la)[F0:F1 + 1]
    fc_dyn["ar"] = forecast_lags(e_lag, ly, F0, F1, True, la)[F0:F1 + 1]
    cr2 = correlogram(e_lag.e, LAGS)
    R["ar"] = {"lags0": la0, "lags": la, "steps": st3, "model": summary(e_lag, lab), "corr_res": corr_dict(cr2)}

    # ---------- АРКС-модель для logy ----------
    ma1 = sig_ac(cr2, 12)
    fit_arma = lambda a, mm: ls("arma", ly[:NT], [("C", None)], dep="LOGY", ar=a, ma=mm)
    e_arma, la2, ma2, st4 = prune(fit_arma, la, ma1)
    fc_static["arma"] = forecast_arma(e_arma, ly, X1, F0, F1, False)[F0:F1 + 1]
    fc_dyn["arma"] = forecast_arma(e_arma, ly, X1, F0, F1, True)[F0:F1 + 1]
    R["arma"] = {"ma0": ma1, "steps": st4, "model": summary(e_arma, lab), "corr_res": corr_dict(correlogram(e_arma.e, LAGS))}

    # ---------- Оцінка прогнозів ----------
    R["static"] = {kk: {"fc": v.tolist(), "eval": forecast_eval(act, v), "fc_y": np.exp(v).tolist(),
                        "eval_y": forecast_eval(y[F0:], np.exp(v))} for kk, v in fc_static.items()}
    R["dynamic"] = {kk: {"fc": v.tolist(), "eval": forecast_eval(act, v), "fc_y": np.exp(v).tolist(),
                         "eval_y": forecast_eval(y[F0:], np.exp(v))} for kk, v in fc_dyn.items()}
    figs(name, y, ly, lab_list, NT, R)
    return R


# ---------- Рисунки для протоколу ----------
def figs(name, y, ly, labs, NT, R):
    os.makedirs(OUT, exist_ok=True)
    t = np.arange(len(y))
    step = max(1, len(y) // 8)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for ax, v, ttl in ((axs[0], y, name), (axs[1], ly, "log(%s)" % name)):
        ax.plot(t, v, "k-", lw=0.8)
        ax.axvline(NT - 0.5, color="0.5", ls="--", lw=0.8)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks(t[::step]); ax.set_xticklabels([labs[i][:4] for i in t[::step]], fontsize=7, rotation=45)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "series_%s.png" % name), dpi=200); plt.close(fig)
    styles = {"trend1": ("0.55", ":"), "trend2": ("0.55", "--"), "trend3": ("0.55", "-."), "arima": ("k", "--"),
              "ar": ("0.3", ":"), "arma": ("0.3", "-.")}
    names = {"trend1": "тренд 1-го порядку", "trend2": "тренд 2-го порядку", "trend3": "тренд 3-го порядку",
             "arima": "АРІКС", "ar": "АР", "arma": "АРКС"}
    for kind in ("static", "dynamic"):
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        x = np.arange(R["H"])
        ax.plot(x, R["actual_log"], "k-o", lw=1.6, ms=3, label="реальне значення logy")
        for kk, v in R[kind].items():
            c, ls_ = styles[kk]
            ax.plot(x, v["fc"], color=c, ls=ls_, marker=".", lw=1.1, label=names[kk])
        ax.set_xticks(x); ax.set_xticklabels(R["labels_test"], fontsize=7)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fc_%s_%s.png" % (kind, name)), dpi=200); plt.close(fig)


def run_all():
    os.makedirs(OUT, exist_ok=True)
    res = {"series": [analyse(*s) for s in SERIES]}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    return res


if __name__ == "__main__":
    run_all()
    print("результати збережено: %s" % os.path.join(OUT, "results.json"))
