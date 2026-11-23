# Перевірка відтворення EViews на прикладі методички ЛР-4 (агрегат M1 США, US_M1.txt)
import os, sys
import numpy as np
from ats import ls, lag, forecast_arma, forecast_lags, forecast_eval

D = sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
y = np.loadtxt(os.path.join(D, "US_M1.txt"))
k = np.arange(1, len(y) + 1, dtype=float)
NT, H = 124, 4
F0, F1 = NT, NT + H - 1
logy = np.log(y)
dlogy = logy - lag(logy, 1)
tr = lambda v: v[:NT]
rows = []


def check(label, got, ref, tol):
    rows.append((label, ref, got, abs(got - ref) <= tol))


# ---------- Тренди ----------
t2 = ls("trend2", tr(y), [("C", None), ("K", tr(k)), ("K*K", tr(k * k))])
check("trend2 C(3)", t2.coef[2], 59536587, 1)
check("trend2 R2", t2["R-squared"], 0.990498, 5e-7)
check("trend2 DW", t2["Durbin-Watson stat"], 0.385007, 5e-7)
f2 = t2.beta[0] + t2.beta[1] * k + t2.beta[2] * k * k
ev = forecast_eval(y[F0:F1 + 1], f2[F0:F1 + 1])
for key, ref, tol in [("MAPE", 3.437945, 5e-6), ("Theil", 0.018128, 5e-7), ("Bias", 0.568661, 5e-7),
                      ("Variance", 0.296051, 5e-7), ("Covariance", 0.135288, 5e-7)]:
    check("trend2 forecast " + key, ev[key], ref, tol)

t1 = ls("trend1", tr(logy), [("C", None), ("K", tr(k))])
check("trend1 C(1)", t1.coef[0], 25.47495, 5e-6)
check("trend1 C(2)", t1.coef[1], 0.015726, 5e-7)
check("trend1 SSR", t1["Sum squared resid"], 0.606017, 5e-7)
check("trend1 DW", t1["Durbin-Watson stat"], 0.172794, 5e-7)
f1 = t1.beta[0] + t1.beta[1] * k
for i, ref in enumerate([27.4406, 27.4563, 27.4721, 27.4878]):
    check("trend1 forecast %d" % (i + 1), f1[F0 + i], ref, 1e-4)

# ---------- АР(4) та АРІКС(4,1,4) для dlogy ----------
a4 = ls("ar4", tr(dlogy), [("C", None)], dep="DLOGY", ar=[1, 3, 4])
for nm, i, ref in [("C", 0, 0.015171), ("AR(1)", 1, -0.071629), ("AR(3)", 2, -0.126503), ("AR(4)", 3, 0.778280)]:
    check("ar4 " + nm, a4.coef[i], ref, 2e-6)
check("ar4 SSR", a4["Sum squared resid"], 0.020537, 5e-7)
check("ar4 DW", a4["Durbin-Watson stat"], 1.850656, 5e-6)

am = ls("ar4ma4", tr(dlogy), [("C", None)], dep="DLOGY", ar=[1, 3, 4], ma=[4])
for nm, i, ref in [("C", 0, 0.022841), ("AR(1)", 1, -0.015120), ("AR(3)", 2, -0.007358), ("AR(4)", 3, 0.981706),
                   ("MA(4)", 4, -0.938908)]:
    check("ar4ma4 " + nm, am.coef[i], ref, 2e-4)
check("ar4ma4 SSR", am["Sum squared resid"], 0.015413, 5e-6)
check("ar4ma4 R2", am["R-squared"], 0.848861, 5e-5)
X1 = np.ones((len(y), 1))
ds = forecast_arma(am, dlogy, X1, F0, F1, False)
dd = forecast_arma(am, dlogy, X1, F0, F1, True)
for i, ref in enumerate([-0.019953, 0.034743, 0.009746, 0.050085]):
    check("АРІКС static dlogy_f %d" % (i + 1), ds[F0 + i], ref, 3e-5)
for i, ref in enumerate([-0.02033, 0.034753, 0.009635, 0.049368]):
    check("АРІКС dynamic dlogy_f %d" % (i + 1), dd[F0 + i], ref, 3e-5)
ls_static = ds[F0:F1 + 1] + logy[F0 - 1:F1]
ld = np.cumsum(dd[F0:F1 + 1]) + logy[F0 - 1]
for i, ref in enumerate([27.45294, 27.49257, 27.48656, 27.52375]):
    check("АРІКС static logy_f %d" % (i + 1), ls_static[i], ref, 3e-5)
for i, ref in enumerate([27.45255612, 27.48730916, 27.49694384, 27.54631223]):
    check("АРІКС dynamic logy_f %d" % (i + 1), ld[i], ref, 3e-5)

# ---------- АР(1) та АРКС(1,4) для logy ----------
a1 = ls("ar1", tr(logy), [("C", None), ("LOGY(-1)", tr(lag(logy, 1)))], dep="LOGY")
check("ar1 C(1) (у тексті методички пропущено знак)", a1.coef[0], -0.0838, 5e-5)
check("ar1 C(2)", a1.coef[1], 1.0037, 5e-5)
s1 = forecast_lags(a1, logy, F0, F1, False, [1])
d1 = forecast_lags(a1, logy, F0, F1, True, [1])
for i, ref in enumerate([27.4914, 27.4763, 27.4953, 27.4922]):
    check("АР(1) static %d" % (i + 1), s1[F0 + i], ref, 1e-4)
for i, ref in enumerate([27.4914, 27.51, 27.5288, 27.5475]):
    check("АР(1) dynamic %d" % (i + 1), d1[F0 + i], ref, 1e-4)
check("trend2 forecast RMSE (EViews)", ev["RMSE"], 3.20e10, 5e8)
check("trend2 forecast MAE (EViews)", ev["MAE"], 2.95e10, 5e8)
mrs = lambda f: forecast_eval(logy[F0:F1 + 1], f)["RMSE"] / np.sqrt(H)
check("тренд static sqrt(sum e2)/N", mrs(f1[F0:F1 + 1]), 0.0173, 6e-5)
check("АРІКС static sqrt(sum e2)/N", mrs(ls_static), 0.0087, 1e-4)
check("АР(1) static sqrt(sum e2)/N", mrs(s1[F0:F1 + 1]), 0.0179, 6e-5)
check("АРІКС dynamic sqrt(sum e2)/N", mrs(ld), 0.00666, 1e-5)
check("АР(1) dynamic sqrt(sum e2)/N", mrs(d1[F0:F1 + 1]), 0.01819, 6e-6)

if __name__ == "__main__":
    bad = 0
    for lab_, ref, got, ok in rows:
        bad += not ok
        print("%-30s методичка %16.6f   отримано %16.6f   %s" % (lab_, ref, got, "OK" if ok else "!!"))
    print("\nперевірок: %d, розбіжностей: %d" % (len(rows), bad))
    if len(sys.argv) > 2:
        import json
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump([{"label": l, "ref": float(r), "got": float(g), "ok": bool(o)} for l, r, g, o in rows], f,
                      ensure_ascii=False, indent=1)
    am2 = ls("ar1ma4", tr(logy), [("C", None)], dep="LOGY", ar=[1], ma=[1, 3, 4])
    print("\nar1ma4:", np.round(am2.coef, 4), "a0 =", round(am2.intercept, 4), "SSR", am2["Sum squared resid"])
    print("static ", np.round(forecast_arma(am2, logy, X1, F0, F1, False)[F0:], 4), "(методичка 27.462 27.4875 27.4907 27.5066)")
    print("dynamic", np.round(forecast_arma(am2, logy, X1, F0, F1, True)[F0:], 4), "(методичка 27.462 27.4907 27.502 27.5295)")
