# Перевірка відтворення EViews на тестових даних методички ЛР-3
import os, sys
import numpy as np
from ats import ls, lag, movav, correlogram, describe, durbin_watson, eviews_text

D = sys.argv[1] if len(sys.argv) > 1 else "../data"
# D - папка ATS_lab_03_new з файлами example_*.txt
rd = lambda f: np.loadtxt(os.path.join(D, f))
rows = []


def check(label, got, ref, tol):
    ok = abs(got - ref) <= tol
    rows.append((label, ref, got, ok))


# ---------- Частина 1: DW ----------
check("DW example_for_DW", durbin_watson(rd("example_for_DW.txt")), 1.22881, 5e-6)

# ---------- Описова статистика (рис. 2) ----------
y = rd("example_data.txt")
d = describe(y)
for k, ref in [("Mean", -0.023067), ("Median", -0.018040), ("Std. Dev.", 0.381094),
               ("Skewness", 0.019862), ("Kurtosis", 2.388550), ("Jarque-Bera", 1.564370),
               ("Probability", 0.457406)]:
    check("desc " + k, d[k], ref, 1e-6)

# ---------- Корелограма y (рис. А.3) ----------
c = correlogram(y, 13)
for i, (ac, pac, q) in enumerate([(0.466, 0.466, 22.326), (-0.161, -0.482, 25.015), (-0.322, 0.023, 35.893)]):
    check("AC(%d)" % (i + 1), c["AC"][i], ac, 5e-4)
    check("PAC(%d)" % (i + 1), c["PAC"][i], pac, 5e-4)
    check("Q(%d)" % (i + 1), c["Q"][i], q, 5e-3)

# ---------- АР(1), АР(2) (рис. 9, А.6) ----------
e1 = ls("AR1", y, [("C", None), ("Y(-1)", lag(y, 1))])
for k, ref in [("R-squared", 0.216793), ("Sum squared resid", 11.25689), ("Akaike info criterion", 0.704142),
               ("Durbin-Watson stat", 1.549877), ("Log likelihood", -32.85503)]:
    check("AR1 " + k, e1[k], ref, 5e-6)
check("AR1 c(2)", e1.coef[1], 0.465547, 5e-7)
e2 = ls("AR2", y, [("C", None), ("Y(-1)", lag(y, 1)), ("Y(-2)", lag(y, 2))])
check("AR2 c", e2.coef[0], -0.021125, 5e-7)
check("AR2 y(-1)", e2.coef[1], 0.690188, 5e-7)
check("AR2 y(-2)", e2.coef[2], -0.483355, 5e-7)
check("AR2 DW", e2["Durbin-Watson stat"], 1.974741, 5e-6)

# ---------- АРКС(2,1), власне КС по залишках АР(2) (рис. А.8) ----------
mv = movav(e2.resid, 5)
y1 = y - mv
e3 = ls("ARMA_2_1", y1, [("C", None), ("Y(-1)", lag(y, 1)), ("Y(-2)", lag(y, 2)), ("MV(-1)", lag(mv, 1))], dep="Y1")
check("ARMA_2_1 nobs", e3.T, 93, 0)
check("ARMA_2_1 MV(-1)", e3.coef[3], -0.708681, 5e-7)
check("ARMA_2_1 R2", e3["R-squared"], 0.548451, 5e-6)
check("ARMA_2_1 DW", e3["Durbin-Watson stat"], 2.057499, 5e-6)

# ---------- АРКС(2,1) засобами EViews: ls y c ar(1) ar(2) ma(1) (рис. А.9) ----------
e4 = ls("ARMA_2_1_B", y, [("C", None)], ar=2, ma=1)
for nm, i, ref in [("C", 0, -0.026556), ("AR(1)", 1, 0.664291), ("AR(2)", 2, -0.471170), ("MA(1)", 3, 0.033675)]:
    check("ARMA_2_1_B " + nm, e4.coef[i], ref, 1e-4)
check("ARMA_2_1_B SE AR(1)", e4.se[1], 0.190244, 1e-3)
check("ARMA_2_1_B SSR", e4["Sum squared resid"], 8.613713, 1e-6)
check("ARMA_2_1_B DW", e4["Durbin-Watson stat"], 1.992145, 1e-4)
check("ARMA_2_1_B backcast", e4.backcast, 2, 0)

# ---------- КС по y: корелограма mvy (рис. А.10) і ls y1 ar(1) ar(2) mvy(-1) (рис. А.11) ----------
mvy = movav(y, 5)
cm = correlogram(mvy, 13)
check("MVY n", cm["n"], 96, 0)
check("MVY PAC(2)", cm["PAC"][1], -0.709, 5e-4)
check("MVY PAC(6)", cm["PAC"][5], 0.312, 5e-4)
y1 = y - mvy
e5 = ls("ARMAY_2_1", y1, [("MVY(-1)", lag(mvy, 1))], dep="Y1", ar=2)
for nm, i, ref in [("MVY(-1)", 0, -0.782354), ("AR(1)", 1, 0.624671), ("AR(2)", 2, -0.546118)]:
    check("ARMAY_2_1 " + nm, e5.coef[i], ref, 2e-5)
check("ARMAY_2_1 SSR", e5["Sum squared resid"], 5.464503, 2e-5)

# ---------- Множинна регресія (табл. Б.3) ----------
Y, X1, X2 = rd("example_y.txt"), rd("example_x1.txt"), rd("example_x2.txt")
q1 = ls("EQ1", Y, [("C", None), ("X1", X1), ("X2", X2)])
q2 = ls("EQ2", np.log(Y), [("C", None), ("LOG(X1)", np.log(X1)), ("LOG(X2)", np.log(X2))])
q3 = ls("EQ3", Y, [("C", None), ("X1", X1), ("LOG(X2)", np.log(X2))])
q4 = ls("EQ4", Y, [("C", None), ("Y(-1)", lag(Y, 1)), ("X1", X1), ("LOG(X2)", np.log(X2))])
for q, r2, ssr, aic, dw in [(q1, 0.997091, 34073631, 17.42310, 0.922350), (q2, 0.988716, 0.109377, -2.133886, 0.821642),
                            (q3, 0.998324, 19632591, 16.87177, 1.324509), (q4, 0.999458, 5883709, 15.82984, 1.938497)]:
    check(q.name + " R2", q["R-squared"], r2, 5e-6)
    check(q.name + " SSR", q["Sum squared resid"], ssr, max(1, ssr * 1e-6))
    check(q.name + " AIC", q["Akaike info criterion"], aic, 5e-5)
    check(q.name + " DW", q["Durbin-Watson stat"], dw, 5e-6)

if __name__ == "__main__":
    bad = 0
    for lab, ref, got, ok in rows:
        bad += not ok
        print("%-28s %16.6f %16.6f  %s" % (lab, ref, got, "OK" if ok else "!!"))
    print("\nперевірок: %d, розбіжностей: %d" % (len(rows), bad))
    if len(sys.argv) > 2:
        import json
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump([{"label": l, "ref": float(r), "got": float(g), "ok": bool(o)} for l, r, g, o in rows], f,
                      ensure_ascii=False, indent=1)
    print("\n" + eviews_text(e4))
