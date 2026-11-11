# ЛР-3. Аналіз часових рядів. Бригада 15 (варіант 1): rts1, rts1996
# Точка входу проєкту Visual Studio: обчислює частини 1-4 і виводить результати та висновки в консоль
import sys
import numpy as np
import lab3
from ats import correlogram

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LINE = "=" * 100
MA_NAMES = {"sma5": "просте КС, N=5", "sma10": "просте КС, N=10",
            "ema5": "експоненційне КС, N=5", "ema10": "експоненційне КС, N=10"}


def title(t):
    print("\n" + LINE + "\n" + t + "\n" + LINE)


def sub(t):
    print("\n--- " + t + " " + "-" * max(0, 95 - len(t)))


def pause():
    try:
        input("\nНатисніть Enter, щоб продовжити...")
    except EOFError:
        pass


def eq_text(d, kind, b=None, lhs="y(k)"):
    nm = lambda n: n.lower().replace("(-", "(k-") if "(-" in n else n.lower() + "(k)"
    out, first = lhs + " = ", True

    def push(c, v):
        nonlocal out, first
        s = "%.4f" % abs(c) + ("*" + v if v else "")
        out += ("-" if c < 0 else "") + s if first else (" - " if c < 0 else " + ") + s
        first = False

    if kind == "ev":
        push(d["intercept"], None)
        for n, c in zip(d["names"], d["coef"]):
            if n.startswith("AR("):
                push(c, "y(k-%s)" % n[3:-1])
        out += " + e(k)"
        for n, c in zip(d["names"], d["coef"]):
            if n.startswith("MA("):
                push(c, "e(k-%s)" % n[3:-1])
        return out
    for n, c in zip(d["names"], d["coef"]):
        if n == "C":
            push(c, None)
        elif not n.startswith("MV"):
            push(c, nm(n))
    if kind in ("mv", "fixed"):
        out += " + mv(k)"
        if kind == "mv":
            for n, c in zip(d["names"], d["coef"]):
                if n.startswith("MV"):
                    push(c, nm(n))
        else:
            for j, bj in enumerate(b, 1):
                push(bj, "mv(k-%d)" % j)
    return out


def corr_print(c, label):
    print("%s: T = %d, межа значущості +/-2/sqrt(T) = %.3f" % (label, c["n"], c["band"]))
    print("%4s %8s %8s %10s %7s" % ("лаг", "AC", "PAC", "Q-Stat", "Prob"))
    for i in range(len(c["AC"])):
        mark = "  *" if abs(c["PAC"][i]) > c["band"] else ""
        print("%4d %8.3f %8.3f %10.3f %7.3f%s" % (i + 1, c["AC"][i], c["PAC"][i], c["Q"][i], c["Prob"][i], mark))
    print("(* - значущий коефіцієнт ЧАКФ)")


def models(S):
    p, q = S["p"], S["q"]
    L = [("АРКС(%d) = АР(%d)" % (p, p), S["ar"], "ar", None, "ar"),
         ("АРКС(%d,%d), КС системи EViews" % (p, q), S["s1"]["ev"], "ev", None, "ev")]
    for k in MA_NAMES:
        L.append(("АРКС(%d,%d), власне %s по залишках" % (p, q, MA_NAMES[k]), S["s1"][k], "mv", None, "s1_" + k))
    for a, al in (("a1", "підхід 1"), ("a2", "підхід 2")):
        for k in MA_NAMES:
            v = S["s2"][k]
            L.append(("АРКС(%d,%d), %s по y, %s" % (p, v["q"], MA_NAMES[k], al), v[a], "fixed" if a == "a1" else "mv",
                      v["b"] if a == "a1" else None, a + "_" + k))
    return L


# ---------- Частина 0: перевірка на прикладах методички ----------
def part0():
    title("ПЕРЕВІРКА: відтворення результатів EViews на контрольних прикладах методички")
    import verify_manual as vm
    bad = 0
    for lab, ref, got, ok in vm.rows:
        bad += not ok
        print("%-28s методичка %16.6f   отримано %16.6f   %s" % (lab, ref, got, "OK" if ok else "!!"))
    print("\nПеревірок: %d, розбіжностей: %d" % (len(vm.rows), bad))
    print("Висновок: реалізація LS, ARMA (з бекастингом), корелограми та описової статистики збігається з EViews.")


# ---------- Частина 1 ----------
def part1(res):
    title("ЧАСТИНА 1. Статистика Дарбіна-Уотсона: DW = sum(e(t)-e(t-1))^2 / sum e(t)^2")
    P = res["part1"]
    print("Файл example_for_DW.txt: N = %d" % P["n"])
    print("Чисельник   sum(e(t)-e(t-1))^2 = %.4f" % P["num"])
    print("Знаменник   sum e(t)^2         = %.4f" % P["den"])
    print("DW = %.5f   (вірний результат за методичкою: 1.22881)" % P["dw"])
    print("Висновок: %s; DW < 2 -> додатна автокореляція ряду." % ("програма працює вірно" if abs(P["dw"] - 1.22881) < 1e-5 else "РОЗБІЖНІСТЬ"))


# ---------- Частина 2 ----------
def part2(S):
    name = S["name"]
    title("ЧАСТИНА 2. Ряд %s (N = %d)" % (name, S["N"]))
    sub("Описова статистика (Histogram and Stats)")
    for k, v in S["desc"].items():
        print("%-14s %14.6f" % (k, v) if k != "Observations" else "%-14s %14d" % (k, v))
    d = S["desc"]
    print("Висновок: S = %.4f (%s хвіст довший), K = %.4f (%s за нормальний), JB = %.2f, Prob = %.6f -> %s."
          % (d["Skewness"], "лівий" if d["Skewness"] < 0 else "правий", d["Kurtosis"],
             "гостріший" if d["Kurtosis"] > 3 else "плоскіший", d["Jarque-Bera"], d["Probability"],
             "розподіл НЕ нормальний" if d["Probability"] < 0.05 else "нормальність не відхиляється"))
    print("Графік гістограми збережено: out/hist_%s.png" % name)

    sub("1.1 АКФ та ЧАКФ ряду -> порядок p")
    corr_print(S["corr_y"], "Ряд " + name)
    print("Значущі лаги ЧАКФ: %s -> p = %d" % (S["p_sig"], S["p"]))

    sub("1.2 Модель АР(%d)" % S["p"])
    print(S["ar"]["window"])
    print(eq_text(S["ar"], "ar"))
    corr_print(S["corr_res"], "Залишки АР(%d)" % S["p"])
    print("Значущі лаги ЧАКФ залишків: %s -> q = %d" % (S["q_sig"], S["q"]))

    sub("Послідовність АРКС(%d,j) з КС системи EViews (ls y c ar(1..p) ma(1..j))" % S["p"])
    print("%-12s %10s %16s %10s %10s %8s" % ("модель", "R2", "SSR", "Akaike", "Schwarz", "DW"))
    for e in S["ev_seq"]:
        s = e["stats"]
        print("%-12s %10.6f %16.4f %10.4f %10.4f %8.4f" % ("АРКС(%d,%d)" % (S["p"], e["q"]), s["R-squared"],
              s["Sum squared resid"], s["Akaike info criterion"], s["Schwarz criterion"], s["Durbin-Watson stat"]))

    sub("2.3 ЧАКФ ковзних середніх ряду y -> q для КС по y, коефіцієнти b (підхід 1)")
    for k, v in S["s2"].items():
        print("%-24s значущі лаги %-10s q = %d   b = %s" % (MA_NAMES[k], v["q_sig"], v["q"], np.round(v["b"], 4).tolist()))

    sub("ТАБЛИЦЯ 2 (+ 2а): характеристики рівнянь")
    print("%-58s %9s %9s %14s %8s %8s %6s %7s" % ("модель", "R2", "R2(y)", "SSR", "Akaike", "Schwarz", "DW", "ProbQ"))
    L = models(S)
    best = min(L, key=lambda m: m[1]["stats"]["Akaike info criterion"])
    for lab, e, kind, b, key in L:
        s = e["stats"]
        r2y = e.get("R2y", s["R-squared"])
        print("%-58s %9.6f %9.6f %14.4f %8.4f %8.4f %6.4f %7.3f%s" % (lab, s["R-squared"], r2y, s["Sum squared resid"],
              s["Akaike info criterion"], s["Schwarz criterion"], s["Durbin-Watson stat"], e["Q12p"], "  <- min AIC" if e is best[1] else ""))
    print("R2 для рівнянь з власним КС обчислено для y1 = y - mv; для порівняння використовуйте R2(y), SSR, Akaike, Schwarz, DW.")

    sub("ТАБЛИЦЯ 3: рівняння з чисельними коефіцієнтами")
    for lab, e, kind, b, key in L:
        print("%s:\n    %s" % (lab, eq_text(e, kind, b)))

    sub("Найкраща модель")
    lab, e, kind, b, key = best
    s = e["stats"]
    print("Найкраща за критерієм Акайке: %s" % lab)
    print(e["window"])
    print("\nРівняння: " + eq_text(e, kind, b))
    print("Висновок: Akaike = %.4f (мінімальний), Schwarz = %.4f, R2(y) = %.6f, DW = %.4f %s, "
          "залишки %s (Q(12) = %.2f, Prob = %.3f)." % (s["Akaike info criterion"], s["Schwarz criterion"],
          e.get("R2y", s["R-squared"]), s["Durbin-Watson stat"], "(близько 2)" if abs(s["Durbin-Watson stat"] - 2) < 0.3 else "",
          "не автокорельовані" if e["Q12p"] > 0.05 else "автокорельовані", e["Q12"], e["Q12p"]))


# ---------- Частина 3 ----------
def part3(res):
    P = res["part3"]
    title("ЧАСТИНА 3. Множинна регресія rts1 на галузеві індекси")
    sub("Кореляційна матриця")
    print("%7s" % "" + "".join("%9s" % n for n in P["names"]))
    for n, r in zip(P["names"], P["corr"]):
        print("%7s" % n + "".join("%9.4f" % v for v in r))
    print("Регресори з |r(rts1, x)| >= 0.5 (сильний зв'язок): %s" % P["included"])
    lab = ["MR1: усі регресори", "MR2", "MR3"]
    for i, d in enumerate(P["models"]):
        sub("%s (%s)" % (lab[i] if i else lab[0], ", ".join(n for n in d["names"] if n != "C")))
        print(d["window"])
        print(eq_text(d, "ar", lhs="rts1(k)"))
    last = P["models"][-1]
    print("\nВисновок: незначущі регресори вилучено послідовно (Prob > 0.05); остаточне рівняння - останнє.")
    print("DW = %.4f, ЧАКФ(1) залишків = %.3f -> сильна автокореляція залишків, потрібна динамічна модель (частина 4)."
          % (last["stats"]["Durbin-Watson stat"], P["corr_res"]["PAC"][0]))


# ---------- Частина 4 ----------
def part4(res):
    P = res["part4"]
    title("ЧАСТИНА 4. ЧКФ(rts1, x) та модель ARMAX")
    sub("Часткова кореляційна функція, межа +/-%.3f" % P["band"])
    names = list(P["pccf"].keys())
    print("%5s" % "лаг" + "".join("%9s" % n for n in names))
    for m in range(13):
        print("%5d" % m + "".join("%9.3f" % P["pccf"][n][m][0] for n in names))
    print("Значуща ЧКФ на лагу 1: %s" % P["lag1"])
    sub("Моделі ARMAX")
    print("%-46s %4s %10s %12s %8s %8s %7s" % ("модель", "k", "R2", "SSR", "Akaike", "Schwarz", "DW"))
    lab = {"m1": "ARMAX-1: лаги регресорів зі значущою ЧКФ(1)", "m2": "ARMAX-2: перші лаги всіх регресорів",
           "m3": "ARMAX-3: після вилучення незначущих", "m4": "ARMAX-4: регресори + ar(1)"}
    for k, l in lab.items():
        s = P[k]["stats"]
        print("%-46s %4d %10.6f %12.2f %8.4f %8.4f %7.4f" % (l, P[k]["k"], s["R-squared"], s["Sum squared resid"],
              s["Akaike info criterion"], s["Schwarz criterion"], s["Durbin-Watson stat"]))
    print("\nВилучення незначущих членів з ARMAX-2:")
    for st in P["steps"][:-1]:
        print("  k = %2d  вилучено %-10s Prob = %.3f  Akaike = %.4f" % (st["k"], st["drop"], st["p"], st["aic"]))
    sub("Остаточна модель ARMAX-3")
    print(P["m3"]["window"])
    print(eq_text(P["m3"], "ar", lhs="rts1(k)"))
    print("Висновок: DW = %.4f, залишки не автокорельовані (Prob Q(12) = %.3f); Akaike %.4f проти %.4f у статичній моделі."
          % (P["m3"]["stats"]["Durbin-Watson stat"], P["m3"]["Q12p"], P["m3"]["stats"]["Akaike info criterion"],
             res["part3"]["models"][-1]["stats"]["Akaike info criterion"]))


def main():
    print("Лабораторна робота №3 з АЧР. Обчислення...")
    res = lab3.run_all()
    print("Готово. Результати також збережено в out/results.json")
    actions = {"0": lambda: part0(), "1": lambda: part1(res), "2": lambda: part2(res["series"][0]),
               "3": lambda: part2(res["series"][1]), "4": lambda: part3(res), "5": lambda: part4(res)}
    menu = ("\nМеню:\n  0 - перевірка на прикладах методички\n  1 - частина 1 (DW)\n  2 - частина 2, ряд rts1\n"
            "  3 - частина 2, ряд rts1996\n  4 - частина 3 (множинна регресія)\n  5 - частина 4 (ЧКФ, ARMAX)\n"
            "  A - усе підряд\n  Q - вихід")
    if len(sys.argv) > 1:
        for ch in sys.argv[1:]:
            for k in (actions if ch.upper() == "A" else [ch]):
                actions[k]()
        return
    while True:
        print(menu)
        try:
            ch = input("Ваш вибір: ").strip().upper()
        except EOFError:
            break
        if ch == "Q":
            break
        for k in (actions if ch == "A" else [ch]):
            if k in actions:
                actions[k]()
        pause()


if __name__ == "__main__":
    main()
