# ЛР-4. Аналіз часових рядів. Бригада 15 (варіант 1): RSAFSNA, CURRNS
# Точка входу проєкту Visual Studio: будує моделі, прогнози і виводить результати та висновки в консоль
import sys
import lab4

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LINE = "=" * 104
NAMES = {"trend1": "Тренд 1-го порядку", "trend2": "Тренд 2-го порядку", "trend3": "Тренд 3-го порядку",
         "arima": "АРІКС", "ar": "АР", "arma": "АРКС"}


def title(t):
    print("\n" + LINE + "\n" + t + "\n" + LINE)


def sub(t):
    print("\n--- " + t + " " + "-" * max(0, 99 - len(t)))


def pause():
    try:
        input("\nНатисніть Enter, щоб продовжити...")
    except EOFError:
        pass


def corr_print(c, label, L=24):
    print("%s: T = %d, межа значущості +/-2/sqrt(T) = %.3f" % (label, c["n"], c["band"]))
    print("%4s %8s %8s %10s %7s" % ("лаг", "AC", "PAC", "Q-Stat", "Prob"))
    for i in range(min(L, len(c["AC"]))):
        m = ("  AC*" if abs(c["AC"][i]) > c["band"] else "     ") + (" PAC*" if abs(c["PAC"][i]) > c["band"] else "")
        print("%4d %8.3f %8.3f %10.2f %7.3f%s" % (i + 1, c["AC"][i], c["PAC"][i], c["Q"][i], c["Prob"][i], m))


def steps_print(steps):
    for s in steps:
        print("  ar=%-22s ma=%-22s Akaike %8.4f  Schwarz %8.4f  max Prob %.3f (%s)"
              % (s["ar"], s["ma"], s["aic"], s["sc"], s["pmax"], s["worst"]))


def fc_table(S, kind):
    keys = list(S[kind].keys())
    print("%-9s %10s" % ("Час", "logy") + "".join("%13s" % NAMES[k][:12] for k in keys))
    for i, lab in enumerate(S["labels_test"]):
        print("%-9s %10.5f" % (lab, S["actual_log"][i]) + "".join("%13.5f" % S[kind][k]["fc"][i] for k in keys))
    for m, f in (("RMSE", "%13.5f"), ("MAE", "%13.5f"), ("MAPE", "%13.4f"), ("Theil", "%13.6f"),
                 ("Bias", "%13.4f"), ("Variance", "%13.4f"), ("Covariance", "%13.4f")):
        print("%-20s" % m + "".join(f % S[kind][k]["eval"][m] for k in keys))
    best = min(keys, key=lambda k: S[kind][k]["eval"]["RMSE"])
    print("Найменша RMSE: %s (%.5f); MAPE у вихідних одиницях %.3f %%"
          % (NAMES[best], S[kind][best]["eval"]["RMSE"], S[kind][best]["eval_y"]["MAPE"]))
    return best


def part0():
    title("ПЕРЕВІРКА: відтворення прикладу методички (агрегат M1 США)")
    import verify_manual as vm
    bad = 0
    for lab, ref, got, ok in vm.rows:
        bad += not ok
        print("%-46s методичка %16.6f   отримано %16.6f   %s" % (lab, ref, got, "OK" if ok else "!!"))
    print("\nПеревірок: %d, розбіжностей: %d" % (len(vm.rows), bad))


def series(S):
    title("РЯД %s: %s" % (S["name"], S["desc"]))
    print("Дані: %s - %s, %d значень. Навчальна вибірка %s - %s (%d), перевірочна %s - %s (%d)."
          % (S["first"], S["last"], S["N"], S["train"][0], S["train"][1], S["NT"], S["test"][0], S["test"][1], S["H"]))
    print("Графіки ряду та logy: out/series_%s.png" % S["name"])

    sub("Тренди для logy (навчальна вибірка)")
    for d in S["trends"]:
        print(d["window"])
    sub("Сезонність: АКФ відхилень logy від тренду 2-го порядку")
    c = S["corr_detr"]
    print("AC(12) = %.3f, AC(24) = %.3f, AC(36) = %.3f -> сезонний ефект з періодом 12 місяців"
          % (c["AC"][11], c["AC"][23], c["AC"][35]))

    sub("АРІКС: корелограма dlogy = d(logy)")
    corr_print(S["corr_dlogy"], "dlogy", 36)
    A = S["arima"]
    print("Значущі лаги ЧАКФ 1..12: %s" % A["ar0"])
    print("Вилучення незначущих АР-складових:")
    steps_print(A["steps_ar"])
    print("Значущі лаги АКФ залишків 1..12 -> КС-складові: %s" % A["ma0"])
    steps_print(A["steps_arma"])
    print(A["model"]["window"])
    print("R2 відносно logy: %.6f; Q(24) залишків = %.2f, Prob = %.3f" % (A["R2_logy"], A["model"]["Q24"], A["model"]["Q24p"]))

    sub("АР-модель для logy")
    corr_print(S["corr_logy"], "logy", 13)
    print("Значущі лаги ЧАКФ 1..13: %s" % S["ar"]["lags0"])
    steps_print(S["ar"]["steps"])
    print(S["ar"]["model"]["window"])

    sub("АРКС-модель для logy")
    print("Значущі лаги АКФ залишків АР-моделі 1..12: %s" % S["arma"]["ma0"])
    steps_print(S["arma"]["steps"])
    print(S["arma"]["model"]["window"])

    sub("Статистичні характеристики моделей")
    print("%-22s %10s %14s %10s %10s %10s" % ("модель", "R2", "SSR", "DW", "Akaike", "Schwarz"))
    for d in S["trends"] + [A["model"], S["ar"]["model"], S["arma"]["model"]]:
        s = d["stats"]
        print("%-22s %10.6f %14.6f %10.4f %10.4f %10.4f" % (d["name"], s["R-squared"], s["Sum squared resid"],
              s["Durbin-Watson stat"], s["Akaike info criterion"], s["Schwarz criterion"]))

    sub("Статичний прогноз на %d кроків (графік: out/fc_static_%s.png)" % (S["H"], S["name"]))
    bs = fc_table(S, "static")
    sub("Динамічний прогноз на %d кроків (графік: out/fc_dynamic_%s.png)" % (S["H"], S["name"]))
    bd = fc_table(S, "dynamic")
    print("\nВисновок: при статичному прогнозуванні найкраща модель - %s, при динамічному - %s."
          % (NAMES[bs], NAMES[bd]))


def main():
    print("Лабораторна робота №4 з АЧР. Обчислення...")
    res = lab4.run_all()
    print("Готово. Результати також збережено в out/results.json")
    actions = {"0": part0, "1": lambda: series(res["series"][0]), "2": lambda: series(res["series"][1])}
    menu = ("\nМеню:\n  0 - перевірка на прикладі методички\n  1 - ряд RSAFSNA\n  2 - ряд CURRNS\n"
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
