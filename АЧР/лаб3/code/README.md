# ЛР-3 з АЧР — проєкт Visual Studio

Бригада №15 (варіант 1): ряди rts1, rts1996; частини 3–4 — галузеві індекси RTS.

## Запуск у Visual Studio 2022
1. Відкрити `ATS_Lab3.sln` (потрібне навантаження «Python development»).
2. Solution Explorer → Python Environments → середовище → «Install from requirements.txt» (numpy, scipy, matplotlib).
3. F5 (або Ctrl+F5). Стартовий файл — `main.py`.

## Запуск у VS Code / терміналі
```
pip install -r requirements.txt
python main.py        # меню
python main.py A      # усе підряд
python main.py 2 4    # окремі пункти
```

## Меню
| Пункт | Що виводиться |
|---|---|
| 0 | перевірка на прикладах методички (62 значення, DW = 1,22881) |
| 1 | частина 1: статистика Дарбіна–Уотсона |
| 2 / 3 | частина 2 для rts1 / rts1996: описова статистика, АКФ/ЧАКФ → p, АР(p), ЧАКФ залишків → q, таблиці 2, 2а, 3, вікно найкращої моделі, висновок |
| 4 | частина 3: кореляційна матриця, рівняння множинної регресії |
| 5 | частина 4: ЧКФ, моделі ARMAX |

Після запуску в `out/` з'являються `results.json` і гістограми `hist_*.png`.

## Файли
- `main.py` — точка входу, вивід результатів і висновків
- `lab3.py` — обчислення частин 1–4
- `ats.py` — відтворення процедур EViews (ls, ar/ma з бекастингом, correl, Histogram and Stats)
- `verify_manual.py` — перевірка на контрольних прикладах методички
- `eviews/` — програми EViews (*.prg)
- `data/` — дані з архіву ATS_lab_03_new
