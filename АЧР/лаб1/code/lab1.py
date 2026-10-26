"""
Лабораторна робота №1. Оцінювання параметрів АРКС(p,q) за допомогою МНК та РМНК.
Бригада №1:  a0=0, a1=0.22, a2=-0.18, a3=0.08, b1=0.5, b2=0.25, b3=0.25

Модель об'єкта (АРКС(3,3)):
    y(k) = a0 + a1 y(k-1) + a2 y(k-2) + a3 y(k-3) + v(k) + b1 v(k-1) + b2 v(k-2) + b3 v(k-3)
Випадкова складова вводиться в рівняння ЛИШЕ ОДИН РАЗ - через v(k) (додаткової похибки eps = 0).

Режими запуску
--------------
1) Повна лабораторна (за замовчуванням, об'єкт - бригада №1): 9 моделей АРКС(1,1)...АРКС(3,3),
   таблиця S / R2 / IKA, графіки перехідних процесів та критеріїв
       python lab1.py

2) Власний об'єкт: користувач задає ПОРЯДОК АВТОРЕГРЕСІЇ p, ПОРЯДОК КОВЗНОГО СЕРЕДНЬОГО q та коефіцієнти
   a0 a1 a2 a3 b1 b2 b3 (коефіцієнти при затримках, більших за p та q, ігноруються, тобто = 0).
   Оцінюються всі 9 моделей, графіки перехідних процесів будуються для моделі АРКС(p,q):
       python lab1.py --p 2 --q 1 --coeffs 0 0.22 -0.18 0.08 0.5 0.25 0.25

3) Перевірка по тестових даних викладача (файли v.txt, y.txt, test.txt):
       python lab1.py --v-file Test+/v.txt --y-file Test+/y.txt --coef-file Test+/test.txt

Додаткові параметри: --seed, --beta, --out
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- Варіант (бригада) 1 ----------------
A0, A1, A2, A3 = 0.0, 0.22, -0.18, 0.08
B1, B2, B3 = 0.5, 0.25, 0.25
THETA_TRUE = np.array([A0, A1, A2, A3, 1.0, B1, B2, B3])   # (a0,a1,a2,a3,1,b1,b2,b3)

MARGIN = 3          # спільний початок вибірки (k = MARGIN...) для всіх моделей
N_EFF = 100         # кількість рядків матриці вимірів (N >= 100 за умовою)
N = N_EFF + MARGIN  # довжина згенерованих рядів v та y
SEED = 2024         # зерно генератора псевдовипадкових чисел
BETA = 300.0        # P0 = beta * I (за методичкою: "кілька сотень")
OUT_DIR = 'output'
ALL_ORDERS = [(p, q) for p in (1, 2, 3) for q in (1, 2, 3)]


# ---------------- Генерація даних ----------------
def white_noise(n, seed=SEED):
    """Білий шум: eps = sum(12 рівномірних U(0,1)) - 6  ->  нормальний, M=0, D=1.
    (Сума 12 величин U(0,1) має M=6, D=1; для U(-1,1) дисперсія була б 4, тому береться [0;1].)"""
    rng = np.random.default_rng(seed)
    return rng.random((n, 12)).sum(axis=1) - 6.0


def make_theta(coeffs, p=3, q=3):
    """theta = (a0,a1,a2,a3,1,b1,b2,b3) для об'єкта порядку АРКС(p,q):
    коефіцієнти при затримках, більших за p та q, обнуляються."""
    c = list(coeffs)
    a = [c[i] if i <= p else 0.0 for i in (1, 2, 3)]
    b = [c[3 + i] if i <= q else 0.0 for i in (1, 2, 3)]
    return np.array([c[0], *a, 1.0, *b])


def generate_y(theta, v):
    """АРКС(3,3): y(k)=a0+a1y(k-1)+a2y(k-2)+a3y(k-3)+v(k)+b1v(k-1)+b2v(k-2)+b3v(k-3)
    (випадкова складова вводиться лише один раз - через v(k); y(k)=0, v(k)=0 при k<0)"""
    a0, a1, a2, a3, _, b1, b2, b3 = theta
    y = np.zeros(len(v))
    for k in range(len(v)):
        y[k] = a0 + v[k]
        for i, a in enumerate((a1, a2, a3), start=1):
            if k - i >= 0:
                y[k] += a * y[k - i]
        for i, b in enumerate((b1, b2, b3), start=1):
            if k - i >= 0:
                y[k] += b * v[k - i]
    return y


def generate_X(p, q, v, y):
    """Матриця вимірів: рядок k = (1, y(k-1..k-p), v(k), v(k-1..k-q)), k = MARGIN..len(y)-1"""
    rows = []
    for k in range(MARGIN, len(y)):
        r = [1.0] + [y[k - i] for i in range(1, p + 1)]
        r += [v[k]] + [v[k - i] for i in range(1, q + 1)]
        rows.append(r)
    return np.array(rows), y[MARGIN:]


def param_names(p, q):
    return ['a0'] + ['a%d' % i for i in range(1, p + 1)] + ['v(k)'] + ['b%d' % i for i in range(1, q + 1)]


def true_values(theta_true, p, q):
    """Фактичні значення параметрів у порядку param_names(p, q)"""
    return [theta_true[0]] + list(theta_true[1:1 + p]) + [theta_true[4]] + list(theta_true[5:5 + q])


def check_stability(theta):
    """Достатня умова стійкості: sum|a_i| < 1 (a0 не враховується)"""
    s = float(np.sum(np.abs(theta[1:4])))
    return s, s < 1.0


# ---------------- Завантаження даних з файлів ----------------
def _to_float(s):
    return float(s.replace(',', '.'))


def load_series(path):
    """Завантаження часового ряду з текстового файлу (одне число в рядку; допускається кома)"""
    with open(path, encoding='utf-8-sig') as f:
        data = f.read().split()
    return np.array([_to_float(s) for s in data])


def load_coeffs(path):
    """Файл вигляду a0=0.005 / a1=0.1 / ... / b1=0.01"""
    c = {}
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            if '=' in line:
                k, val = line.strip().split('=', 1)
                c[k.strip()] = _to_float(val.strip())
    return c


def orders_from_coeffs(c):
    """Порядки АР та КС за наявними ключами a1..a3 / b1..b3"""
    p = max([i for i in (1, 2, 3) if 'a%d' % i in c] or [1])
    q = max([i for i in (1, 2, 3) if 'b%d' % i in c] or [1])
    return p, q


def align_series(v, y):
    """Якщо v довший за y (як у тестових даних: 100 проти 97), то y(k) відповідає
    останнім len(y) значенням v - відкидаємо зайві перші значення v."""
    if len(v) > len(y):
        v = v[len(v) - len(y):]
    elif len(y) > len(v):
        y = y[len(y) - len(v):]
    return v, y


# ---------------- МНК ----------------
COND_LIMIT = 1e10   # cond(X^T X) вище цієї межі - матриця вважається виродженою (колінеарні стовпці)


def is_singular(X):
    """X^T X вироджена (або майже вироджена): тоді (X^T X)^-1 не існує / числово недостовірна.
    Це трапляється, коли модель має більший порядок, ніж об'єкт (стовпці X лінійно залежні)."""
    return np.linalg.cond(X.T @ X) > COND_LIMIT


def mnk(X, Y):
    """theta = (X^T X)^-1 X^T Y.
    Якщо X^T X вироджена - обернену знайти неможливо, тому повертається розв'язок із
    псевдооберненою матрицею (мінімальної норми); у таблиці такі моделі позначаються '*'."""
    if is_singular(X):
        return np.linalg.pinv(X) @ Y
    try:
        return np.linalg.inv(X.T @ X) @ X.T @ Y
    except np.linalg.LinAlgError:
        return np.linalg.pinv(X) @ Y


def mnk_history(X, Y):
    """Оцінки МНК за перші i+1 рядків матриці X (NaN, поки X^T X вироджена / рядків менше, ніж параметрів)"""
    m = X.shape[1]
    hist = np.full((len(Y), m), np.nan)
    for i in range(len(Y)):
        if i + 1 >= m:
            hist[i] = mnk(X[:i + 1], Y[:i + 1])
    return hist


# ---------------- РМНК (без обернення матриці) ----------------
def rmnk(X, Y, beta=BETA):
    """theta(0)=0, P(0)=beta*I, beta >> 0;
       P(i) = P(i-1) - P(i-1)x^T x P(i-1) / (1 + x P(i-1) x^T)
       theta(i) = theta(i-1) + P(i) x^T (y(i) - x theta(i-1))"""
    m = X.shape[1]
    theta = np.zeros(m)
    P = np.identity(m) * beta
    hist = np.zeros((len(Y), m))
    for i in range(len(Y)):
        x = X[i, :].reshape(1, m)
        P = P - (P @ x.T @ x @ P) / (1 + x @ P @ x.T)
        e = Y[i] - x @ theta
        theta = theta + (P @ x.T * e).ravel()
        hist[i] = theta
    return theta, hist


# ---------------- Критерії якості ----------------
def criteria(X, Y, theta, p, q):
    """S = sum e^2;  R2 = var(y_hat)/var(y);  IKA = N ln(S) + 2n,  n = p+q+1"""
    Yh = X @ theta
    e = Y - Yh
    S = float(e @ e)
    R2 = float(np.var(Yh) / np.var(Y))
    n = p + q + 1
    IKA = len(Y) * np.log(max(S, 1e-300)) + 2 * n
    return S, R2, IKA


# ---------------- Допоміжні функції виводу ----------------
def print_table(res):
    print('%-10s | %-30s | %-30s' % ('', 'МНК', 'РМНК'))
    print('%-10s | %9s %10s %9s | %9s %10s %9s' % ('', 'S', 'R2', 'IKA', 'S', 'R2', 'IKA'))
    singular = False
    for (p, q), r in res.items():
        mark = '*' if r.get('singular') else ' '
        singular = singular or bool(r.get('singular'))
        print('АРКС(%d,%d)%s | %9.4g %10.6f %9.1f | %9.4g %10.6f %9.1f' % ((p, q, mark) + r['mnk'] + r['rmnk']))
    if singular:
        print('* матриця X^T X вироджена (колінеарні стовпці: порядок моделі вищий за порядок об\'єкта);')
        print('  для МНК використано псевдообернену матрицю, оцінки неоднозначні, S/R2/IKA для МНК недостовірні.')


def print_best(res):
    """Найкраща модель за кожним критерієм (S - min, R2 - max, IKA - min)"""
    for method in ('mnk', 'rmnk'):
        # для МНК моделі з виродженою X^T X (позначені '*') у порівнянні не беруть участі
        cand = {pq: r for pq, r in res.items() if not (method == 'mnk' and r.get('singular'))}
        best_s = min(cand, key=lambda pq: cand[pq][method][0])
        best_r = max(cand, key=lambda pq: cand[pq][method][1])
        best_i = min(cand, key=lambda pq: cand[pq][method][2])
        print('%-4s: min S -> АРКС(%d,%d); max R2 -> АРКС(%d,%d); min IKA -> АРКС(%d,%d)'
              % ((method.upper(),) + best_s + best_r + best_i))


def print_estimates(res, theta_true=None):
    for (p, q), r in res.items():
        names = param_names(p, q)
        print('АРКС(%d,%d)  параметри %s' % (p, q, names))
        if theta_true is not None:
            print('   факт.  ', np.round(true_values(theta_true, p, q), 6))
        print('   МНК    ', np.round(r['t1'], 6))
        print('   РМНК   ', np.round(r['t2'], 6))


def estimate_models(v, y, orders, beta=BETA):
    res = {}
    for p, q in orders:
        X, Y = generate_X(p, q, v, y)
        t1 = mnk(X, Y)
        t2, _ = rmnk(X, Y, beta)
        res[(p, q)] = dict(mnk=criteria(X, Y, t1, p, q), rmnk=criteria(X, Y, t2, p, q),
                           t1=t1, t2=t2, X=X, Y=Y, singular=is_singular(X))
    return res


# ---------------- Графіки ----------------
def plot_transient(v, y, theta_true, p, q, out_dir, beta=BETA):
    """Рис. 1-7: перехідний процес оцінок кожного коефіцієнта моделі АРКС(p,q).
    Крім a0..a3, b1..b3 будується й коефіцієнт при v(k) (має прямувати до 1)."""
    X, Y = generate_X(p, q, v, y)
    h_mnk = mnk_history(X, Y)
    _, h_rmnk = rmnk(X, Y, beta)
    k = np.arange(MARGIN, MARGIN + len(Y))
    names = param_names(p, q)
    true_full = true_values(theta_true, p, q)
    for j, name in enumerate(names):
        label = 'coef_v' if name == 'v(k)' else name
        plt.figure()
        plt.plot(k, np.full(len(k), true_full[j]), 'k', label='Фактична')
        plt.plot(k, h_mnk[:, j], 'r', label='МНК')
        plt.plot(k, h_rmnk[:, j], 'b', label='РМНК')
        plt.title('Оцінка %s' % ('коефіцієнта при v(k)' if name == 'v(k)' else name))
        plt.xlabel('k')
        plt.ylabel(name)
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'coef_%s.png' % label.replace('coef_', '')), dpi=200)
        plt.close()


def plot_criteria(res, out_dir):
    """Рис. 8: S, R2, IKA залежно від порядку моделі (моделі впорядковані за кількістю параметрів n=p+q+1)"""
    keys = sorted(res, key=lambda pq: (pq[0] + pq[1], pq[0]))
    labels = ['АРКС(%d,%d)\nn=%d' % (p, q, p + q + 1) for p, q in keys]
    x = np.arange(len(keys))
    fig, axs = plt.subplots(1, 3, figsize=(15.68, 6.72))
    for ax, idx, title in zip(axs, range(3), ('S', 'R2', 'IKA')):
        ax.plot(x, [res[pq]['mnk'][idx] for pq in keys], 'o-', label='МНК')
        ax.plot(x, [res[pq]['rmnk'][idx] for pq in keys], 'o-', label='РМНК')
        if title == 'S':
            ax.set_yscale('log')
        ax.set_title('Значення %s для МНК та РМНК' % title)
        ax.set_xlabel('Модель (за зростанням кількості параметрів n)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'criteria.png'), dpi=100)
    plt.close()


# ---------------- Режими роботи ----------------
def run_full_lab(args):
    os.makedirs(args.out, exist_ok=True)
    s, ok = check_stability(THETA_TRUE)
    print('Умова стійкості: |a1|+|a2|+|a3| = %.2f %s 1' % (s, '<' if ok else '>='))
    v = white_noise(N, args.seed)
    y = generate_y(THETA_TRUE, v)
    res = estimate_models(v, y, ALL_ORDERS, args.beta)
    print('Довжина рядів N = %d, рядків матриці вимірів = %d (k = %d..%d)' % (N, N - MARGIN, MARGIN, N - 1))
    print_table(res)
    print_best(res)
    print('theta true ', THETA_TRUE)
    print('(3,3) МНК  ', res[(3, 3)]['t1'])
    print('(3,3) РМНК ', res[(3, 3)]['t2'])
    plot_transient(v, y, THETA_TRUE, 3, 3, args.out, args.beta)
    plot_criteria(res, args.out)
    print('Графіки збережено в папці', args.out)


def run_custom(args):
    """Користувач сам задає порядки p, q ОБ'ЄКТА та його коефіцієнти"""
    os.makedirs(args.out, exist_ok=True)
    c = args.coeffs if args.coeffs else [A0, A1, A2, A3, B1, B2, B3]
    theta = make_theta(c, args.p, args.q)
    s, ok = check_stability(theta)
    print('Об\'єкт АРКС(%d,%d), theta = %s' % (args.p, args.q, theta))
    print('Умова стійкості: |a1|+|a2|+|a3| = %.4f %s 1' % (s, '<' if ok else '>='))
    if not ok:
        print('УВАГА: достатня умова стійкості не виконується')
    v = white_noise(N, args.seed)
    y = generate_y(theta, v)
    res = estimate_models(v, y, ALL_ORDERS, args.beta)
    print_table(res)
    print_best(res)
    r = res[(args.p, args.q)]
    print('Оцінки для моделі АРКС(%d,%d) того ж порядку, що й об\'єкт:' % (args.p, args.q))
    print('   МНК  theta =', r['t1'])
    print('   РМНК theta =', r['t2'])
    plot_transient(v, y, theta, args.p, args.q, args.out, args.beta)
    plot_criteria(res, args.out)
    print('Графіки збережено в папці', args.out)


def run_from_files(args):
    """Перевірка по тестових даних викладача"""
    os.makedirs(args.out, exist_ok=True)
    v = load_series(args.v_file)
    y = load_series(args.y_file)
    print('Завантажено: len(v) = %d, len(y) = %d' % (len(v), len(y)))
    v, y = align_series(v, y)
    print('Після вирівнювання: len(v) = %d, len(y) = %d' % (len(v), len(y)))
    if args.coef_file:
        c = load_coeffs(args.coef_file)
        print('Коефіцієнти з файлу:', c)
        p, q = orders_from_coeffs(c)
        print('Порядок об\'єкта за файлом: АРКС(%d,%d)' % (p, q))
    else:
        c, p, q = {}, None, None
    orders = [(p, q)] if p else ALL_ORDERS
    res = estimate_models(v, y, orders, args.beta)
    print_table(res)
    theta = None
    if c and p:
        theta = np.array([c.get('a0', 0), c.get('a1', 0), c.get('a2', 0), c.get('a3', 0), 1.0,
                          c.get('b1', 0), c.get('b2', 0), c.get('b3', 0)])
        s, ok = check_stability(theta)
        print('Умова стійкості: |a1|+|a2|+|a3| = %.4f %s 1' % (s, '<' if ok else '>='))
    print_estimates(res, theta)
    if theta is not None:
        print('Примітка: коефіцієнт при v(k) у тестових даних може відрізнятися від 1 (залежить від того,')
        print('як викладач формував ряд y); порівнюйте насамперед a0..a3, b1..b3 та S.')
        plot_transient(v, y, theta, p, q, args.out, args.beta)
        print('Графіки збережено в папці', args.out)


def validate(args, ap):
    for name in ('p', 'q'):
        val = getattr(args, name)
        if val is not None and val not in (1, 2, 3):
            ap.error('--%s має бути 1, 2 або 3' % name)
    if (args.p is None) != (args.q is None) and not (args.v_file and args.y_file):
        ap.error('параметри --p та --q задаються разом')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='ЛР1: МНК та РМНК для АРКС(p,q)')
    ap.add_argument('--p', type=int, help='порядок авторегресії об\'єкта (1..3)')
    ap.add_argument('--q', type=int, help='порядок ковзного середнього об\'єкта (1..3)')
    ap.add_argument('--coeffs', type=float, nargs=7, metavar=('a0', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3'),
                    help='коефіцієнти об\'єкта a0 a1 a2 a3 b1 b2 b3')
    ap.add_argument('--v-file', help='текстовий файл із рядом v (одне число в рядку)')
    ap.add_argument('--y-file', help='текстовий файл із рядом y (одне число в рядку)')
    ap.add_argument('--coef-file', help='файл із коефіцієнтами вигляду a0=0.005')
    ap.add_argument('--seed', type=int, default=SEED, help='зерно генератора шуму')
    ap.add_argument('--beta', type=float, default=BETA, help='P(0) = beta * I')
    ap.add_argument('--out', default=OUT_DIR, help='папка для графіків')
    args = ap.parse_args()
    validate(args, ap)

    if args.v_file and args.y_file:
        run_from_files(args)
    elif args.p and args.q:
        run_custom(args)
    else:
        run_full_lab(args)
