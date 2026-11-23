# Аналіз часових рядів. Відтворення обчислень EViews (LS, AR/MA з бекастингом, fit/forecast, корелограма, описова статистика)
import numpy as np
from scipy import stats, optimize, signal


# ---------- Операції над рядами (аналоги виразів EViews) ----------
def lag(x, k):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if k == 0:
        return x.copy()
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def movav(x, n):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for k in range(n - 1, len(x)):
        out[k] = x[k - n + 1:k + 1].sum() / n
    return out


def ema_weights(n):
    alpha = 2 / (n + 1)
    return np.array([(1 - alpha) ** (n - i + 1) for i in range(1, n + 1)])


def ema(x, n):
    x = np.asarray(x, float)
    w = ema_weights(n)
    out = np.full(len(x), np.nan)
    for k in range(n - 1, len(x)):
        out[k] = (w * x[k - n + 1:k + 1]).sum() / w.sum()
    return out


def ma_coefs(q):
    if q == 1:
        return np.array([1.0])
    alpha = 2 / (q + 1)
    v = np.array([(1 - alpha) ** i for i in range(1, q + 1)])
    return v / v.sum()


def durbin_watson(e):
    e = np.asarray(e, float)
    e = e[~np.isnan(e)]
    return np.sum(np.diff(e) ** 2) / np.sum(e ** 2)


# ---------- Описова статистика (View -> Descriptive Statistics -> Histogram and Stats) ----------
def describe(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    m = x.mean()
    sd = x.std(ddof=1)
    sb = x.std(ddof=0)
    s = np.mean(((x - m) / sb) ** 3)
    k = np.mean(((x - m) / sb) ** 4)
    jb = n / 6 * (s ** 2 + (k - 3) ** 2 / 4)
    return {"Mean": m, "Median": np.median(x), "Maximum": x.max(), "Minimum": x.min(),
            "Std. Dev.": sd, "Skewness": s, "Kurtosis": k, "Jarque-Bera": jb,
            "Probability": stats.chi2.sf(jb, 2), "Observations": n}


# ---------- Корелограма: АКФ, ЧАКФ (Дарбін-Левінсон), Q-статистика Л'юнга-Бокса ----------
def acf(x, nlags):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    d = x - x.mean()
    den = np.sum(d ** 2)
    return np.array([np.sum(d[s:] * d[:-s]) / den for s in range(1, nlags + 1)])


def pacf_from_acf(r):
    L = len(r)
    phi = np.zeros((L + 1, L + 1))
    phi[1, 1] = r[0]
    for k in range(2, L + 1):
        num = r[k - 1] - sum(phi[k - 1, j] * r[k - j - 1] for j in range(1, k))
        den = 1 - sum(phi[k - 1, j] * r[j - 1] for j in range(1, k))
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
    return np.array([phi[k, k] for k in range(1, L + 1)])


def correlogram(x, nlags=12):
    x = np.asarray(x, float)
    n = int(np.sum(~np.isnan(x)))
    r = acf(x, nlags)
    pac = pacf_from_acf(r)
    j = np.arange(1, nlags + 1)
    q = n * (n + 2) * np.cumsum(r ** 2 / (n - j))
    return {"n": n, "AC": r, "PAC": pac, "Q": q, "Prob": stats.chi2.sf(q, j),
            "band": 2 / np.sqrt(n)}


# ---------- Оцінювання рівнянь (команда ls) ----------
class Equation:
    def __init__(self, name, dep, names, coef, se, y, e, resid, fitted, first, last, k,
                 const, method="Least Squares", iters=None, backcast=None):
        self.name, self.dep, self.names = name, dep, names
        self.coef, self.se = np.asarray(coef), np.asarray(se)
        self.first, self.last = first, last
        self.resid, self.fitted = resid, fitted
        self.method, self.iters, self.backcast = method, iters, backcast
        T = len(y)
        self.T, self.k = T, k
        self.e = e
        ssr = float(np.sum(e ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        self.tstat = self.coef / self.se
        self.prob = 2 * stats.t.sf(np.abs(self.tstat), T - k)
        r2 = 1 - ssr / tss
        ll = -T / 2 * (1 + np.log(2 * np.pi) + np.log(ssr / T))
        self.st = {
            "R-squared": r2,
            "Adjusted R-squared": 1 - (1 - r2) * (T - 1) / (T - k),
            "S.E. of regression": np.sqrt(ssr / (T - k)),
            "Sum squared resid": ssr,
            "Log likelihood": ll,
            "Durbin-Watson stat": durbin_watson(e),
            "Mean dependent var": y.mean(),
            "S.D. dependent var": y.std(ddof=1),
            "Akaike info criterion": -2 * ll / T + 2 * k / T,
            "Schwarz criterion": -2 * ll / T + k * np.log(T) / T,
            "Hannan-Quinn criter.": -2 * ll / T + 2 * k * np.log(np.log(T)) / T,
        }
        if const and k > 1:
            F = (r2 / (k - 1)) / ((1 - r2) / (T - k))
            self.st["F-statistic"] = F
            self.st["Prob(F-statistic)"] = stats.f.sf(F, k - 1, T - k)

    def __getitem__(self, key):
        return self.st[key]


def _valid(cols):
    m = np.ones(len(cols[0]), bool)
    for c in cols:
        m &= ~np.isnan(c)
    return m


def ls(name, y, regs, dep="Y", ar=0, ma=0, starts=()):
    """regs: список пар (ім'я, ряд); 'C' - константа. ar/ma - порядок або список лагів складових ar(i), ma(j)."""
    y = np.asarray(y, float)
    n = len(y)
    names = [r[0] for r in regs]
    X = np.column_stack([np.ones(n) if r[0] == "C" else np.asarray(r[1], float) for r in regs]) \
        if regs else np.zeros((n, 0))
    const = "C" in names
    ok = _valid([y] + [X[:, i] for i in range(X.shape[1])])
    arl = list(range(1, ar + 1)) if isinstance(ar, int) else sorted(ar)
    mal = list(range(1, ma + 1)) if isinstance(ma, int) else sorted(ma)
    if not arl and not mal:
        idx = np.where(ok)[0]
        Y, Z = y[idx], X[idx]
        b, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        e = Y - Z @ b
        s2 = e @ e / (len(Y) - len(b))
        se = np.sqrt(np.diag(s2 * np.linalg.inv(Z.T @ Z)))
        resid = np.full(n, np.nan); resid[idx] = e
        fit = np.full(n, np.nan); fit[idx] = Z @ b
        eq = Equation(name, dep, names, b, se, Y, e, resid, fit, idx[0] + 1, idx[-1] + 1, len(b), const)
        eq.X, eq.beta, eq.ar_lags, eq.ma_lags = X, b, [], []
        return eq
    return _arma(name, y, X, names, ok, arl, mal, dep, const, starts)


def _lm(f, z, tol, maxit):
    e = f(z)
    ssr = e @ e
    lam, it = 1e-3, 0
    while it < maxit:
        it += 1
        J = _jac(f, z, e)
        g, H = J.T @ e, J.T @ J
        while True:
            step = -np.linalg.lstsq(H + lam * np.diag(np.diag(H) + 1e-12), g, rcond=None)[0]
            cand = z + step
            ec = f(cand)
            sc = ec @ ec if np.all(np.isfinite(ec)) else np.inf
            if sc <= ssr:
                lam = max(lam / 10, 1e-12)
                break
            lam *= 10
            if lam > 1e12:
                return z, e, ssr, it
        rel = np.max(np.abs(step) / np.maximum(np.abs(z), 1e-10))
        z, e, ssr = cand, ec, sc
        if rel < tol:
            break
    return z, e, ssr, it


def _jac(f, z, e):
    J = np.empty((len(e), len(z)))
    for i in range(len(z)):
        h = 1e-7 * max(1.0, abs(z[i]))
        d = z.copy(); d[i] += h
        J[:, i] = (f(d) - e) / h
    return J


def _full(vals, lags, L):
    v = np.zeros(L)
    for c, l in zip(vals, lags):
        v[l - 1] = c
    return v


def _arma(name, y, X, names, ok, arl, mal, dep, const, starts=(), tol=1e-9, maxit=2000):
    n = len(y)
    p, q = (max(arl) if arl else 0), (max(mal) if mal else 0)
    na, nm = len(arl), len(mal)
    okp = ok.copy()
    for i in range(1, p + 1):
        okp &= np.concatenate([np.zeros(i, bool), ok[:-i]])
    idx = np.where(okp)[0]
    t0, t1 = idx[0], idx[-1]
    if not np.all(okp[t0:t1 + 1]):
        raise ValueError("ARMA потребує неперервної вибірки")
    m = X.shape[1]
    ci = names.index("C") if const else None
    rng = np.arange(t0 - p, t1 + 1)
    Yw, Xw = y[rng], X[rng]
    N = t1 - t0 + 1

    def filt(v, rho):
        out = v[p:].copy()
        for i in range(1, p + 1):
            if rho[i - 1]:
                out -= rho[i - 1] * v[p - i:len(v) - i]
        return out

    def innov(z):
        b, rho, th = z[:m], _full(z[m:m + na], arl, p), _full(z[m + na:], mal, q)
        uh = filt(Yw, rho)
        for j in range(m):
            uh -= b[j] if j == ci else b[j] * filt(Xw[:, j], rho)
        if q == 0:
            return uh
        den = np.concatenate([[1.0], th])
        eta = signal.lfilter([1.0], den, uh[::-1])[::-1]
        pre = np.array([sum(th[j - 1] * eta[t + j] for j in range(1, q + 1) if 0 <= t + j < N)
                        for t in range(-q, 0)])
        return signal.lfilter([1.0], den, np.concatenate([pre, uh]))[q:]

    b0, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None) if m else (np.zeros(0),)
    u0 = y - (X @ b0 if m else 0)
    if na:
        Zr = np.column_stack([lag(u0, i) for i in arl])
        okr = _valid([u0] + [Zr[:, i] for i in range(na)])
        r0, *_ = np.linalg.lstsq(Zr[okr], u0[okr], rcond=None)
    else:
        r0 = np.zeros(0)
    if ci is not None:
        b0 = b0.copy(); b0[ci] *= (1 - r0.sum())
    z0 = np.concatenate([b0, r0, np.zeros(nm)])
    cands = [z0]
    if nm:
        e0 = innov(z0)
        Ze = np.column_stack([lag(e0, j) for j in mal])
        oke = _valid([Ze[:, j] for j in range(nm)])
        th0, *_ = np.linalg.lstsq(Ze[oke], e0[oke], rcond=None)
        cands.append(np.concatenate([b0, r0, np.clip(th0, -0.9, 0.9)]))
    for st in starts:
        cands.append(_to_z(np.asarray(st, float), m, na, ci))
    best = None
    for z in cands:
        res = _lm(innov, z, tol, maxit)
        if best is None or res[2] < best[2] - 1e-10 * max(1, best[2]):
            best = res
    z, e, ssr, it = best
    J = _jac(innov, z, e)
    k = len(z)
    covz = (e @ e / (N - k)) * np.linalg.pinv(J.T @ J)
    par, G = z.copy(), np.eye(k)
    srho = z[m:m + na].sum()
    if ci is not None:
        par[ci] = z[ci] / (1 - srho)
        G[ci, ci] = 1 / (1 - srho)
        G[ci, m:m + na] = z[ci] / (1 - srho) ** 2
    se = np.sqrt(np.diag(G @ covz @ G.T))
    Y = y[t0:t1 + 1]
    resid = np.full(n, np.nan); resid[t0:t1 + 1] = e
    fit = np.full(n, np.nan); fit[t0:t1 + 1] = Y - e
    allnames = names + ["AR(%d)" % i for i in arl] + ["MA(%d)" % j for j in mal]
    eq = Equation(name, dep, allnames, par, se, Y, e, resid, fit, t0 + 1, t1 + 1, k, const,
                  iters=it, backcast=(t0 + 1 - q) if q else None)
    eq.intercept = z[ci] if ci is not None else 0.0
    eq.p, eq.q, eq.ar_lags, eq.ma_lags = p, q, arl, mal
    eq.X, eq.beta = X, par[:m]
    eq.rho, eq.theta = _full(z[m:m + na], arl, p), _full(z[m + na:], mal, q)
    eq.ar_roots = np.roots(np.concatenate([[1.0], -eq.rho])) if p else np.array([])
    eq.ma_roots = np.roots(np.concatenate([[1.0], eq.theta])) if q else np.array([])
    return eq


def _to_z(par, m, na, ci):
    z = par.copy()
    if ci is not None:
        z[ci] = par[ci] * (1 - par[m:m + na].sum())
    return z


# ---------- Прогнозування (fit - статичний, forecast - динамічний) ----------
def forecast_arma(eq, y, X, start, end, dynamic):
    """Прогноз рівняння зі складовими ar/ma на спостереження start..end (індекси від 0).
    Статичний (fit): інновації перераховуються бекастингом на розширеній вибірці з фактичними даними.
    Динамічний (forecast): інновації прогнозного періоду дорівнюють нулю."""
    y = np.asarray(y, float)
    p, q = eq.p, eq.q
    u = y - X @ eq.beta
    out = np.full(len(y), np.nan)
    if not dynamic:
        t0 = eq.first - 1
        uh = np.array([u[t] - sum(eq.rho[i - 1] * u[t - i] for i in range(1, p + 1)) for t in range(t0, end + 1)])
        e = uh
        if q:
            N = len(uh)
            den = np.concatenate([[1.0], eq.theta])
            eta = signal.lfilter([1.0], den, uh[::-1])[::-1]
            pre = np.array([sum(eq.theta[j - 1] * eta[t + j] for j in range(1, q + 1) if 0 <= t + j < N)
                            for t in range(-q, 0)])
            e = signal.lfilter([1.0], den, np.concatenate([pre, uh]))[q:]
        out[start:end + 1] = y[start:end + 1] - e[start - t0:]
        return out
    eps = np.concatenate([eq.resid, np.zeros(len(y) - len(eq.resid))])
    uf = u.copy()
    for t in range(start, end + 1):
        uh = sum(eq.rho[i - 1] * uf[t - i] for i in range(1, p + 1))
        uh += sum(eq.theta[j - 1] * (0.0 if t - j >= start else eps[t - j]) for j in range(1, q + 1))
        out[t] = X[t] @ eq.beta + uh
        uf[t] = uh
    return out


def forecast_lags(eq, y, start, end, dynamic, lags, exog=None):
    """Прогноз рівняння з явними лагами залежної змінної: регресори C, y(-l) для l у lags, далі exog."""
    y = np.asarray(y, float)
    yf = y.copy()
    out = np.full(len(y), np.nan)
    for t in range(start, end + 1):
        src = yf if dynamic else y
        row = [1.0] + [src[t - l] for l in lags] + ([] if exog is None else list(exog[t]))
        out[t] = np.dot(eq.beta, row)
        yf[t] = out[t]
    return out


def forecast_eval(actual, fc):
    a, f = np.asarray(actual, float), np.asarray(fc, float)
    h = len(a)
    mse = np.mean((f - a) ** 2)
    sa, sf = a.std(ddof=0), f.std(ddof=0)
    r = np.corrcoef(a, f)[0, 1] if h > 1 and sa > 0 and sf > 0 else 0.0
    return {"RMSE": np.sqrt(mse), "MAE": np.mean(np.abs(f - a)), "MAPE": 100 * np.mean(np.abs((f - a) / a)),
            "Theil": np.sqrt(mse) / (np.sqrt(np.mean(f ** 2)) + np.sqrt(np.mean(a ** 2))),
            "Bias": (f.mean() - a.mean()) ** 2 / mse, "Variance": (sf - sa) ** 2 / mse,
            "Covariance": 2 * (1 - r) * sf * sa / mse}


# ---------- Вивід у вигляді вікна EViews ----------
STAT_ROWS = [("R-squared", "Mean dependent var"), ("Adjusted R-squared", "S.D. dependent var"),
             ("S.E. of regression", "Akaike info criterion"), ("Sum squared resid", "Schwarz criterion"),
             ("Log likelihood", "F-statistic"), ("Durbin-Watson stat", "Prob(F-statistic)")]


def eviews_text(eq, lab=str):
    L = ["Dependent Variable: %s" % eq.dep, "Method: %s" % eq.method,
         "Sample(adjusted): %s %s" % (lab(eq.first), lab(eq.last)),
         "Included observations: %d after adjusting endpoints" % eq.T]
    if eq.backcast is not None:
        L.append("Backcast: %s %s" % (lab(eq.backcast), lab(eq.first - 1)))
    L.append("=" * 76)
    L.append("%-18s%15s%15s%15s%11s" % ("Variable", "Coefficient", "Std. Error", "t-Statistic", "Prob."))
    L.append("=" * 76)
    for nm, c, s, t, p in zip(eq.names, eq.coef, eq.se, eq.tstat, eq.prob):
        L.append("%-18s%15s%15s%15.6f%11.4f" % (nm, _num(c), _num(s), t, p))
    L.append("=" * 76)
    rows = STAT_ROWS if "F-statistic" in eq.st else STAT_ROWS[:4] + [("Log likelihood", "Durbin-Watson stat")]
    for a, b in rows:
        L.append("%-22s%14.6f    %-22s%14.6f" % (a, eq.st[a], b, eq.st[b]))
    L.append("=" * 76)
    for lab, roots in (("Inverted AR Roots", getattr(eq, "ar_roots", [])), ("Inverted MA Roots", getattr(eq, "ma_roots", []))):
        if len(roots):
            rs = [_croot(r) for r in roots]
            for i in range(0, len(rs), 4):
                L.append("%-18s %s" % (lab if i == 0 else "", "  ".join("%13s" % x for x in rs[i:i + 4])))
    return "\n".join(L)


def _num(x):
    return "%.6f" % x if x == 0 or 1e-4 <= abs(x) < 1e7 else "%.6E" % x


def _croot(r):
    r = complex(r)
    if abs(r.imag) < 1e-10:
        return "%.2f" % r.real
    return "%.2f%+.2fi" % (r.real, r.imag)
