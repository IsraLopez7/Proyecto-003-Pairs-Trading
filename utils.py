import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def adf_pvalue(series: pd.Series) -> float:
    s = pd.Series(series).dropna().astype(float)
    if len(s) < 20:
        return 1.0
    try:
        return adfuller(s, autolag="AIC")[1]
    except Exception:
        return 1.0

def eg_cointegration(a: pd.Series, b: pd.Series):
    """Engle–Granger: OLS a~const+b → ADF de residuales."""
    x = sm.add_constant(b.values, has_constant='add')
    model = sm.OLS(a.values, x).fit()
    alpha = float(model.params[0])
    beta  = float(model.params[1])
    resid = a - (alpha + beta*b)
    pval_resid = adf_pvalue(resid)
    return {
        "alpha": alpha, "beta": beta, "resid": resid,
        "p_resid": pval_resid, "r2": float(model.rsquared)
    }

def half_life_ou(spread: pd.Series) -> float:
    s = pd.Series(spread).dropna().astype(float)
    if len(s) < 30:
        return np.inf
    y = s.diff().dropna()
    x = s.shift(1).dropna().loc[y.index]
    try:
        beta = np.polyfit(x, y, 1)[0]
    except Exception:
        return np.inf
    if not np.isfinite(beta) or beta >= 0:
        return np.inf
    hl = -np.log(2.0) / beta
    return float(hl) if np.isfinite(hl) and hl > 0 else np.inf

def rolling_split(df: pd.DataFrame, ratios=(0.6,0.2,0.2)):
    n = len(df)
    n1 = int(n*ratios[0])
    n2 = int(n*ratios[1])
    train = df.iloc[:n1].copy()
    test  = df.iloc[n1:n1+n2].copy()
    val   = df.iloc[n1+n2:].copy()
    return train, test, val
