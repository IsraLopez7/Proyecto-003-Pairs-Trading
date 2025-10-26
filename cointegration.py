# cointegration.py
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class CointegrationAnalysis:
    """
    Utilidades de Engle–Granger usadas por main/backtesting:
      - calculate_correlation
      - run_ols_regression  -> (beta0, beta1, residuals)
      - test_stationarity   -> (is_stationary, p_value) sobre residuales
      - calculate_half_life -> OU half-life en días
    """

    @staticmethod
    def calculate_correlation(series_a, series_b) -> float:
        a = np.asarray(series_a, dtype=float)
        b = np.asarray(series_b, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return np.nan
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    @staticmethod
    def run_ols_regression(series_y, series_x):
        """
        OLS: y = beta0 + beta1 * x + e
        Devuelve (beta0, beta1, residuals)
        """
        y = np.asarray(series_y, dtype=float)
        x = np.asarray(series_x, dtype=float)
        mask = np.isfinite(y) & np.isfinite(x)
        y = y[mask]; x = x[mask]
        X = sm.add_constant(x)
        model = sm.OLS(y, X)
        res = model.fit()
        beta0 = float(res.params[0])
        beta1 = float(res.params[1])
        residuals = res.resid.astype(float)
        return beta0, beta1, residuals

    @staticmethod
    def test_stationarity(residuals, autolag: str = "AIC"):
        """
        ADF a residuales. Devuelve (is_stationary, p_value).
        """
        r = np.asarray(residuals, dtype=float)
        r = r[np.isfinite(r)]
        if r.size < 50:
            return False, 1.0
        try:
            adf_stat, p_value, _, _, _, _ = adfuller(r, autolag=autolag)
        except Exception:
            return False, 1.0
        return (p_value < 0.05), float(p_value)

    @staticmethod
    def calculate_half_life(residuals) -> float:
        """
        Half-life de reversión del proceso OU discreto estimado sobre residuales.
        HL = -ln(2) / theta, donde y_t - y_{t-1} = theta * y_{t-1} + ruido
        """
        x = np.asarray(residuals, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 60:
            return np.inf
        x = x - np.nanmean(x)
        if np.nanstd(x) < 1e-8:
            return np.inf
        y = np.diff(x)
        x_lag = x[:-1]
        try:
            theta = np.polyfit(x_lag, y, 1)[0]
        except Exception:
            return np.inf
        if not np.isfinite(theta) or theta >= 0:
            return np.inf
        return float(-np.log(2.0) / theta)
