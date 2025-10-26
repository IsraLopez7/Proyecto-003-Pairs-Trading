# kalman_filter.py
# ------------------------------------------------------------
# Filtro de Kalman para alpha-beta dinámicos:
#   y_t = alpha_t + beta_t * x_t + e_t
#   [alpha_t, beta_t]' = [alpha_{t-1}, beta_{t-1}]' + w_t
# donde w_t ~ N(0, Q), e_t ~ N(0, R_t)
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np

from config import (
    # Estado y seeds
    KALMAN_STATE,
    INITIAL_ALPHA,
    INITIAL_HEDGE_RATIO,

    # Q y R (alias en config.py)
    KALMAN_DELTA,      # ~ intensidad para Q
    KALMAN_R_EWMA,     # factor EWMA para R_t (0..1)
    KALMAN_R_FLOOR,    # piso numérico de R
    KALMAN_Q_MIN,      # límites de Q
    KALMAN_Q_MAX,

    # Otros
    MAX_Z_CAP,
)

class KalmanFilter:
    """
    Filtro de Kalman para estimar alpha_t y beta_t dinámicos entre dos series de precios.
    Devuelve:
      spreads   : innovación (y_t - y_pred_t) o residuo tras la actualización
      z_scores  : innovación normalizada por sqrt(var_ewma)
      betas     : trayectoria de beta_t (hedge ratio)
      alphas    : trayectoria de alpha_t
    """

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def reset(self):
        self.alpha = float(INITIAL_ALPHA)
        self.beta  = float(INITIAL_HEDGE_RATIO)
        # Matriz de covarianza inicial (un poco grande pero acotada)
        self.P = np.eye(2) * 1e3
        # Varianza observacional EWMA (se inicializa en run_filter)
        self.R = None

    def run_filter(self, y_prices, x_prices, residuals_train=None):
        """
        y_prices: serie de precios del activo 1 (y)
        x_prices: serie de precios del activo 2 (x)
        residuals_train: residuales de OLS (TRAIN) para seedear R

        returns: spreads, z_scores, betas, alphas (np.ndarray)
        """
        y = np.asarray(y_prices, dtype=float)
        x = np.asarray(x_prices, dtype=float)
        n = len(y)
        if len(x) != n:
            raise ValueError("y_prices y x_prices deben tener la misma longitud.")

        # Limpieza mínima
        mask = ~(np.isfinite(y) & np.isfinite(x))
        if mask.any():
            # forward-fill simple
            for arr in (y, x):
                isn = ~np.isfinite(arr)
                if isn.any():
                    arr[isn] = np.interp(np.flatnonzero(isn), np.flatnonzero(~isn), arr[~isn])

        # Semillas
        if residuals_train is not None:
            rseed = float(np.nanvar(np.asarray(residuals_train, dtype=float)))
            if not np.isfinite(rseed) or rseed <= 0:
                rseed = 1.0
        else:
            # var de la diferencia simple como aproximación
            diffy = np.diff(y)
            rseed = float(np.nanvar(diffy)) if diffy.size > 3 else 1.0
            if not np.isfinite(rseed) or rseed <= 0:
                rseed = 1.0

        self.R = max(KALMAN_R_FLOOR, rseed)

        # Q a partir de delta (forma estándar para random-walk)
        # Q ≈ delta/(1-delta) * I, acotado en [Q_MIN, Q_MAX]
        q_scalar = KALMAN_DELTA / max(1e-12, (1.0 - KALMAN_DELTA))
        q_scalar = float(np.clip(q_scalar, KALMAN_Q_MIN, KALMAN_Q_MAX))
        Q = np.eye(2) * q_scalar

        # Inicialización del estado
        theta = np.array([self.alpha, self.beta], dtype=float)  # [alpha, beta]
        P = self.P.copy()

        alphas = np.zeros(n, dtype=float)
        betas  = np.zeros(n, dtype=float)
        spreads = np.zeros(n, dtype=float)
        z_scores = np.zeros(n, dtype=float)

        # Varianza EWMA de la innovación para normalizar Z
        var_ewma = self.R

        for t in range(n):
            xt = x[t]
            yt = y[t]

            # --- PREDICCIÓN DEL ESTADO (random walk) ---
            theta_pred = theta               # F = I
            P_pred = P + Q

            # --- PREDICCIÓN DE LA OBSERVACIÓN ---
            # H_t = [1, x_t]
            H = np.array([1.0, float(xt)], dtype=float)

            y_pred = H @ theta_pred
            innov = float(yt - y_pred)       # innovación (residuo de predicción)

            # --- ACTUALIZACIÓN ---
            S = float(H @ P_pred @ H.T + self.R)   # var de la innovación
            if S <= 0 or not np.isfinite(S):
                # fallback numérico
                S = float(max(1e-8, abs(self.R)))

            K = (P_pred @ H.T) / S           # Ganancia de Kalman (2x1)
            theta = theta_pred + K * innov   # estado posterior
            P = P_pred - np.outer(K, H) @ P_pred

            # Guardar trayectorias
            alpha_t, beta_t = theta[0], theta[1]
            alphas[t] = alpha_t
            betas[t]  = beta_t

            # Residuo tras actualización (equivalente a spread con alpha/beta actualizados)
            spread_t = float(yt - (alpha_t + beta_t * xt))
            spreads[t] = spread_t

            # Actualizar R con EWMA sobre la innovación al cuadrado
            self.R = max(KALMAN_R_FLOOR, (1.0 - KALMAN_R_EWMA) * self.R + KALMAN_R_EWMA * (innov * innov))
            # EWMA para normalización de Z (puede ser la misma regla)
            var_ewma = max(KALMAN_R_FLOOR, (1.0 - KALMAN_R_EWMA) * var_ewma + KALMAN_R_EWMA * (innov * innov))
            std_ewma = float(np.sqrt(var_ewma))

            z = spread_t / std_ewma if std_ewma > 0 else 0.0
            # Cap en Z para evitar explosiones numéricas
            z_scores[t] = float(np.clip(z, -MAX_Z_CAP, MAX_Z_CAP))

        # Persistir último estado
        self.alpha = float(theta[0])
        self.beta  = float(theta[1])
        self.P     = P

        return spreads, z_scores, betas, alphas
