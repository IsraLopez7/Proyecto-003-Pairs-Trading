"""
Implementación del Filtro de Kalman como Proceso de Decisión Secuencial
siguiendo el marco de Powell para Análisis de Decisión Secuencial
"""

import numpy as np
from config import *

class KalmanFilter:
    """
    Filtro de Kalman formulado como proceso de decisión secuencial
    
    Los 5 elementos del modelo matemático:
    1. Estado (S_t): Ratio de cobertura dinámico β_t
    2. Acción (x_t): Decisión de trading basada en Z-score
    3. Información exógena (W_t): Nuevos precios observados
    4. Función de transición: S_{t+1} = f(S_t, x_t, W_{t+1})
    5. Función objetivo: Minimizar error de predicción
    
    Los 6 pasos del proceso de modelado:
    1. Problema: Estimar ratio de cobertura dinámico para pairs trading
    2. Modelo: Filtro de Kalman con ecuaciones de estado y observación
    3. Política: Trading basado en Z-score con matrices Q y R
    4. Implementación: Actualización secuencial del filtro
    5. Incertidumbre: Modelada a través de matrices de covarianza
    6. Evaluación: Backtesting con costos de transacción
    """
    
    def __init__(self, initial_hedge_ratio=INITIAL_HEDGE_RATIO, delta=DELTA):
        """
        Inicializa el filtro de Kalman
        
        Parámetros:
        - initial_hedge_ratio: Estimación inicial del ratio de cobertura β
        - delta: Factor de inicialización para la covarianza
        """
        # Estado inicial (ratio de cobertura)
        self.beta = initial_hedge_ratio
        self.beta_history = [self.beta]
        
        # Matrices del filtro
        self.R = None  # Covarianza del ruido de observación (se estima)
        self.Q = delta / (1 - delta) # Covarianza del ruido del proceso
        self.P = 1  # Covarianza del error de estimación inicial
        
        # Historial para análisis
        self.P_history = [self.P]
        self.e_history = []  # Errores de predicción
        self.Q_history = []  # Para seguimiento del z-score
        
    def initialize_R(self, residuals):
        """
        Inicializa R basándose en la varianza de los residuos históricos
        """
        self.R = np.var(residuals)
        print(f"R (varianza del ruido de observación) inicializada: {self.R:.6f}")
        
    def predict(self):
        """
        Paso de predicción (Time Update)
        Predice el siguiente estado y su covarianza
        """
        # En un random walk: β_t|t-1 = β_t-1|t-1
        # La predicción del ratio de cobertura es el valor anterior
        beta_pred = self.beta
        
        # Actualizar covarianza de predicción
        # P_t|t-1 = P_t-1|t-1 + Q
        P_pred = self.P + self.Q
        
        return beta_pred, P_pred
    
    def update(self, P1_t, P2_t):
        """
        Paso de actualización (Measurement Update)
        Actualiza el estado basándose en nueva observación
        
        Parámetros:
        - P1_t: Precio del activo 1 en tiempo t (KO)
        - P2_t: Precio del activo 2 en tiempo t (PEP)
        
        Retorna:
        - e_t: Error de predicción (innovación)
        - Q_t: Z-score para decisión de trading
        """
        # Paso de predicción
        beta_pred, P_pred = self.predict()
        
        # Error de predicción (innovación)
        # e_t = P1_t - β_t|t-1 * P2_t
        e_t = P1_t - beta_pred * P2_t
        self.e_history.append(e_t)
        
        # Ganancia de Kalman
        # K_t = P_t|t-1 * P2_t / (P2_t^2 * P_t|t-1 + R)
        K = (P_pred * P2_t) / (P2_t**2 * P_pred + self.R)
        
        # Actualización del estado (ratio de cobertura)
        # β_t|t = β_t|t-1 + K_t * e_t
        self.beta = beta_pred + K * e_t
        self.beta_history.append(self.beta)
        
        # Actualización de la covarianza
        # P_t|t = (1 - K_t * P2_t) * P_t|t-1
        self.P = (1 - K * P2_t) * P_pred
        self.P_history.append(self.P)
        
        # Calcular Z-score para señal de trading
        # Q_t = e_t / sqrt(R)
        if self.R > 0:
            Q_t = e_t / np.sqrt(self.R)
        else:
            Q_t = 0
        self.Q_history.append(Q_t)
        
        return e_t, Q_t
    
    def get_trading_signal(self, Q_t):
        """
        Genera señal de trading basada en Z-score
        
        Política de trading:
        - Si |Q_t| > Z_SCORE_ENTRY: Abrir posición
        - Si |Q_t| < Z_SCORE_EXIT: Cerrar posición
        
        Retorna:
        - 1: Long spread (comprar KO, vender PEP)
        - -1: Short spread (vender KO, comprar PEP)
        - 0: Neutral/cerrar posición
        """
        if abs(Q_t) > Z_SCORE_ENTRY:
            return -np.sign(Q_t)  # Mean reversion
        elif abs(Q_t) < Z_SCORE_EXIT:
            return 0
        else:
            return None  # Mantener posición actual
    
    def run_filter(self, P1_prices, P2_prices):
        """
        Ejecuta el filtro de Kalman sobre toda la serie de precios
        
        Parámetros:
        - P1_prices: Array de precios del activo 1 (KO)
        - P2_prices: Array de precios del activo 2 (PEP)
        
        Retorna:
        - spreads: Serie de spreads e_t
        - z_scores: Serie de Z-scores Q_t
        - hedge_ratios: Serie de ratios de cobertura β_t
        """
        spreads = []
        z_scores = []
        
        # Inicializar R con una estimación inicial
        initial_spread = P1_prices[:20] - self.beta * P2_prices[:20]
        self.initialize_R(initial_spread)
        
        # Procesar cada observación secuencialmente
        for t in range(len(P1_prices)):
            e_t, Q_t = self.update(P1_prices[t], P2_prices[t])
            spreads.append(e_t)
            z_scores.append(Q_t)
            
            # Actualizar R adaptativamente (ventana móvil)
            if t > 30:
                recent_errors = self.e_history[-30:]
                self.R = np.var(recent_errors)
        
        return np.array(spreads), np.array(z_scores), np.array(self.beta_history[1:])
    
    def reset(self):
        """
        Reinicia el filtro para nueva ejecución
        """
        self.beta = INITIAL_HEDGE_RATIO
        self.beta_history = [self.beta]
        self.P = 1
        self.P_history = [self.P]
        self.e_history = []
        self.Q_history = []
        self.R = None