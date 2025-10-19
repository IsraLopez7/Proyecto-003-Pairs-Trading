"""
Módulo para análisis de cointegración
"""

import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from config import *

class CointegrationAnalysis:
    def __init__(self):
        """
        Inicializa el análisis de cointegración
        """
        self.beta_0 = None  # Intercepto
        self.beta_1 = None  # Coeficiente de cobertura
        self.residuals = None
        self.adf_result = None
        
    def calculate_correlation(self, prices1, prices2):
        """
        Calcula la correlación entre dos series de precios
        """
        correlation = np.corrcoef(prices1, prices2)[0, 1]
        print(f"Correlación entre activos: {correlation:.4f}")
        
        if correlation < CORRELATION_THRESHOLD:
            print(f"Advertencia: Correlación ({correlation:.4f}) menor al umbral ({CORRELATION_THRESHOLD})")
        
        return correlation
    
    def run_ols_regression(self, P1, P2):
        """
        Ejecuta regresión OLS: P1 = β0 + β1*P2 + ε
        donde P1 es KO y P2 es PEP
        """
        # Añadir columna de unos para el intercepto
        X = np.column_stack([np.ones(len(P2)), P2])
        
        # Calcular coeficientes usando OLS: β = (X'X)^(-1)X'y
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ P1
        
        self.beta_0 = beta[0]
        self.beta_1 = beta[1]
        
        # Calcular residuos
        self.residuals = P1 - (self.beta_0 + self.beta_1 * P2)
        
        # Calcular R-cuadrado
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((P1 - np.mean(P1)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"\nRegresión OLS:")
        print(f"P_{TICKER_1} = {self.beta_0:.4f} + {self.beta_1:.4f} * P_{TICKER_2} + ε")
        print(f"R-cuadrado: {r_squared:.4f}")
        
        return self.beta_0, self.beta_1, self.residuals
    
    def test_stationarity(self, residuals=None):
        """
        Aplica el test ADF (Augmented Dickey-Fuller) a los residuos
        """
        if residuals is None:
            residuals = self.residuals
            
        if residuals is None:
            raise ValueError("Primero debe ejecutar la regresión OLS")
        
        # Aplicar test ADF
        self.adf_result = adfuller(residuals, autolag='AIC')
        
        adf_stat = self.adf_result[0]
        p_value = self.adf_result[1]
        critical_values = self.adf_result[4]
        
        print(f"\nTest de Augmented Dickey-Fuller:")
        print(f"Estadístico ADF: {adf_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Valores críticos:")
        for key, value in critical_values.items():
            print(f"  {key}: {value:.4f}")
        
        is_cointegrated = p_value < ADF_PVALUE_THRESHOLD
        
        if is_cointegrated:
            print(f"Resultado: Los pares están COINTEGRADOS (p-value < {ADF_PVALUE_THRESHOLD})")
        else:
            print(f"Resultado: Los pares NO están cointegrados (p-value >= {ADF_PVALUE_THRESHOLD})")
        
        return is_cointegrated, p_value
    
    def calculate_half_life(self, residuals=None):
        """
        Calcula el half-life de mean reversion
        """
        if residuals is None:
            residuals = self.residuals
            
        # Crear serie de diferencias
        residuals_lag = np.roll(residuals, 1)
        residuals_diff = residuals - residuals_lag
        
        # Eliminar el primer valor (NaN por el lag)
        residuals_lag = residuals_lag[1:]
        residuals_diff = residuals_diff[1:]
        
        # Regresión para calcular theta
        theta = np.polyfit(residuals_lag, residuals_diff, 1)[0]
        
        # Half-life = -ln(2) / theta
        if theta < 0:
            half_life = -np.log(2) / theta
            print(f"Half-life de mean reversion: {half_life:.1f} días")
        else:
            half_life = np.inf
            print("Half-life: Infinito (no hay mean reversion)")
        
        return half_life