"""
Configuración del proyecto de Pairs Trading
"""

# Configuración de datos
DATA_FILE = 'pares_KO_PEP_diario_2010_2025.csv'
TICKER_1 = 'KO'
TICKER_2 = 'PEP'

# División de datos (60% entrenamiento, 20% testing, 20% validación)
TRAIN_RATIO = 0.6
TEST_RATIO = 0.2
VAL_RATIO = 0.2

# Parámetros de trading
COMMISSION_RATE = 0.00125  # 0.125% por transacción
BORROW_RATE = 0.0025  # 0.25% anualizado
POSITION_SIZE = 0.8  # Usar 80% del capital disponible
INITIAL_CAPITAL = 100000  # Capital inicial

# Parámetros de la estrategia
CORRELATION_THRESHOLD = 0.7  # Umbral de correlación para selección de pares
ADF_PVALUE_THRESHOLD = 0.05  # p-value para test de cointegración
Z_SCORE_ENTRY = 2.0  # Entrar cuando z-score > 2
Z_SCORE_EXIT = 0.5  # Salir cuando z-score < 0.5

# Parámetros del filtro de Kalman
INITIAL_HEDGE_RATIO = 1.0
DELTA = 0.0001  # Para inicialización del filtro