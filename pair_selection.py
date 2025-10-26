"""
Configuración del proyecto de Pairs Trading (versión universo 50 → 25 pares → 1 mejor)
"""

# =========================
# Universo (yfinance)
# =========================
# 50 tickers US grandes y de varios sectores (puedes cambiarlos)
UNIVERSE_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","NFLX","DIS","CMCSA",
    "JPM","BAC","WFC","GS","MS","V","MA","C","PYPL","AXP",
    "XOM","CVX","COP","SLB","PSX",
    "PFE","JNJ","MRK","ABT","TMO",
    "KO","PEP","PG","WMT","COST","MCD","SBUX","NKE","TGT","HD",
    "IBM","ORCL","CRM","ADBE","INTC","AMD","QCOM","MU","BA","CAT"
]
UNIVERSE_START = "2010-01-01"   # ~15 años diarios
UNIVERSE_END   = None           # None = hoy (yfinance)
USE_ECONOMIC_RELATION = True    # True = prefiltrar por mismo sector/industria si se consigue (mejor paridad económica)

# =========================
# Splits cronológicos
# =========================
TRAIN_RATIO, TEST_RATIO, VAL_RATIO = 0.60, 0.20, 0.20

# =========================
# Selección de pares (rolling corr)
# =========================
ROLLING_CORR_WINDOW_UNIVERSE = 60   # días para correlación rolling
MIN_MEAN_ROLLING_CORR        = 0.50 # umbral sugerido
TOP_PAIRS_AFTER_CORR         = 100    # quedarnos con 25 pares
MIN_OVERLAP_DAYS             = 250   # exige datos suficientes por par (~3 años)

# =========================
# Cointegración (Engle–Granger)
# =========================
ADF_PVALUE_THRESHOLD   = 0.30  # residuales deben ser estacionarios (p < 0.05)
PRICE_NONSTAT_PVALUE   = 0.05  # precios log ADF p > 0.10 (no estacionarios)
HL_ACCEPT_RANGE_DAYS   = (2, 500)  # half-life razonable
CORRELATION_THRESHOLD  = 0.70      # redundante, deja aquí por compatibilidad

# =========================
# Costos & sizing
# =========================
INITIAL_CAPITAL  = 100_000
POSITION_SIZE    = 0.80            # 80% del capital
COMMISSION_RATE  = 0.00125         # 0.125% por pierna (entrada/salida)
BORROW_RATE      = 0.0025          # 0.25% anual
BORROW_DAY_BASIS = 365
MIN_TRADE_VALUE  = 100.0

# =========================
# Señales (por defecto, se pueden sobreescribir)
# =========================
Z_SCORE_ENTRY   = 0.60    # Reducido de 1.60
Z_SCORE_EXIT    = 0.10   # Reducido de 0.45
CONFIRM_MIN     = 0.40   # Más agresivo
CONFIRM_GAP     = 0.10
EXECUTION_LAG   = 1
MAX_Z_CAP       = 6.0
STOP_LOSS_Z     = 3.5
MIN_HOLD_DAYS   = 0

# =========================
# Aprendizaje de umbrales
# =========================
USE_TRAINED_THRESHOLDS = False
TRAIN_QUANT_ENTRY   = 0.80
TRAIN_QUANT_EXIT    = 0.60
TRAIN_QUANT_CONFIRM = 0.70

# =========================
# Re-hedge y gestión
# =========================
REHEDGE_ABS_DELTA_BETA     = 0.05
REHEDGE_MIN_DELTA_NOTIONAL = 250.0
TIME_STOP_MULTIPLIER       = 2.0
ENTRY_COOLDOWN_DAYS        = 0

# =========================
# Gates de régimen (rolling en backtest)
# =========================
ROLLING_ADF_WINDOW = 126
ROLLING_ADF_PVAL   = 0.30
ROLLING_CORR_WINDOW = 40
ROLLING_CORR_MIN    = 0.15
HL_MULT_RANGE       = (0.10, 5.0)
BETA_STAB_WINDOW    = 20
BETA_STAB_MAX_RANGE = 0.10

# =========================
# Risk sizing por spread
# =========================
SPREAD_VOL_WINDOW = 60
SPREAD_RISK_BPS   = 20           # 0.20% del capital

# =========================
# Kalman
# =========================
KALMAN_STATE        = "alpha_beta"
INITIAL_ALPHA       = 0.0
INITIAL_HEDGE_RATIO = 1.0
DELTA        = 5e-5
EWMA_ALPHA_R = 0.15
R_FLOOR      = 1e-8
Q_MIN, Q_MAX = 1e-8, 1e-3

# =========================
# Output
# =========================
PRINT_DIAGNOSTICS = True
SAVE_PLOTS        = True

# =========================
# Compatibilidad con módulos antiguos
# =========================
# (Para que kalman_filter.py y otros funcionen sin tocar su código)
try:
    KALMAN_STATE      = KALMAN_STATE
except NameError:
    KALMAN_STATE      = "alpha_beta"

# Los nombres "nuevos" del Kalman que esperan otros archivos
KALMAN_DELTA   = globals().get("DELTA", 5e-6)
KALMAN_R_EWMA  = globals().get("EWMA_ALPHA_R", 0.05)
KALMAN_R_FLOOR = globals().get("R_FLOOR", 1e-8)
KALMAN_Q_MIN   = globals().get("Q_MIN", 1e-8)
KALMAN_Q_MAX   = globals().get("Q_MAX", 1e-3)

# Otras claves que algunos archivos podrían esperar
CONFIRM_MIN = globals().get("CONFIRM_MIN", 1.10)
CONFIRM_GAP = globals().get("CONFIRM_GAP", 0.20)

# Cooldown / hold
ENTRY_COOLDOWN_DAYS = globals().get("ENTRY_COOLDOWN_DAYS", 2)
MIN_HOLD_DAYS       = globals().get("MIN_HOLD_DAYS", 1)


Z_ENTRY         = Z_SCORE_ENTRY
Z_EXIT          = Z_SCORE_EXIT
COOLDOWN_DAYS        = ENTRY_COOLDOWN_DAYS
KALMAN_DELTA   = DELTA
KALMAN_R_EWMA  = EWMA_ALPHA_R
KALMAN_R_FLOOR = R_FLOOR
KALMAN_Q_MIN   = Q_MIN
KALMAN_Q_MAX   = Q_MAX
HL_MIN = None 
HL_MAX = None
Z_STOP        = STOP_LOSS_Z
EXEC_LAG      = EXECUTION_LAG
TIME_STOP_MULT= TIME_STOP_MULTIPLIER
OUT_DIR       = "outputs"
