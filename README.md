# Proyecto-003-Pairs-Trading
Proyecto #3 de trading

Implementación de una estrategia de arbitraje estadístico (pairs trading) utilizando las acciones de Coca-Cola (KO) y PepsiCo (PEP). La estrategia emplea filtros de Kalman para estimar dinámicamente el ratio de cobertura y generar señales de trading basadas en reversión a la media.

Un filtro de Kalman es un algoritmo recursivo que proporciona estimaciones óptimas del estado de un sistema dinámico. El algoritmo está diseñado para estimar los estados no observados (p. ej., posición, velocidad) de un sistema a lo largo del tiempo combinando predicciones previas con observaciones entrantes con ruido.

# Recomendado: Python 3.12 (vale 3.10–3.12)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python main.py

Objetivo: de un universo de 50 tickers descargados con yfinance (≈15 años diarios), elegir 25 pares candidatos por correlación histórica, confirmar cointegración con Engle–Granger, escoger el mejor par, estimar α,β con Kalman y backtestear una estrategia de mean-reversion con costos reales.

Descarga 50 → Limpia datos
      │
      ├─ Rolling Corr (60d) + filtro económico opcional
      │     └→ TOP 25 pares
      │
      ├─ Engle–Granger por par:
      │     OLS: P_A = β0 + β1 P_B + ε
      │     ADF(ε) < 0.05 y half-life razonable
      │     └→ ranking (score)
      │
      ├─ Elegir mejor par (o TOP-K)
      │
      ├─ Split temporal: 60% TRAIN / 20% TEST / 20% VALID
      │
      ├─ TRAIN:
      │     OLS + ADF(ε) + half-life
      │     Kalman(α,β) → Z-score
      │     Umbrales base (o aprendidos)
      │
      ├─ TEST:
      │     Grid de (entry/exit/confirm/stop)
      │     Backtest con costos → mejor cfg
      │
      └─ VALID:
            Backtest final + métricas + gráficas


### Estructura del Proyecto
backtesting.py        # Motor de backtest, señales, costos, métricas y plots
cointegration.py      # Engle–Granger (OLS, ADF residual) + half-life OU
config.py             # Parámetros globales (universo, umbrales, costos, Kalman…)
kalman_filter.py      # Filtro de Kalman α–β (estado [alpha, beta])
main.py               # Orquestación del pipeline completo
pair_selection.py     # Descarga, rolling corr, filtros y ranking de pares
utils.py              # Splits, ADF simple, utilidades numéricas
requirements.txt      # Dependencias
README.md             # Este documento


El programa realizará automáticamente:

Carga y división de datos (60% entrenamiento, 20% prueba, 20% validación)
Análisis de cointegración con test ADF
Backtest con filtro de Kalman dinámico
Generación de estadísticas y gráficos
Validación de resultados

Componentes Clave
Filtro de Kalman como Proceso de Decisión Secuencial

Estado: Ratio de cobertura dinámico (β)
Acción: Decisiones de trading basadas en Z-score
Política: Umbrales de entrada/salida con matrices Q y R adaptativas

Parámetros de Trading

Capital inicial: $100,000
Tamaño de posición: 80% del capital disponible
Comisión: 0.125% por transacción
Tasa de préstamo: 0.25% anualizada
Umbral Z-score entrada: ±2.0
Umbral Z-score salida: ±0.5

Análisis de Cointegración

Regresión OLS: P₁ = β₀ + β₁P₂ + ε
Test ADF sobre residuos
Cálculo de half-life de reversión

Resultados
El programa genera:

Evolución del spread
Ratio de cobertura dinámico
Señales de trading (Z-score)
Valor del portfolio
Distribución de retornos
Estadísticas de performance (Sharpe ratio, win rate, etc.)

Archivos de Salida

pairs_trading_results.png: Gráficos principales
returns_distribution.png: Distribución de retornos
Trading_signals.png: Señales de los trades