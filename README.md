# Proyecto-003-Pairs-Trading
Proyecto #3 de trading

Implementación de una estrategia de arbitraje estadístico (pairs trading) utilizando las acciones de Coca-Cola (KO) y PepsiCo (PEP). La estrategia emplea filtros de Kalman para estimar dinámicamente el ratio de cobertura y generar señales de trading basadas en reversión a la media.

Un filtro de Kalman es un algoritmo recursivo que proporciona estimaciones óptimas del estado de un sistema dinámico. El algoritmo está diseñado para estimar los estados no observados (p. ej., posición, velocidad) de un sistema a lo largo del tiempo combinando predicciones previas con observaciones entrantes con ruido.

### Estructura del Proyecto
pairs_trading/
│
├── config.py              # Configuración y parámetros
├── data_handler.py        # Manejo y preprocesamiento de datos
├── cointegration.py       # Análisis de cointegración
├── kalman_filter.py       # Implementación del filtro de Kalman
├── trading_strategy.py    # Lógica de trading
├── backtesting.py         # Motor de backtesting
├── main.py               # Archivo principal
├── requirements.txt       # Dependencias
└── pares_KO_PEP_diario_2010_2025.csv  # Datos históricos

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
