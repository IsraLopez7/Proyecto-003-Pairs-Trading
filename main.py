"""
Archivo principal para ejecutar la estrategia de Pairs Trading
con KO y PEP usando Filtros de Kalman
"""

import warnings
warnings.filterwarnings('ignore')

from data_handler import DataHandler
from cointegration import CointegrationAnalysis
from backtesting import Backtester
from config import *

def main():
    print("="*60)
    print("PAIRS TRADING: KO-PEP CON FILTROS DE KALMAN")
    print("="*60)
    
    # 1. CARGAR Y PREPARAR DATOS
    print("\n1. CARGANDO DATOS...")
    print("-"*40)
    data_handler = DataHandler()
    data = data_handler.load_data()
    
    # 2. DIVIDIR DATOS
    print("\n2. DIVISIÓN DE DATOS...")
    print("-"*40)
    train_data, test_data, val_data = data_handler.split_data()
    
    # 3. ANÁLISIS DE COINTEGRACIÓN (en datos de entrenamiento)
    print("\n3. ANÁLISIS DE COINTEGRACIÓN...")
    print("-"*40)
    coint_analysis = CointegrationAnalysis()
    
    # Obtener precios de entrenamiento
    ko_train, pep_train = data_handler.get_prices(train_data)
    
    # Calcular correlación
    correlation = coint_analysis.calculate_correlation(ko_train, pep_train)
    
    # Ejecutar regresión OLS
    beta_0, beta_1, residuals = coint_analysis.run_ols_regression(ko_train, pep_train)
    
    # Test de estacionariedad
    is_cointegrated, p_value = coint_analysis.test_stationarity(residuals)
    
    # Calcular half-life
    half_life = coint_analysis.calculate_half_life(residuals)
    
    # 4. BACKTEST EN DATOS DE PRUEBA
    print("\n4. EJECUTANDO BACKTEST EN DATOS DE PRUEBA...")
    print("-"*40)
    
    ko_test, pep_test = data_handler.get_prices(test_data)
    dates_test = test_data['fecha'].values
    
    backtester_test = Backtester(test_data)
    results_test = backtester_test.run_backtest(ko_test, pep_test, dates_test)
    
    # Calcular estadísticas
    stats_test = backtester_test.strategy.calculate_statistics()
    if stats_test:
        print("\nResultados en Datos de Prueba:")
        backtester_test.print_statistics(stats_test)
    
    # 5. VALIDACIÓN FINAL
    print("\n5. VALIDACIÓN FINAL...")
    print("-"*40)
    
    ko_val, pep_val = data_handler.get_prices(val_data)
    dates_val = val_data['fecha'].values
    
    backtester_val = Backtester(val_data)
    results_val = backtester_val.run_backtest(ko_val, pep_val, dates_val)
    
    # Calcular estadísticas
    stats_val = backtester_val.strategy.calculate_statistics()
    if stats_val:
        print("\nResultados en Datos de Validación:")
        backtester_val.print_statistics(stats_val)
    
    # 6. GENERAR GRÁFICOS
    print("\n6. GENERANDO VISUALIZACIONES...")
    print("-"*40)
    
    # Gráficos para el conjunto de validación
    print("Generando gráficos de resultados...")
    backtester_val.plot_results(save_plots=True)
    
    print("Generando distribución de retornos...")
    backtester_val.plot_returns_distribution(save_plot=True)
    
    # 7. RESUMEN FINAL
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    
    if is_cointegrated:
        print("✓ Los pares KO-PEP están COINTEGRADOS")
    else:
        print("✗ Los pares KO-PEP NO están cointegrados")
    
    print(f"Correlación histórica: {correlation:.4f}")
    print(f"Half-life de reversión: {half_life:.1f} días")
    
    if stats_test and stats_val:
        print("\nComparación de Performance:")
        print(f"  Prueba - Retorno Total: {stats_test['total_return']*100:.2f}%")
        print(f"  Validación - Retorno Total: {stats_val['total_return']*100:.2f}%")
        print(f"  Prueba - Sharpe Ratio: {stats_test['sharpe_ratio']:.2f}")
        print(f"  Validación - Sharpe Ratio: {stats_val['sharpe_ratio']:.2f}")
    
    print("\nEjecución completada exitosamente.")
    print("="*60)

if __name__ == "__main__":
    main()