"""
Motor de backtesting para la estrategia de pairs trading
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from kalman_filter import KalmanFilter
from trading_strategy import PairsTradingStrategy
from config import *

class Backtester:
    def __init__(self, data):
        """
        Inicializa el backtester
        """
        self.data = data
        self.kalman = KalmanFilter()
        self.strategy = PairsTradingStrategy()
        self.results = None
        
    def run_backtest(self, ko_prices, pep_prices, dates):
        """
        Ejecuta el backtest sobre los datos proporcionados
        """
        print("\nIniciando backtest...")
        
        # Reiniciar filtro y estrategia
        self.kalman.reset()
        self.strategy = PairsTradingStrategy()
        
        # Ejecutar filtro de Kalman
        spreads, z_scores, hedge_ratios = self.kalman.run_filter(ko_prices, pep_prices)
        
        # Variables para tracking
        current_position = 0
        entry_date = None
        
        # Iterar sobre cada día
        for i in range(len(ko_prices)):
            # Obtener señal de trading
            signal = self.kalman.get_trading_signal(z_scores[i])
            
            # Si hay señal y es diferente a la posición actual
            if signal is not None:
                # Cerrar posición existente si es necesario
                if current_position != 0 and signal == 0:
                    days_held = (dates[i] - entry_date).days if entry_date else 1
                    self.strategy.close_position(ko_prices[i], pep_prices[i], dates[i], days_held)
                    current_position = 0
                    entry_date = None
                
                # Abrir nueva posición
                elif current_position == 0 and signal != 0:
                    self.strategy.open_position(signal, ko_prices[i], pep_prices[i], 
                                               hedge_ratios[i], dates[i])
                    current_position = signal
                    entry_date = dates[i]
            
            # Actualizar valor del portfolio
            self.strategy.update_portfolio_value(ko_prices[i], pep_prices[i])
            
            # Aplicar costo de préstamo diario para posiciones cortas
            if current_position != 0 and self.strategy.borrowed_value > 0:
                daily_borrow_cost = self.strategy.borrowed_value * BORROW_RATE / 365
                self.strategy.capital -= daily_borrow_cost
        
        # Cerrar posición final si queda abierta
        if current_position != 0:
            days_held = (dates[-1] - entry_date).days if entry_date else 1
            self.strategy.close_position(ko_prices[-1], pep_prices[-1], dates[-1], days_held)
        
        # Guardar resultados
        self.results = {
            'spreads': spreads,
            'z_scores': z_scores,
            'hedge_ratios': hedge_ratios,
            'portfolio_values': self.strategy.portfolio_value,
            'trades': self.strategy.trades
        }
        
        return self.results
    
    def plot_results(self, save_plots=False):
        """
        Genera gráficos de resultados
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        
        # Gráfico 1: Evolución del spread
        axes[0].plot(self.results['spreads'], label='Spread', color='blue', alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0].set_title('Evolución del Spread (KO - β*PEP)')
        axes[0].set_ylabel('Spread')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Z-Score con umbrales
        axes[1].plot(self.results['z_scores'], label='Z-Score', color='purple', alpha=0.7)
        axes[1].axhline(y=Z_SCORE_ENTRY, color='red', linestyle='--', label=f'Entrada: ±{Z_SCORE_ENTRY}')
        axes[1].axhline(y=-Z_SCORE_ENTRY, color='red', linestyle='--')
        axes[1].axhline(y=Z_SCORE_EXIT, color='green', linestyle='--', label=f'Salida: ±{Z_SCORE_EXIT}')
        axes[1].axhline(y=-Z_SCORE_EXIT, color='green', linestyle='--')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_title('Z-Score para Señales de Trading')
        axes[1].set_ylabel('Z-Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Gráfico 3: Ratio de cobertura dinámico
        axes[2].plot(self.results['hedge_ratios'], label='β (Ratio de Cobertura)', color='orange')
        axes[2].set_title('Ratio de Cobertura Dinámico a lo Largo del Tiempo')
        axes[2].set_ylabel('β')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Gráfico 4: Valor del portfolio
        axes[3].plot(self.results['portfolio_values'], label='Valor Portfolio', color='green')
        axes[3].axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', label='Capital Inicial')
        axes[3].set_title('Evolución del Valor del Portfolio')
        axes[3].set_xlabel('Días de Trading')
        axes[3].set_ylabel('Valor ($)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('pairs_trading_results.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_returns_distribution(self, save_plot=False):
        """
        Grafica la distribución de retornos por operación
        """
        closed_trades = [t for t in self.strategy.trades if t['action'] == 'close']
        
        if len(closed_trades) == 0:
            print("No hay operaciones cerradas para graficar")
            return
        
        returns = [t['return'] * 100 for t in closed_trades]  # Convertir a porcentaje
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histograma de retornos
        axes[0].hist(returns, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0].set_title('Distribución de Retornos por Operación')
        axes[0].set_xlabel('Retorno (%)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].grid(True, alpha=0.3)
        
        # Estadísticas
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        axes[0].axvline(x=mean_return, color='green', linestyle='-', 
                       label=f'Media: {mean_return:.2f}%')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(returns, vert=True)
        axes[1].set_ylabel('Retorno (%)')
        axes[1].set_title('Box Plot de Retornos')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('returns_distribution.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_statistics(self, stats):
        """
        Imprime estadísticas de trading
        """
        print("\n" + "="*50)
        print("ESTADÍSTICAS DE TRADING")
        print("="*50)
        print(f"Total de operaciones cerradas: {stats['total_trades']}")
        print(f"Retorno promedio por operación: {stats['avg_return']*100:.2f}%")
        print(f"Desviación estándar de retornos: {stats['std_return']*100:.2f}%")
        print(f"Sharpe Ratio (anualizado): {stats['sharpe_ratio']:.2f}")
        print(f"Mejor operación: {stats['max_return']*100:.2f}%")
        print(f"Peor operación: {stats['min_return']*100:.2f}%")
        print(f"Tasa de éxito (Win Rate): {stats['win_rate']*100:.1f}%")
        print(f"Retorno total de la estrategia: {stats['total_return']*100:.2f}%")
        print("="*50)