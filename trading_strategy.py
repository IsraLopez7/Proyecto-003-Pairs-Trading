"""
Implementación de la estrategia de trading de pares
"""

import numpy as np
import pandas as pd
from config import *

class PairsTradingStrategy:
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        """
        Inicializa la estrategia de trading
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # -1: short spread, 0: neutral, 1: long spread
        self.trades = []
        self.portfolio_value = [initial_capital]
        self.positions_history = []
        
        # Para tracking de posiciones
        self.ko_position = 0
        self.pep_position = 0
        self.entry_spread_value = 0
        self.borrowed_value = 0  # Valor prestado para posiciones cortas
        
    def calculate_position_size(self, ko_price, pep_price, hedge_ratio):
        """
        Calcula el tamaño de las posiciones considerando el ratio de cobertura
        y usando el 80% del capital disponible
        """
        available_capital = self.capital * POSITION_SIZE
        
        # Valor total necesario para la posición
        # Para 1 unidad de KO necesitamos hedge_ratio unidades de PEP
        total_value_per_unit = ko_price + abs(hedge_ratio) * pep_price
        
        # Número de unidades que podemos comprar
        units = available_capital / total_value_per_unit
        
        ko_shares = int(units)
        pep_shares = int(units * abs(hedge_ratio))
        
        return ko_shares, pep_shares
    
    def open_position(self, signal, ko_price, pep_price, hedge_ratio, date):
        """
        Abre una posición basada en la señal
        """
        if self.position != 0:
            return  # Ya hay una posición abierta
        
        ko_shares, pep_shares = self.calculate_position_size(ko_price, pep_price, hedge_ratio)
        
        if signal == 1:  # Long spread: Comprar KO, Vender PEP
            # Comprar KO
            ko_cost = ko_shares * ko_price * (1 + COMMISSION_RATE)
            # Vender PEP (short)
            pep_revenue = pep_shares * pep_price * (1 - COMMISSION_RATE)
            self.borrowed_value = pep_shares * pep_price  # Registrar préstamo
            
            self.ko_position = ko_shares
            self.pep_position = -pep_shares
            self.position = 1
            
            net_cost = ko_cost - pep_revenue
            self.capital -= net_cost
            
        elif signal == -1:  # Short spread: Vender KO, Comprar PEP
            # Vender KO (short)
            ko_revenue = ko_shares * ko_price * (1 - COMMISSION_RATE)
            self.borrowed_value = ko_shares * ko_price  # Registrar préstamo
            # Comprar PEP
            pep_cost = pep_shares * pep_price * (1 + COMMISSION_RATE)
            
            self.ko_position = -ko_shares
            self.pep_position = pep_shares
            self.position = -1
            
            net_cost = pep_cost - ko_revenue
            self.capital -= net_cost
        
        self.entry_spread_value = ko_price - hedge_ratio * pep_price
        
        trade = {
            'date': date,
            'action': 'open',
            'signal': signal,
            'ko_position': self.ko_position,
            'pep_position': self.pep_position,
            'ko_price': ko_price,
            'pep_price': pep_price,
            'hedge_ratio': hedge_ratio
        }
        self.trades.append(trade)
    
    def close_position(self, ko_price, pep_price, date, days_held=1):
        """
        Cierra la posición actual
        """
        if self.position == 0:
            return  # No hay posición para cerrar
        
        # Calcular costos de préstamo
        borrow_cost = self.borrowed_value * BORROW_RATE * (days_held / 365)
        
        if self.position == 1:  # Cerrar long spread
            # Vender KO
            ko_revenue = abs(self.ko_position) * ko_price * (1 - COMMISSION_RATE)
            # Comprar PEP para cubrir short
            pep_cost = abs(self.pep_position) * pep_price * (1 + COMMISSION_RATE)
            
            net_revenue = ko_revenue - pep_cost - borrow_cost
            
        else:  # Cerrar short spread
            # Comprar KO para cubrir short
            ko_cost = abs(self.ko_position) * ko_price * (1 + COMMISSION_RATE)
            # Vender PEP
            pep_revenue = abs(self.pep_position) * pep_price * (1 - COMMISSION_RATE)
            
            net_revenue = pep_revenue - ko_cost - borrow_cost
        
        self.capital += net_revenue
        
        # Calcular retorno de la operación
        initial_investment = abs(self.ko_position) * self.trades[-1]['ko_price'] + \
                           abs(self.pep_position) * self.trades[-1]['pep_price']
        trade_return = net_revenue / initial_investment if initial_investment > 0 else 0
        
        trade = {
            'date': date,
            'action': 'close',
            'ko_price': ko_price,
            'pep_price': pep_price,
            'pnl': net_revenue,
            'return': trade_return,
            'borrow_cost': borrow_cost
        }
        self.trades.append(trade)
        
        # Resetear posiciones
        self.position = 0
        self.ko_position = 0
        self.pep_position = 0
        self.borrowed_value = 0
        
        return trade_return
    
    def update_portfolio_value(self, ko_price, pep_price):
        """
        Actualiza el valor del portfolio considerando posiciones abiertas
        """
        # Valor en efectivo
        portfolio_value = self.capital
        
        # Valor de posiciones abiertas (mark-to-market)
        if self.position != 0:
            ko_value = self.ko_position * ko_price
            pep_value = self.pep_position * pep_price
            portfolio_value += ko_value + pep_value
        
        self.portfolio_value.append(portfolio_value)
        return portfolio_value
    
    def calculate_statistics(self):
        """
        Calcula estadísticas de trading
        """
        # Filtrar solo operaciones cerradas
        closed_trades = [t for t in self.trades if t['action'] == 'close']
        
        if len(closed_trades) == 0:
            print("No hay operaciones cerradas para analizar")
            return None
        
        returns = [t['return'] for t in closed_trades]
        
        stats = {
            'total_trades': len(closed_trades),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'total_return': (self.portfolio_value[-1] - self.initial_capital) / self.initial_capital
        }
        
        return stats