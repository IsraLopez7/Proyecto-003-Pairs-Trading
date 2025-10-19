"""
Módulo para manejo y preprocesamiento de datos
"""

import pandas as pd
import numpy as np
from config import *

class DataHandler:
    def __init__(self, data_path=DATA_FILE):
        """
        Inicializa el manejador de datos
        """
        self.data_path = data_path
        self.data = None
        self.train_data = None
        self.test_data = None
        self.val_data = None
        
    def load_data(self):
        """
        Carga los datos desde el archivo CSV
        """
        self.data = pd.read_csv(self.data_path)
        self.data['fecha'] = pd.to_datetime(self.data['fecha'], format='%d/%m/%Y')
        self.data = self.data.sort_values('fecha')
        self.data = self.data.reset_index(drop=True)
        
        # Verificar y limpiar datos faltantes
        if self.data.isnull().any().any():
            print(f"Datos faltantes detectados: {self.data.isnull().sum().sum()} valores")
            self.data = self.data.ffill()
            
        print(f"Datos cargados: {len(self.data)} registros")
        print(f"Período: {self.data['fecha'].iloc[0]} a {self.data['fecha'].iloc[-1]}")
        
        return self.data
    
    def split_data(self):
        """
        Divide los datos en entrenamiento, prueba y validación (cronológicamente)
        """
        n = len(self.data)
        train_end = int(n * TRAIN_RATIO)
        test_end = int(n * (TRAIN_RATIO + TEST_RATIO))
        
        self.train_data = self.data.iloc[:train_end].copy()
        self.test_data = self.data.iloc[train_end:test_end].copy()
        self.val_data = self.data.iloc[test_end:].copy()
        
        print(f"\nDivisión de datos:")
        print(f"Entrenamiento: {len(self.train_data)} registros ({self.train_data['fecha'].iloc[0]} a {self.train_data['fecha'].iloc[-1]})")
        print(f"Prueba: {len(self.test_data)} registros ({self.test_data['fecha'].iloc[0]} a {self.test_data['fecha'].iloc[-1]})")
        print(f"Validación: {len(self.val_data)} registros ({self.val_data['fecha'].iloc[0]} a {self.val_data['fecha'].iloc[-1]})")
        
        return self.train_data, self.test_data, self.val_data
    
    def get_prices(self, data_subset=None):
        """
        Obtiene los precios ajustados para ambos activos
        """
        if data_subset is None:
            data_subset = self.data
            
        ko_prices = data_subset[f'{TICKER_1}_AdjClose'].values
        pep_prices = data_subset[f'{TICKER_2}_AdjClose'].values
        
        return ko_prices, pep_prices
    
    def calculate_returns(self, data_subset=None):
        """
        Calcula los retornos logarítmicos
        """
        if data_subset is None:
            data_subset = self.data
            
        ko_returns = data_subset[f'{TICKER_1}_LogRet'].values
        pep_returns = data_subset[f'{TICKER_2}_LogRet'].values
        
        # Eliminar NaN del primer valor
        ko_returns = ko_returns[~np.isnan(ko_returns)]
        pep_returns = pep_returns[~np.isnan(pep_returns)]
        
        return ko_returns, pep_returns