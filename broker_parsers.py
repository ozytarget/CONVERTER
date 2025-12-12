"""
Parsers para diferentes formatos de brokers.
Cada broker tiene su propio formato de datos.
"""

import pandas as pd
import re
from typing import Tuple, Dict, List
from io import BytesIO

class BrokerParser:
    """Clase base para parsers de brokers"""
    
    @staticmethod
    def parse(file_data: BytesIO, filename: str) -> pd.DataFrame:
        """Parse genérico - debe ser sobrescrito"""
        raise NotImplementedError
    
    @staticmethod
    def detect_broker(filename: str, content_sample: str) -> str:
        """Detecta el broker basado en el nombre del archivo o contenido"""
        raise NotImplementedError


class InteractiveBrokersParser(BrokerParser):
    """Parser para Interactive Brokers (CSV)"""
    
    @staticmethod
    def parse(file_data: BytesIO, filename: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_data, encoding='utf-8')
            
            # Mapear columnas IB a formato estándar
            column_mapping = {
                'Open Date': 'Date Acquired',
                'Close Date': 'Date Sold',
                'Quantity': 'Quantity',
                'T. Price': 'Price',
                'Proceeds': 'Proceeds',
                'Basis': 'Cost Basis',
                'Realised P&L': 'Gain or (loss)',
                'Symbol': 'Description',
            }
            
            df = df.rename(columns=column_mapping)
            df = InteractiveBrokersParser._clean_ib_data(df)
            return df
        except Exception as e:
            raise ValueError(f"Error procesando Interactive Brokers: {str(e)}")
    
    @staticmethod
    def _clean_ib_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpia datos específicos de IB"""
        if 'Date Acquired' in df.columns:
            df['Date Acquired'] = pd.to_datetime(df['Date Acquired'], errors='coerce').dt.strftime('%m/%d/%Y')
        if 'Date Sold' in df.columns:
            df['Date Sold'] = pd.to_datetime(df['Date Sold'], errors='coerce').dt.strftime('%m/%d/%Y')
        
        numeric_cols = ['Proceeds', 'Cost Basis', 'Gain or (loss)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Asegurar que existe Gain or (loss)
        if 'Gain or (loss)' not in df.columns and 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
            df['Gain or (loss)'] = df['Proceeds'] - df['Cost Basis']
        
        return df


class ThinkorswimParser(BrokerParser):
    """Parser para TD Ameritrade thinkorswim (CSV)"""
    
    @staticmethod
    def parse(file_data: BytesIO, filename: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_data, encoding='utf-8', skiprows=1)
            
            column_mapping = {
                'Open Date': 'Date Acquired',
                'Close Date': 'Date Sold',
                'Qty': 'Quantity',
                'Proceeds': 'Proceeds',
                'Basis': 'Cost Basis',
                'Gain/Loss': 'Gain or (loss)',
                'Symbol': 'Description',
            }
            
            df = df.rename(columns=column_mapping)
            df = ThinkorswimParser._clean_thinkorswim_data(df)
            return df
        except Exception as e:
            raise ValueError(f"Error procesando Thinkorswim: {str(e)}")
    
    @staticmethod
    def _clean_thinkorswim_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpia datos específicos de thinkorswim"""
        if 'Date Acquired' in df.columns:
            df['Date Acquired'] = pd.to_datetime(df['Date Acquired'], errors='coerce').dt.strftime('%m/%d/%Y')
        if 'Date Sold' in df.columns:
            df['Date Sold'] = pd.to_datetime(df['Date Sold'], errors='coerce').dt.strftime('%m/%d/%Y')
        
        numeric_cols = ['Proceeds', 'Cost Basis', 'Gain or (loss)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Asegurar que existe Gain or (loss)
        if 'Gain or (loss)' not in df.columns and 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
            df['Gain or (loss)'] = df['Proceeds'] - df['Cost Basis']
        
        return df


class FidelityParser(BrokerParser):
    """Parser para Fidelity (CSV/Excel)"""
    
    @staticmethod
    def parse(file_data: BytesIO, filename: str) -> pd.DataFrame:
        try:
            if filename.endswith('.xlsx'):
                df = pd.read_excel(file_data)
            else:
                df = pd.read_csv(file_data, encoding='utf-8')
            
            column_mapping = {
                'Open Date': 'Date Acquired',
                'Sell Date': 'Date Sold',
                'Quantity': 'Quantity',
                'Price': 'Price',
                'Proceeds': 'Proceeds',
                'Cost Basis': 'Cost Basis',
                'Gain/Loss': 'Gain or (loss)',
                'Symbol': 'Description',
            }
            
            df = df.rename(columns=column_mapping)
            df = FidelityParser._clean_fidelity_data(df)
            return df
        except Exception as e:
            raise ValueError(f"Error procesando Fidelity: {str(e)}")
    
    @staticmethod
    def _clean_fidelity_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpia datos específicos de Fidelity"""
        if 'Date Acquired' in df.columns:
            df['Date Acquired'] = pd.to_datetime(df['Date Acquired'], errors='coerce').dt.strftime('%m/%d/%Y')
        if 'Date Sold' in df.columns:
            df['Date Sold'] = pd.to_datetime(df['Date Sold'], errors='coerce').dt.strftime('%m/%d/%Y')
        
        numeric_cols = ['Proceeds', 'Cost Basis', 'Gain or (loss)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Asegurar que existe Gain or (loss)
        if 'Gain or (loss)' not in df.columns and 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
            df['Gain or (loss)'] = df['Proceeds'] - df['Cost Basis']
        
        return df


class CharlesSchwartzParser(BrokerParser):
    """Parser para Charles Schwab (CSV)"""
    
    @staticmethod
    def parse(file_data: BytesIO, filename: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_data, encoding='utf-8')
            
            column_mapping = {
                'Open Date': 'Date Acquired',
                'Close Date': 'Date Sold',
                'Quantity': 'Quantity',
                'Price': 'Price',
                'Proceeds': 'Proceeds',
                'Cost': 'Cost Basis',
                'Gain/Loss': 'Gain or (loss)',
                'Symbol': 'Description',
            }
            
            df = df.rename(columns=column_mapping)
            df = CharlesSchwartzParser._clean_schwab_data(df)
            return df
        except Exception as e:
            raise ValueError(f"Error procesando Charles Schwab: {str(e)}")
    
    @staticmethod
    def _clean_schwab_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpia datos específicos de Schwab"""
        if 'Date Acquired' in df.columns:
            df['Date Acquired'] = pd.to_datetime(df['Date Acquired'], errors='coerce').dt.strftime('%m/%d/%Y')
        if 'Date Sold' in df.columns:
            df['Date Sold'] = pd.to_datetime(df['Date Sold'], errors='coerce').dt.strftime('%m/%d/%Y')
        
        numeric_cols = ['Proceeds', 'Cost Basis', 'Gain or (loss)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Asegurar que existe Gain or (loss)
        if 'Gain or (loss)' not in df.columns and 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
            df['Gain or (loss)'] = df['Proceeds'] - df['Cost Basis']
        
        return df


class TradeStationParser(BrokerParser):
    """Parser para TradeStation (CSV)"""
    
    @staticmethod
    def parse(file_data: BytesIO, filename: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_data, encoding='utf-8', skiprows=2)
            
            column_mapping = {
                'Entry Date': 'Date Acquired',
                'Exit Date': 'Date Sold',
                'Qty': 'Quantity',
                'Exit Price': 'Price',
                'Proceeds': 'Proceeds',
                'Entry Cost': 'Cost Basis',
                'P&L $': 'Gain or (loss)',
                'Symbol': 'Description',
            }
            
            df = df.rename(columns=column_mapping)
            df = TradeStationParser._clean_tradestation_data(df)
            return df
        except Exception as e:
            raise ValueError(f"Error procesando TradeStation: {str(e)}")
    
    @staticmethod
    def _clean_tradestation_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpia datos específicos de TradeStation"""
        if 'Date Acquired' in df.columns:
            df['Date Acquired'] = pd.to_datetime(df['Date Acquired'], errors='coerce').dt.strftime('%m/%d/%Y')
        if 'Date Sold' in df.columns:
            df['Date Sold'] = pd.to_datetime(df['Date Sold'], errors='coerce').dt.strftime('%m/%d/%Y')
        
        numeric_cols = ['Proceeds', 'Cost Basis', 'Gain or (loss)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Asegurar que existe Gain or (loss)
        if 'Gain or (loss)' not in df.columns and 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
            df['Gain or (loss)'] = df['Proceeds'] - df['Cost Basis']
        
        return df


class RobinhoodParser(BrokerParser):
    """Parser para Robinhood (CSV)"""
    
    @staticmethod
    def parse(file_data: BytesIO, filename: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_data, encoding='utf-8')
            
            # Robinhood tiene columnas específicas
            column_mapping = {
                'Date Opened': 'Date Acquired',
                'Date Closed': 'Date Sold',
                'Shares': 'Quantity',
                'Average Price Paid': 'Price',
                'Proceeds': 'Proceeds',
                'Amount Invested': 'Cost Basis',
                'Gain/Loss': 'Gain or (loss)',
                'Ticker Symbol': 'Description',
                'Open Date': 'Date Acquired',
                'Close Date': 'Date Sold',
                'Quantity': 'Quantity',
                'Proceeds Amount': 'Proceeds',
                'Total Cost': 'Cost Basis',
                'Total Return': 'Gain or (loss)',
                'Symbol': 'Description',
            }
            
            df = df.rename(columns=column_mapping)
            df = RobinhoodParser._clean_robinhood_data(df)
            return df
        except Exception as e:
            raise ValueError(f"Error procesando Robinhood: {str(e)}")
    
    @staticmethod
    def _clean_robinhood_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpia datos específicos de Robinhood"""
        # Robinhood puede usar diferentes formatos de fecha
        date_cols = ['Date Acquired', 'Date Sold']
        for col in date_cols:
            if col in df.columns:
                # Intenta varios formatos
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%m/%d/%Y')
        
        # Limpiar valores numéricos
        numeric_cols = ['Proceeds', 'Cost Basis', 'Gain or (loss)', 'Quantity']
        for col in numeric_cols:
            if col in df.columns:
                # Robinhood puede tener valores con $ y comas
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Si falta 'Gain or (loss)', calcular de Proceeds - Cost Basis
        if 'Gain or (loss)' not in df.columns and 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
            df['Gain or (loss)'] = df['Proceeds'] - df['Cost Basis']
        
        return df


class BrokerDetector:
    """Detecta automáticamente el broker basado en características del archivo"""
    
    BROKERS_MAP = {
        'interactive_brokers': InteractiveBrokersParser,
        'thinkorswim': ThinkorswimParser,
        'fidelity': FidelityParser,
        'charles_schwab': CharlesSchwartzParser,
        'tradestation': TradeStationParser,
        'robinhood': RobinhoodParser,
    }
    
    @staticmethod
    def detect_and_parse(file_data: BytesIO, filename: str) -> Tuple[pd.DataFrame, str]:
        """
        Detecta el broker y parsea el archivo automáticamente
        Retorna: (DataFrame, nombre_broker)
        """
        
        # Intentar detección por nombre de archivo
        filename_lower = filename.lower()
        
        detection_keywords = {
            'interactive_brokers': ['interactive', 'ib_'],
            'thinkorswim': ['thinkorswim', 'thinkorswim'],
            'fidelity': ['fidelity', 'fid_'],
            'charles_schwab': ['schwab', 'charles'],
            'tradestation': ['tradestation', 'ts_'],
            'robinhood': ['robinhood', 'rh_', 'hood'],
        }
        
        for broker_name, keywords in detection_keywords.items():
            if any(kw in filename_lower for kw in keywords):
                parser_class = BrokerDetector.BROKERS_MAP[broker_name]
                df = parser_class.parse(file_data, filename)
                return df, broker_name
        
        # Si no se detecta por nombre, intentar detección inteligente por contenido
        file_data.seek(0)
        try:
            df_test = pd.read_csv(file_data, nrows=1, encoding='utf-8')
            columns = df_test.columns.str.lower()
            file_data.seek(0)
            
            # Detección por columnas características
            if 'open date' in columns and 'basis' in columns:
                return InteractiveBrokersParser.parse(file_data, filename), 'interactive_brokers'
            elif 'close date' in columns and 'realised' in columns:
                return ThinkorswimParser.parse(file_data, filename), 'thinkorswim'
            elif 'sell date' in columns and 'cost basis' in columns:
                return FidelityParser.parse(file_data, filename), 'fidelity'
            elif 'exit date' in columns and 'entry cost' in columns:
                return TradeStationParser.parse(file_data, filename), 'tradestation'
            elif 'ticker symbol' in columns or 'date opened' in columns:
                return RobinhoodParser.parse(file_data, filename), 'robinhood'
            else:
                return CharlesSchwartzParser.parse(file_data, filename), 'charles_schwab'
        except:
            pass
        
        raise ValueError("No se pudo detectar el formato del broker. Por favor especifica manualmente.")
