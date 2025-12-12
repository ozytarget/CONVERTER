"""
Parser Universal Inteligente para Brokers
Detecta automáticamente el formato y mapea columnas de cualquier broker
"""

import pandas as pd
import re
from typing import Tuple, Dict, List, Optional
from io import BytesIO
from difflib import SequenceMatcher

class UniversalBrokerParser:
    """Parser universal que funciona con cualquier formato de broker"""
    
    # Mapeos de palabras clave para detectar tipos de columnas
    DATE_ACQUIRED_KEYWORDS = [
        'date acquired', 'open date', 'purchase date', 'buy date', 'entry date',
        'date opened', 'acquisition date', 'fecha compra', 'fecha adquisición',
        'open_date', 'acquired', 'bought', 'entry_date', 'date_acquired'
    ]
    
    DATE_SOLD_KEYWORDS = [
        'date sold', 'close date', 'sale date', 'sell date', 'exit date',
        'fecha venta', 'fecha cierre', 'fecha salida', 'close_date',
        'sold_date', 'sale_date', 'date_closed', 'date_sold'
    ]
    
    PROCEEDS_KEYWORDS = [
        'proceeds', 'sale proceeds', 'proceeds amount', 'sale amount',
        'monto venta', 'ingresos', 'proceeds $', 'proceeds amount', 'total proceeds'
    ]
    
    COST_BASIS_KEYWORDS = [
        'cost basis', 'basis', 'cost', 'amount invested', 'entry cost',
        'total cost', 'costo base', 'costo', 'inversión', 'cost_basis',
        'purchase price', 'acquisition cost', 'adjusted basis'
    ]
    
    GAIN_LOSS_KEYWORDS = [
        'gain', 'loss', 'gain or loss', 'gain/loss', 'gain or (loss)',
        'p&l', 'profit loss', 'return', 'pl', 'ganancia pérdida',
        'ganancia', 'pérdida', 'total return', 'realized gain', 'realized loss'
    ]
    
    DESCRIPTION_KEYWORDS = [
        'symbol', 'ticker', 'description', 'security', 'instrument',
        'product', 'name', 'símbolo', 'descripción'
    ]
    
    QUANTITY_KEYWORDS = [
        'quantity', 'shares', 'qty', 'amount', 'units', 'cantidad',
        'acciones', 'contratos'
    ]
    
    @staticmethod
    def find_matching_column(df_columns: List[str], keywords: List[str], threshold: float = 0.6) -> Optional[str]:
        """
        Encuentra una columna que coincida con las palabras clave
        Usa fuzzy matching para ser más flexible
        
        Args:
            df_columns: Lista de columnas del DataFrame
            keywords: Lista de palabras clave a buscar
            threshold: Similitud mínima (0-1) para considerar como coincidencia
        
        Returns:
            Nombre de la columna coincidente o None
        """
        df_columns_lower = {col: col.lower() for col in df_columns}
        
        for col, col_lower in df_columns_lower.items():
            for keyword in keywords:
                # Búsqueda exacta primero
                if keyword in col_lower:
                    return col
                
                # Búsqueda fuzzy si no es exacta
                similarity = SequenceMatcher(None, keyword, col_lower).ratio()
                if similarity >= threshold:
                    return col
        
        return None
    
    @staticmethod
    def detect_and_map_columns(df: pd.DataFrame) -> Dict[str, str]:
        """
        Detecta automáticamente las columnas y crea un mapeo a formato estándar
        
        Returns:
            Diccionario con mapeo: {columna_original: columna_estándar}
        """
        df_columns = df.columns.tolist()
        mapping = {}
        
        # Mapear cada tipo de columna
        column_types = {
            'Date Acquired': UniversalBrokerParser.DATE_ACQUIRED_KEYWORDS,
            'Date Sold': UniversalBrokerParser.DATE_SOLD_KEYWORDS,
            'Proceeds': UniversalBrokerParser.PROCEEDS_KEYWORDS,
            'Cost Basis': UniversalBrokerParser.COST_BASIS_KEYWORDS,
            'Gain or (loss)': UniversalBrokerParser.GAIN_LOSS_KEYWORDS,
            'Description': UniversalBrokerParser.DESCRIPTION_KEYWORDS,
            'Quantity': UniversalBrokerParser.QUANTITY_KEYWORDS,
        }
        
        found_columns = set()
        
        for standard_col, keywords in column_types.items():
            matched_col = UniversalBrokerParser.find_matching_column(
                [col for col in df_columns if col not in found_columns],
                keywords
            )
            if matched_col:
                mapping[matched_col] = standard_col
                found_columns.add(matched_col)
        
        return mapping
    
    @staticmethod
    def clean_numeric_value(value) -> float:
        """Limpia valores numéricos de cualquier formato"""
        if pd.isna(value):
            return 0.0
        
        value_str = str(value).strip()
        # Remover símbolos de moneda, signos de porcentaje, espacios, comas
        value_str = re.sub(r'[$%,\s]', '', value_str)
        
        try:
            return float(value_str)
        except ValueError:
            return 0.0
    
    @staticmethod
    def clean_date_value(value) -> str:
        """Limpia y normaliza fechas a formato MM/DD/YYYY"""
        if pd.isna(value):
            return 'Various'
        
        value_str = str(value).strip().lower()
        
        if 'various' in value_str or value_str == '':
            return 'Various'
        
        try:
            parsed_date = pd.to_datetime(value_str, errors='coerce')
            if pd.isna(parsed_date):
                return 'Invalid Date'
            return parsed_date.strftime('%m/%d/%Y')
        except:
            return 'Invalid Date'
    
    @staticmethod
    def parse(file_data: BytesIO, filename: str) -> pd.DataFrame:
        """
        Parse universal que funciona con cualquier formato de broker
        """
        try:
            # Intentar leer el archivo
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(file_data)
            else:
                # Intentar diferentes encodings para CSV
                file_data.seek(0)
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        file_data.seek(0)
                        df = pd.read_csv(file_data, encoding=encoding)
                        break
                    except:
                        continue
            
            # Detectar y mapear columnas
            column_mapping = UniversalBrokerParser.detect_and_map_columns(df)
            
            if not column_mapping:
                raise ValueError("No se pudieron detectar columnas válidas en el archivo")
            
            # Renombrar columnas
            df = df.rename(columns=column_mapping)
            
            # Limpiar datos numéricos
            numeric_cols = ['Proceeds', 'Cost Basis', 'Quantity', 'Gain or (loss)']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].apply(UniversalBrokerParser.clean_numeric_value)
            
            # Limpiar fechas
            date_cols = ['Date Acquired', 'Date Sold']
            for col in date_cols:
                if col in df.columns:
                    df[col] = df[col].apply(UniversalBrokerParser.clean_date_value)
            
            # Asegurar que existe 'Gain or (loss)'
            if 'Gain or (loss)' not in df.columns:
                if 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
                    df['Gain or (loss)'] = df['Proceeds'] - df['Cost Basis']
                else:
                    df['Gain or (loss)'] = 0.0
            
            # Asegurar columnas mínimas
            required_cols = ['Description', 'Date Acquired', 'Date Sold', 'Proceeds', 'Cost Basis', 'Gain or (loss)']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = ''
            
            # Seleccionar solo columnas necesarias
            df = df[required_cols]
            
            # Remover filas vacías
            df = df.dropna(how='all')
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error procesando archivo: {str(e)}")
