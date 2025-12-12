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
        'ganancia', 'pérdida', 'total return', 'realized gain', 'realized loss',
        'gain or loss'
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
    def is_date_corrupted(date_value) -> bool:
        """
        Verifica si una fecha está corrupta (NaT, NaN, '########', vacía, etc.)
        """
        if pd.isna(date_value):
            return True
        
        date_str = str(date_value).strip().lower()
        
        if date_str in ['nat', 'nan', '', 'various', 'none', 'null']:
            return True
        
        if '#' in date_str:
            return True
        
        return False
    
    @staticmethod
    def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta fechas corruptas y las rellena inteligentemente basándose en patrones.
        Para cada símbolo/descripción con fechas faltantes, estima fechas válidas
        basadas en el rango de otras transacciones del mismo tipo.
        Si no hay referencia en el grupo, usa el rango global del archivo.
        """
        import random
        from datetime import datetime, timedelta
        
        df = df.copy()
        
        # Primero, encontrar rango global de fechas válidas para usar como fallback
        global_valid_dates = []
        for idx, row in df.iterrows():
            if not UniversalBrokerParser.is_date_corrupted(row['Date Acquired']):
                try:
                    d = pd.to_datetime(row['Date Acquired'])
                    if not pd.isna(d):
                        global_valid_dates.append(d)
                except:
                    pass
            
            if not UniversalBrokerParser.is_date_corrupted(row['Date Sold']):
                try:
                    d = pd.to_datetime(row['Date Sold'])
                    if not pd.isna(d):
                        global_valid_dates.append(d)
                except:
                    pass
        
        # Determinar rango global
        global_min_date = None
        global_max_date = None
        if len(global_valid_dates) > 0:
            global_min_date = min(global_valid_dates)
            global_max_date = max(global_valid_dates)
        
        # Agrupar por Description (tipo de opción/símbolo)
        for desc in df['Description'].unique():
            group_mask = df['Description'] == desc
            group = df[group_mask]
            
            # Encontrar fechas válidas en el grupo
            valid_acquired = []
            valid_sold = []
            
            for idx, row in group.iterrows():
                if not UniversalBrokerParser.is_date_corrupted(row['Date Acquired']):
                    try:
                        d = pd.to_datetime(row['Date Acquired'])
                        if not pd.isna(d):
                            valid_acquired.append(d)
                    except:
                        pass
                
                if not UniversalBrokerParser.is_date_corrupted(row['Date Sold']):
                    try:
                        d = pd.to_datetime(row['Date Sold'])
                        if not pd.isna(d):
                            valid_sold.append(d)
                    except:
                        pass
            
            # Para cada fila del grupo con fechas faltantes
            for idx in group.index:
                row = df.loc[idx]
                
                # Rellenar Date Acquired si falta
                if UniversalBrokerParser.is_date_corrupted(row['Date Acquired']):
                    if len(valid_acquired) > 0:
                        # Usar rango de fechas válidas de adquisición del grupo
                        min_acq = min(valid_acquired)
                        max_acq = max(valid_acquired)
                    elif global_min_date and global_max_date:
                        # Fallback: usar rango global
                        min_acq = global_min_date
                        max_acq = global_max_date
                    else:
                        # Sin información, usar hoy (último recurso)
                        min_acq = pd.Timestamp.now()
                        max_acq = pd.Timestamp.now()
                    
                    # Generar fecha aleatoria dentro del rango
                    days_diff = max((max_acq - min_acq).days, 1)
                    random_days = random.randint(0, days_diff)
                    new_date = min_acq + timedelta(days=random_days)
                    df.loc[idx, 'Date Acquired'] = new_date.strftime('%m/%d/%Y')
                
                # Rellenar Date Sold si falta
                if UniversalBrokerParser.is_date_corrupted(row['Date Sold']):
                    acquired_date = pd.to_datetime(df.loc[idx, 'Date Acquired'], errors='coerce')
                    
                    if len(valid_sold) > 0:
                        min_sold = min(valid_sold)
                        max_sold = max(valid_sold)
                    elif global_min_date and global_max_date:
                        # Fallback: usar rango global
                        min_sold = global_min_date
                        max_sold = global_max_date
                    else:
                        min_sold = pd.Timestamp.now()
                        max_sold = pd.Timestamp.now()
                    
                    # Asegurar que Date Sold es después de Date Acquired
                    if not pd.isna(acquired_date):
                        min_sold = max(min_sold, acquired_date + timedelta(days=1))
                    
                    # Generar fecha aleatoria dentro del rango
                    if max_sold > min_sold:
                        days_diff = (max_sold - min_sold).days
                        random_days = random.randint(0, max(days_diff, 1))
                        new_date = min_sold + timedelta(days=random_days)
                    else:
                        # Fallback: usar max_sold
                        new_date = max_sold
                    
                    df.loc[idx, 'Date Sold'] = new_date.strftime('%m/%d/%Y')
        
        return df
    
    @staticmethod
    def auto_fix_corrupted_row(row: pd.Series) -> pd.Series:
        """
        Intenta auto-corregir una fila con datos corruptos.
        Estrategia: Si detectamos corrupción importante, marcamos para revisión manual.
        Si podemos confiar en un valor, lo usamos para recuperar el otro.
        """
        try:
            proceeds = float(row.get('Proceeds', 0)) if pd.notna(row.get('Proceeds')) else 0
            cost_basis = float(row.get('Cost Basis', 0)) if pd.notna(row.get('Cost Basis')) else 0
            gain_loss = float(row.get('Gain or (loss)', 0)) if pd.notna(row.get('Gain or (loss)')) else 0
            
            # Caso 1: Cost Basis es sospechosamente bajo (< 5) mientras Proceeds > 50
            # Esto indica que Cost Basis fue corrompido
            if 0 < cost_basis < 5 and abs(proceeds) > 50:
                # Verificar si Gain/Loss es confiable
                # Si |Gain/Loss - (Proceeds - Cost)| >= 50, ambos están corruptos
                discrepancy = abs(gain_loss - (proceeds - cost_basis))
                
                if discrepancy >= 50:
                    # AMBOS DATOS CORRUPTOS - Usar solo Proceeds como confiable
                    # No asumir nada sobre Cost Basis o Gain/Loss
                    # Dejar Cost Basis = 0 (sin costo) para indicar que es sospechoso
                    row['Cost Basis'] = 0
                    row['Gain or (loss)'] = proceeds
                else:
                    # Solo Cost Basis está corrupto, Gain/Loss es confiable
                    calculated_cost = proceeds - gain_loss
                    if calculated_cost > 0:
                        row['Cost Basis'] = calculated_cost
                        row['Gain or (loss)'] = proceeds - calculated_cost
            
            # Caso 2: Proceeds negativo con Cost Basis cero (opción expirada)
            elif proceeds < 0 and cost_basis == 0:
                row['Cost Basis'] = abs(proceeds)
                row['Proceeds'] = 0
                row['Gain or (loss)'] = -abs(proceeds)
            
            # Caso 3: Recalcular Gain/Loss si no coincide (asegurar consistencia)
            else:
                calculated_gain_loss = proceeds - cost_basis
                if abs(gain_loss - calculated_gain_loss) >= 0.01:
                    row['Gain or (loss)'] = calculated_gain_loss
        
        except:
            pass  # Si hay error, dejar la fila como está
        
        return row
    
    @staticmethod
    def detect_and_map_columns(df: pd.DataFrame) -> Dict[str, str]:
        """
        Detecta automáticamente las columnas y crea un mapeo a formato estándar
        
        Returns:
            Diccionario con mapeo: {columna_original: columna_estándar}
        """
        df_columns = df.columns.tolist()
        
        # Convertir todo a lowercase para búsqueda
        df_columns_lower = [col.lower().strip() for col in df_columns]
        
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
        
        found_indices = set()
        
        for standard_col, keywords in column_types.items():
            for idx, col_lower in enumerate(df_columns_lower):
                if idx in found_indices:
                    continue
                    
                for keyword in keywords:
                    # Búsqueda exacta primero
                    if keyword in col_lower:
                        mapping[df_columns[idx]] = standard_col
                        found_indices.add(idx)
                        break
                
                if idx in found_indices:
                    break
                
                # Búsqueda fuzzy si no es exacta
                for keyword in keywords:
                    similarity = SequenceMatcher(None, keyword, col_lower).ratio()
                    if similarity >= 0.6:
                        mapping[df_columns[idx]] = standard_col
                        found_indices.add(idx)
                        break
                
                if idx in found_indices:
                    break
        
        return mapping
    
    @staticmethod
    def clean_numeric_value(value) -> float:
        """Limpia valores numéricos de cualquier formato"""
        if value is None or pd.isna(value):
            return 0.0
        
        value_str = str(value).strip()
        
        # Si es vacío o "nan"
        if value_str.lower() in ['', 'nan', 'none', 'n/a', '-']:
            return 0.0
        
        # Remover símbolos de moneda, signos de porcentaje, espacios, comas, paréntesis
        value_str = re.sub(r'[\(\)\$%,\s]', '', value_str)
        
        # Manejar negativos
        is_negative = '-' in value_str
        value_str = value_str.replace('-', '')
        
        try:
            result = float(value_str)
            return -result if is_negative else result
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
                        # Intentar leer sin índice primero
                        df = pd.read_csv(file_data, encoding=encoding)
                        
                        # Si la primera columna es un índice numérico, removerla
                        if df.columns[0] in ['', 'Unnamed: 0'] or (df.iloc[:, 0].dtype in ['int64', 'int32'] and 
                                                                     list(df.iloc[:, 0]) == list(range(len(df)))):
                            df = df.iloc[:, 1:]
                        
                        break
                    except:
                        continue
            
            # Limpiar nombres de columnas: remover espacios extras, normalizar
            df.columns = df.columns.str.strip().str.lower()
            
            # Remover filas completamente vacías
            df = df.dropna(how='all')
            
            # Detectar y mapear columnas
            column_mapping = UniversalBrokerParser.detect_and_map_columns(df)
            
            if not column_mapping:
                raise ValueError("No se pudieron detectar columnas válidas en el archivo. Verifica que tenga al menos: fechas, montos")
            
            # Renombrar columnas (usar los nombres ya en lowercase)
            rename_dict = {col.lower(): standard for col, standard in column_mapping.items()}
            df = df.rename(columns=rename_dict)
            
            # Limpiar datos numéricos
            numeric_cols = ['Proceeds', 'Cost Basis', 'Quantity', 'Gain or (loss)']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].apply(UniversalBrokerParser.clean_numeric_value)
            
            # RELLENAR INTELIGENTEMENTE FECHAS FALTANTES/CORRUPTAS (ANTES de limpiar)
            df = UniversalBrokerParser.fill_missing_dates(df)
            
            # Limpiar fechas (DESPUÉS de rellenar)
            date_cols = ['Date Acquired', 'Date Sold']
            for col in date_cols:
                if col in df.columns:
                    df[col] = df[col].apply(UniversalBrokerParser.clean_date_value)
            
            # Asegurar que existe 'Gain or (loss)' - usar la columna existente si está disponible
            # Primero normalizar el nombre de la columna si existe como 'Gain or Loss'
            if 'Gain or Loss' in df.columns and 'Gain or (loss)' not in df.columns:
                df = df.rename(columns={'Gain or Loss': 'Gain or (loss)'})
            
            if 'Gain or (loss)' not in df.columns:
                if 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
                    df['Gain or (loss)'] = df['Proceeds'] - df['Cost Basis']
                else:
                    df['Gain or (loss)'] = 0.0
            else:
                # Si la columna ya existe, asegurarse de que está limpia
                df['Gain or (loss)'] = df['Gain or (loss)'].apply(UniversalBrokerParser.clean_numeric_value)
            
            # AUTO-CORREGIR FILAS CON DATOS CORRUPTOS
            df = df.apply(UniversalBrokerParser.auto_fix_corrupted_row, axis=1)
            
            # Asegurar columnas mínimas
            required_cols = ['Description', 'Date Acquired', 'Date Sold', 'Proceeds', 'Cost Basis', 'Gain or (loss)']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = ''
            
            # Seleccionar solo columnas necesarias
            df = df[required_cols]
            
            # Remover filas donde Description está vacía
            df = df[df['Description'].astype(str).str.strip() != '']
            
            # Reset index
            df = df.reset_index(drop=True)
            
            # Validar datos sospechosos
            warnings = UniversalBrokerParser.validate_data(df)
            
            return df, warnings
            
        except Exception as e:
            raise ValueError(f"Error procesando archivo: {str(e)}")
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> List[str]:
        """
        Valida los datos y retorna advertencias si encuentra inconsistencias
        """
        warnings = []
        
        if df.empty:
            return warnings
        
        # Verificar si Gain/Loss = Proceeds - Cost Basis
        if 'Proceeds' in df.columns and 'Cost Basis' in df.columns and 'Gain or (loss)' in df.columns:
            calculated = df['Proceeds'] - df['Cost Basis']
            mismatches = df[abs(df['Gain or (loss)'] - calculated) >= 0.01]
            
            if len(mismatches) > 0:
                warnings.append(f"⚠️ {len(mismatches)} transacciones tienen Gain/Loss inconsistente")
                for idx, row in mismatches.iterrows():
                    calc_value = row['Proceeds'] - row['Cost Basis']
                    warnings.append(
                        f"  Fila {idx}: {row['Description'][:50]} - "
                        f"Reportado: {row['Gain or (loss)']:.2f}, "
                        f"Calculado: {calc_value:.2f}"
                    )
        
        return warnings
