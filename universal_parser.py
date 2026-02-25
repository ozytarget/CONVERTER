"""
Parser Universal Inteligente para cualquier Broker
Detecta automáticamente el formato y mapea columnas a IRS Form 8949.
Importa utilidades compartidas de broker_parsers para consistencia.
"""

import pandas as pd
import re
from typing import Tuple, Dict, List, Optional
from io import BytesIO
from difflib import SequenceMatcher

# Importar utilidades compartidas
from broker_parsers import _find_header_row, _clean_numeric, _clean_date, _read_file

class UniversalBrokerParser:
    """Parser universal que funciona con cualquier formato de broker"""
    
    # Mapeos de palabras clave para detectar tipos de columnas
    # ─── Palabras clave para detectar columnas ───────────────────────────────
    DATE_ACQUIRED_KEYWORDS = [
        'date acquired', 'open date', 'purchase date', 'buy date', 'entry date',
        'date opened', 'acquisition date', 'fecha compra', 'fecha adquisición',
        'open_date', 'date_acquired', 'acquired', 'opening date', 'opening_date',
    ]

    DATE_SOLD_KEYWORDS = [
        'date sold', 'close date', 'sale date', 'sell date', 'exit date',
        'fecha venta', 'fecha cierre', 'close_date', 'sold_date', 'sale_date',
        'date_closed', 'date_sold', 'closing date', 'closing_date',
    ]

    PROCEEDS_KEYWORDS = [
        'proceeds', 'sale proceeds', 'sell proceeds', 'proceeds amount',
        'sale amount', 'total proceeds', 'gross proceeds', 'ingresos',
        'monto venta', 'proceeds $',
    ]

    COST_BASIS_KEYWORDS = [
        'cost basis', 'basis', 'adjusted basis', 'adjusted cost basis',
        'cost', 'amount invested', 'entry cost', 'total cost', 'costo base',
        'costo', 'inversión', 'cost_basis', 'purchase price', 'acquisition cost',
    ]

    GAIN_LOSS_KEYWORDS = [
        'gain or (loss)', 'gain or loss', 'gain/loss', 'realized gain',
        'realized p&l', 'realised p&l', 'net p&l', 'net p/l', 'p&l',
        'p/l', 'profit loss', 'profit/loss', 'gain', 'loss', 'total return',
        'pl', 'ganancia', 'pérdida', 'return',
    ]

    DESCRIPTION_KEYWORDS = [
        'description', 'symbol', 'ticker', 'ticker symbol', 'security',
        'instrument', 'product', 'name', 'fund name', 'underlying',
        'símbolo', 'descripción', 'asset',
    ]

    QUANTITY_KEYWORDS = [
        'quantity', 'shares', 'qty', 'units', 'cantidad', 'acciones',
        'contratos', 'amount', 'number of shares',
    ]
    
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
        Detecta columnas del DataFrame (ya en lowercase) y las mapea al formato estándar.
        Usa búsqueda exacta (substring) primero, luego fuzzy matching.

        Returns:
            {columna_original_lowercase: nombre_estándar}
        """
        df_cols = df.columns.tolist()  # Already lowercase from parse()

        column_types = {
            'Date Acquired': UniversalBrokerParser.DATE_ACQUIRED_KEYWORDS,
            'Date Sold':     UniversalBrokerParser.DATE_SOLD_KEYWORDS,
            'Proceeds':      UniversalBrokerParser.PROCEEDS_KEYWORDS,
            'Cost Basis':    UniversalBrokerParser.COST_BASIS_KEYWORDS,
            'Gain or (loss)': UniversalBrokerParser.GAIN_LOSS_KEYWORDS,
            'Description':   UniversalBrokerParser.DESCRIPTION_KEYWORDS,
            'Quantity':      UniversalBrokerParser.QUANTITY_KEYWORDS,
        }

        mapping: Dict[str, str] = {}
        used: set = set()

        def _find_col(keywords: List[str], fuzzy_threshold: float = 0.72) -> Optional[str]:
            # Pass 1: exact substring match (most reliable)
            for col in df_cols:
                if col in used:
                    continue
                for kw in keywords:
                    if kw in col or col in kw:
                        return col
            # Pass 2: fuzzy match
            best_col, best_score = None, fuzzy_threshold
            for col in df_cols:
                if col in used:
                    continue
                for kw in keywords:
                    score = SequenceMatcher(None, kw, col).ratio()
                    if score > best_score:
                        best_score, best_col = score, col
            return best_col

        for standard_col, keywords in column_types.items():
            match = _find_col(keywords)
            if match:
                mapping[match] = standard_col
                used.add(match)

        return mapping
    
    @staticmethod
    def clean_numeric_value(value) -> float:
        """Alias que llama a la utilidad compartida."""
        return _clean_numeric(value)

    @staticmethod
    def clean_date_value(value) -> str:
        """Alias que llama a la utilidad compartida."""
        return _clean_date(value)

    @staticmethod
    def parse(file_data: BytesIO, filename: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Parser universal que:
        1. Detecta automáticamente la fila de encabezado (puede estar en cualquier fila)
        2. Mapea columnas al formato estándar IRS 8949 con fuzzy matching
        3. Limpia valores numéricos y fechas
        4. Calcula Gain/Loss si no existe
        5. Elimina filas de resumen/totales
        """
        try:
            # Leer archivo con detección automática de encabezado
            df = _read_file(file_data, filename)

            # Guardar nombres originales antes de bajar a lowercase para el mapeo
            original_columns = df.columns.tolist()
            df.columns = [c.lower().strip() for c in df.columns]

            # Detectar y mapear columnas (trabaja con columnas ya en lowercase)
            column_mapping = UniversalBrokerParser.detect_and_map_columns(df)

            if not column_mapping:
                raise ValueError(
                    "No se pudieron detectar columnas válidas en el archivo.\n"
                    "Verifica que el archivo tenga columnas de: fechas, símbolo/descripción, "
                    "ingresos (proceeds) y costo base (cost basis)."
                )

            # Renombrar columnas al formato estándar
            df = df.rename(columns=column_mapping)

            # Eliminar columnas extra que no necesitamos (sin nombre estándar)
            standard_cols = {'Description', 'Date Acquired', 'Date Sold',
                             'Proceeds', 'Cost Basis', 'Gain or (loss)', 'Quantity'}
            extra_cols = [c for c in df.columns if c not in standard_cols]
            df = df.drop(columns=extra_cols, errors='ignore')

            # Limpiar numéricos
            for col in ('Proceeds', 'Cost Basis', 'Gain or (loss)', 'Quantity'):
                if col in df.columns:
                    df[col] = df[col].apply(_clean_numeric)

            # Limpiar fechas (marcando fechas corruptas como 'Various')
            for col in ('Date Acquired', 'Date Sold'):
                if col in df.columns:
                    df[col] = df[col].apply(_clean_date)
                else:
                    df[col] = 'Various'

            # Calcular Gain/Loss si no existe o normalizarlo
            if 'Gain or (loss)' not in df.columns:
                if 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
                    df['Gain or (loss)'] = df['Proceeds'] - df['Cost Basis']
                else:
                    df['Gain or (loss)'] = 0.0

            # AUTO-CORREGIR filas con datos numéricos inconsistentes
            df = df.apply(UniversalBrokerParser.auto_fix_corrupted_row, axis=1)

            # Asegurar que existen todas las columnas requeridas
            required_cols = ['Description', 'Date Acquired', 'Date Sold',
                             'Proceeds', 'Cost Basis', 'Gain or (loss)']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = ''

            df = df[required_cols].copy()

            # Eliminar filas de totales / resumen / vacías
            GARBAGE_PATTERNS = re.compile(
                r'^(total|subtotal|grand total|sum|summary|net|nan|none|n/a|--)$',
                re.IGNORECASE
            )
            df = df[~df['Description'].astype(str).str.strip().apply(
                lambda x: bool(GARBAGE_PATTERNS.match(x)) or x == ''
            )]

            df = df.reset_index(drop=True)

            # Validar y generar advertencias
            warnings = UniversalBrokerParser.validate_data(df)
            return df, warnings

        except Exception as e:
            raise ValueError(f"Error en parser universal: {str(e)}")

    @staticmethod
    def validate_data(df: pd.DataFrame) -> List[str]:
        """Valida datos y retorna lista de advertencias."""
        warnings: List[str] = []
        if df.empty:
            return warnings

        if 'Proceeds' in df.columns and 'Cost Basis' in df.columns and 'Gain or (loss)' in df.columns:
            calculated = df['Proceeds'] - df['Cost Basis']
            mismatches = df[abs(df['Gain or (loss)'] - calculated) >= 0.01]
            if len(mismatches) > 0:
                warnings.append(
                    f"⚠️ {len(mismatches)} transacciones con Gain/Loss que no cuadra con "
                    f"Proceeds − Cost Basis. Revisa los datos originales."
                )
                for row_num, (_, row) in enumerate(mismatches.head(10).iterrows(), 1):
                    calc = row['Proceeds'] - row['Cost Basis']
                    warnings.append(
                        f"  Fila {row_num}: {str(row['Description'])[:40]} | "
                        f"Reportado: ${float(row['Gain or (loss)']):.2f} | "
                        f"Calculado: ${float(calc):.2f}"
                    )

        # Avisar sobre fechas 'Various' (válido para IRS pero útil saberlo)
        if 'Date Acquired' in df.columns:
            various_count = (df['Date Acquired'] == 'Various').sum()
            if various_count > 0:
                warnings.append(
                    f"ℹ️ {various_count} transacciones tienen 'Various' como fecha de "
                    f"adquisición (válido para IRS Form 8949)."
                )

        return warnings

