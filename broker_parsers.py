"""
Parsers para diferentes formatos de brokers.
Normaliza todos los formatos al estándar IRS Form 8949.
Brokers soportados: Interactive Brokers, TD Ameritrade/thinkorswim,
Fidelity, Charles Schwab, TradeStation, Robinhood, Webull,
E*TRADE, Tastytrade, Merrill Edge, Vanguard, Apex Clearing.
El parser universal cubre cualquier otro broker automáticamente.
"""

import pandas as pd
import re
from typing import Tuple, Dict, List, Optional
from io import BytesIO


# ─────────────────────────────────────────────────────────────
# UTILIDADES COMUNES
# ─────────────────────────────────────────────────────────────

def _clean_numeric(val) -> float:
    """Limpia cualquier valor numérico (con $, comas, paréntesis) a float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    s = str(val).strip()
    if not s or s.lower() in ('nan', 'none', 'n/a', '-', '--', ''):
        return 0.0
    s = re.sub(r'[\$,\s%]', '', s)
    if s.startswith('(') and s.endswith(')'):   # (1234.56) → negativo
        s = '-' + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return 0.0


def _clean_date(val) -> str:
    """Normaliza cualquier fecha a MM/DD/YYYY. Devuelve 'Various' si no se puede parsear."""
    if val is None:
        return 'Various'
    s = str(val).strip()
    if not s or s.lower() in ('various', 'nan', 'none', 'n/a', ''):
        return 'Various'
    try:
        dt = pd.to_datetime(s, errors='coerce')
        if pd.isna(dt):
            return 'Various'
        return dt.strftime('%m/%d/%Y')
    except Exception:
        return 'Various'


def _find_header_row(file_data: BytesIO, filename: str, max_scan: int = 30) -> int:
    """
    Busca la fila real del encabezado escaneando las primeras `max_scan` filas.
    Usa heurística: la fila con más palabras clave relacionadas con transacciones.
    """
    HEADER_KEYWORDS = {
        'symbol', 'description', 'date', 'open', 'close', 'acquired', 'sold',
        'proceeds', 'basis', 'cost', 'gain', 'loss', 'qty', 'quantity',
        'shares', 'amount', 'price', 'p&l', 'profit', 'ticker', 'entry', 'exit',
    }
    best_row, best_score = 0, -1
    try:
        file_data.seek(0)
        fname = filename.lower()
        if fname.endswith(('.xlsx', '.xls')):
            raw = pd.read_excel(file_data, header=None, nrows=max_scan, engine='openpyxl')
        else:
            raw = pd.read_csv(file_data, header=None, nrows=max_scan,
                              encoding='utf-8', on_bad_lines='skip')
        for i, row in raw.iterrows():
            score = sum(
                1 for cell in row
                if any(kw in str(cell).lower() for kw in HEADER_KEYWORDS)
            )
            if score > best_score:
                best_score, best_row = score, i
    except Exception:
        pass
    finally:
        file_data.seek(0)
    return int(best_row)  # type: ignore[arg-type]


def _read_file(file_data: BytesIO, filename: str, skiprows: Optional[int] = None) -> pd.DataFrame:
    """
    Lee un CSV o Excel con detección automática de la fila de encabezado.
    Limpia espacios extra en nombres de columna y elimina filas vacías.
    """
    if skiprows is None:
        skiprows = _find_header_row(file_data, filename)

    file_data.seek(0)
    fname = filename.lower()
    try:
        if fname.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_data, header=skiprows, engine='openpyxl')
        else:
            for enc in ('utf-8', 'latin-1', 'cp1252', 'iso-8859-1'):
                try:
                    file_data.seek(0)
                    df = pd.read_csv(file_data, header=skiprows, encoding=enc,
                                     on_bad_lines='skip')
                    break
                except Exception:
                    continue
            else:
                raise ValueError("No se pudo leer el CSV con ningún encoding soportado.")
    except Exception as e:
        raise ValueError(f"Error leyendo archivo: {e}")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how='all').reset_index(drop=True)
    return df


def _normalize_df(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Renombra columnas según mapping, limpia numéricos y fechas,
    calcula Gain/Loss si no existe, retorna DataFrame estándar 8949.
    """
    df = df.rename(columns={k: v for k, v in mapping.items() if v != '_skip'})

    for col in ('Proceeds', 'Cost Basis', 'Gain or (loss)', 'Quantity'):
        if col in df.columns:
            df[col] = df[col].apply(_clean_numeric)

    for col in ('Date Acquired', 'Date Sold'):
        if col in df.columns:
            df[col] = df[col].apply(_clean_date)

    if 'Gain or (loss)' not in df.columns:
        if 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
            df['Gain or (loss)'] = df['Proceeds'] - df['Cost Basis']
        else:
            df['Gain or (loss)'] = 0.0

    required = ['Description', 'Date Acquired', 'Date Sold',
                 'Proceeds', 'Cost Basis', 'Gain or (loss)']
    for col in required:
        if col not in df.columns:
            df[col] = ''

    df = df[required].copy()
    df = df[df['Description'].astype(str).str.strip().ne('')].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────
# PARSERS POR BROKER
# ─────────────────────────────────────────────────────────────

class InteractiveBrokersParser:
    """Interactive Brokers – CSV / Flex Query"""
    NAME = 'Interactive Brokers'
    KEYWORDS = ['interactive', 'ib_', 'ibkr']
    COLUMN_MAP = {
        'Open Date':    'Date Acquired',
        'Close Date':   'Date Sold',
        'Quantity':     'Quantity',
        'T. Price':     'Price',
        'Proceeds':     'Proceeds',
        'Basis':        'Cost Basis',
        'Realised P&L': 'Gain or (loss)',
        'Realized P&L': 'Gain or (loss)',
        'Symbol':       'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        return _normalize_df(df, cls.COLUMN_MAP)


class ThinkorswimParser:
    """TD Ameritrade / thinkorswim"""
    NAME = 'TD Ameritrade / thinkorswim'
    KEYWORDS = ['thinkorswim', 'tdameritrade', 'td_ameritrade', 'tos_']
    COLUMN_MAP = {
        'Open Date':  'Date Acquired',
        'Close Date': 'Date Sold',
        'Qty':        'Quantity',
        'Quantity':   'Quantity',
        'Proceeds':   'Proceeds',
        'Basis':      'Cost Basis',
        'Cost Basis': 'Cost Basis',
        'Gain/Loss':  'Gain or (loss)',
        'Gain or (loss)': 'Gain or (loss)',
        'Symbol':     'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        return _normalize_df(df, cls.COLUMN_MAP)


class FidelityParser:
    """Fidelity Investments"""
    NAME = 'Fidelity'
    KEYWORDS = ['fidelity', 'fid_']
    COLUMN_MAP = {
        'Open Date':   'Date Acquired',
        'Sell Date':   'Date Sold',
        'Date Sold':   'Date Sold',
        'Quantity':    'Quantity',
        'Price':       'Price',
        'Proceeds':    'Proceeds',
        'Cost Basis':  'Cost Basis',
        'Gain/Loss':   'Gain or (loss)',
        'Gain or Loss': 'Gain or (loss)',
        'Symbol':      'Description',
        'Security Description': 'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        return _normalize_df(df, cls.COLUMN_MAP)


class CharlesSchwabParser:
    """Charles Schwab"""
    NAME = 'Charles Schwab'
    KEYWORDS = ['schwab', 'charles schwab', 'schwab_']
    COLUMN_MAP = {
        'Open Date':   'Date Acquired',
        'Close Date':  'Date Sold',
        'Date Sold':   'Date Sold',
        'Quantity':    'Quantity',
        'Price':       'Price',
        'Proceeds':    'Proceeds',
        'Cost':        'Cost Basis',
        'Cost Basis':  'Cost Basis',
        'Gain/Loss':   'Gain or (loss)',
        'Gain or Loss': 'Gain or (loss)',
        'Symbol':      'Description',
        'Security':    'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        # Eliminar filas de totales que Schwab agrega al final
        desc_col = next((c for c in df.columns if c.lower() in ('symbol', 'security')), None)
        if desc_col:
            df = df[~df[desc_col].astype(str).str.lower().isin(
                ['total', 'subtotal', 'grand total', 'nan', ''])]
        return _normalize_df(df, cls.COLUMN_MAP)

    # Alias para compatibilidad
    parse_schwab = parse


# Alias de compatibilidad
CharlesSchwartzParser = CharlesSchwabParser


class TradeStationParser:
    """TradeStation"""
    NAME = 'TradeStation'
    KEYWORDS = ['tradestation', 'ts_']
    COLUMN_MAP = {
        'Entry Date':  'Date Acquired',
        'Exit Date':   'Date Sold',
        'Open Date':   'Date Acquired',
        'Close Date':  'Date Sold',
        'Qty':         'Quantity',
        'Quantity':    'Quantity',
        'Exit Price':  'Price',
        'Proceeds':    'Proceeds',
        'Entry Cost':  'Cost Basis',
        'Cost Basis':  'Cost Basis',
        'P&L $':       'Gain or (loss)',
        'Net P&L':     'Gain or (loss)',
        'Gain/Loss':   'Gain or (loss)',
        'Symbol':      'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        return _normalize_df(df, cls.COLUMN_MAP)


class RobinhoodParser:
    """Robinhood"""
    NAME = 'Robinhood'
    KEYWORDS = ['robinhood', 'rh_', 'hood']
    COLUMN_MAP = {
        'Date Opened':        'Date Acquired',
        'Open Date':          'Date Acquired',
        'Date Closed':        'Date Sold',
        'Close Date':         'Date Sold',
        'Shares':             'Quantity',
        'Quantity':           'Quantity',
        'Average Price Paid': 'Price',
        'Proceeds':           'Proceeds',
        'Proceeds Amount':    'Proceeds',
        'Amount Invested':    'Cost Basis',
        'Total Cost':         'Cost Basis',
        'Cost Basis':         'Cost Basis',
        'Gain/Loss':          'Gain or (loss)',
        'Total Return':       'Gain or (loss)',
        'Ticker Symbol':      'Description',
        'Symbol':             'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        return _normalize_df(df, cls.COLUMN_MAP)


class WebullParser:
    """Webull"""
    NAME = 'Webull'
    KEYWORDS = ['webull', 'wb_']
    COLUMN_MAP = {
        'Open Time':   'Date Acquired',
        'Close Time':  'Date Sold',
        'Open Date':   'Date Acquired',
        'Close Date':  'Date Sold',
        'Qty':         'Quantity',
        'Quantity':    'Quantity',
        'Proceeds':    'Proceeds',
        'Cost':        'Cost Basis',
        'Cost Basis':  'Cost Basis',
        'P&L':         'Gain or (loss)',
        'Profit/Loss': 'Gain or (loss)',
        'Gain/Loss':   'Gain or (loss)',
        'Symbol':      'Description',
        'Ticker':      'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        return _normalize_df(df, cls.COLUMN_MAP)


class ETradeParser:
    """E*TRADE"""
    NAME = 'E*TRADE'
    KEYWORDS = ['etrade', 'e-trade', 'e_trade', 'etrd']
    COLUMN_MAP = {
        'Date Acquired':       'Date Acquired',
        'Acquired':            'Date Acquired',
        'Date Sold':           'Date Sold',
        'Sold':                'Date Sold',
        'Quantity':            'Quantity',
        'Proceeds':            'Proceeds',
        'Cost Basis':          'Cost Basis',
        'Adjusted Cost Basis': 'Cost Basis',
        'Gain or Loss':        'Gain or (loss)',
        'Gain/Loss':           'Gain or (loss)',
        'Description':         'Description',
        'Symbol':              'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        return _normalize_df(df, cls.COLUMN_MAP)


class TastytradeParser:
    """Tastytrade (Tastyworks)"""
    NAME = 'Tastytrade'
    KEYWORDS = ['tastytrade', 'tastyworks', 'tasty']
    COLUMN_MAP = {
        'Opening Date':  'Date Acquired',
        'Closing Date':  'Date Sold',
        'Open Date':     'Date Acquired',
        'Close Date':    'Date Sold',
        'Quantity':      'Quantity',
        'Proceeds':      'Proceeds',
        'Cost Basis':    'Cost Basis',
        'P/L':           'Gain or (loss)',
        'Net P/L':       'Gain or (loss)',
        'Gain/Loss':     'Gain or (loss)',
        'Symbol':        'Description',
        'Underlying':    'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        return _normalize_df(df, cls.COLUMN_MAP)


class MerrillEdgeParser:
    """Merrill Edge / Merrill Lynch"""
    NAME = 'Merrill Edge'
    KEYWORDS = ['merrill', 'merrilledge', 'mlpf']
    COLUMN_MAP = {
        'Date Acquired': 'Date Acquired',
        'Date Sold':     'Date Sold',
        'Quantity':      'Quantity',
        'Proceeds':      'Proceeds',
        'Cost Basis':    'Cost Basis',
        'Gain or Loss':  'Gain or (loss)',
        'Symbol':        'Description',
        'Description':   'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        return _normalize_df(df, cls.COLUMN_MAP)


class VanguardParser:
    """Vanguard"""
    NAME = 'Vanguard'
    KEYWORDS = ['vanguard', 'vgrd']
    COLUMN_MAP = {
        'Acquisition Date': 'Date Acquired',
        'Date Acquired':    'Date Acquired',
        'Sale Date':        'Date Sold',
        'Date Sold':        'Date Sold',
        'Shares':           'Quantity',
        'Quantity':         'Quantity',
        'Sale Proceeds':    'Proceeds',
        'Proceeds':         'Proceeds',
        'Cost Basis':       'Cost Basis',
        'Adjusted Basis':   'Cost Basis',
        'Gain or Loss':     'Gain or (loss)',
        'Fund Name':        'Description',
        'Symbol':           'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        return _normalize_df(df, cls.COLUMN_MAP)


class ApexClearingParser:
    """Apex Clearing (usado por varias plataformas)"""
    NAME = 'Apex Clearing'
    KEYWORDS = ['apex', 'apexclearing', 'apex_']
    COLUMN_MAP = {
        'Open Date':   'Date Acquired',
        'Close Date':  'Date Sold',
        'Quantity':    'Quantity',
        'Proceeds':    'Proceeds',
        'Cost Basis':  'Cost Basis',
        'Gain/Loss':   'Gain or (loss)',
        'Symbol':      'Description',
    }

    @classmethod
    def parse(cls, file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = _read_file(file_data, filename)
        return _normalize_df(df, cls.COLUMN_MAP)


# ─────────────────────────────────────────────────────────────
# REGISTRO GLOBAL DE PARSERS
# ─────────────────────────────────────────────────────────────

ALL_PARSERS = [
    InteractiveBrokersParser,
    ThinkorswimParser,
    FidelityParser,
    CharlesSchwabParser,
    TradeStationParser,
    RobinhoodParser,
    WebullParser,
    ETradeParser,
    TastytradeParser,
    MerrillEdgeParser,
    VanguardParser,
    ApexClearingParser,
]

# Firmas de contenido para detección por texto del archivo
CONTENT_SIGNATURES: Dict[str, str] = {
    'Interactive Brokers': 'Interactive Brokers',
    'realised p&l':        'Interactive Brokers',
    'realized p&l':        'Interactive Brokers',
    't. price':            'Interactive Brokers',
    'entry cost':          'TradeStation',
    'exit price':          'TradeStation',
    'ticker symbol':       'Robinhood',
    'date opened':         'Robinhood',
    'net p/l':             'Tastytrade',
    'tastytrade':          'Tastytrade',
    'tastyworks':          'Tastytrade',
    'acquisition date':    'Vanguard',
    'fund name':           'Vanguard',
    'adjusted cost basis': 'E*TRADE',
    'open time':           'Webull',
    'close time':          'Webull',
    'merrill':             'Merrill Edge',
    'apex clearing':       'Apex Clearing',
    'schwab':              'Charles Schwab',
    'fidelity':            'Fidelity',
    'robinhood':           'Robinhood',
    'webull':              'Webull',
    'vanguard':            'Vanguard',
}

BROKER_PARSER_MAP: Dict[str, type] = {p.NAME: p for p in ALL_PARSERS}


# ─────────────────────────────────────────────────────────────
# DETECTOR AUTOMÁTICO
# ─────────────────────────────────────────────────────────────

class BrokerDetector:
    """
    Detecta automáticamente el broker y parsea el archivo.
    Estrategia:
      1. Nombre del archivo
      2. Contenido (primeras filas) – palabras clave / firmas
      3. Columnas del encabezado detectado
    Si todo falla, lanza ValueError y el caller usa el parser universal.
    """

    @staticmethod
    def detect_and_parse(file_data: BytesIO, filename: str) -> Tuple[pd.DataFrame, str]:
        fname_lower = filename.lower()

        # ── 1. Detección por nombre de archivo ──────────────────
        for parser in ALL_PARSERS:
            if any(kw in fname_lower for kw in parser.KEYWORDS):
                file_data.seek(0)
                return parser.parse(file_data, filename), parser.NAME

        # ── 2. Detección por contenido ───────────────────────────
        file_data.seek(0)
        try:
            if fname_lower.endswith(('.xlsx', '.xls')):
                sample = pd.read_excel(file_data, header=None, nrows=40, engine='openpyxl')
            else:
                sample = pd.read_csv(file_data, header=None, nrows=40,
                                     encoding='utf-8', on_bad_lines='skip')
            content = ' '.join(sample.values.astype(str).flatten()).lower()
        except Exception:
            content = ''

        for sig, broker_name in CONTENT_SIGNATURES.items():
            if sig.lower() in content:
                parser = BROKER_PARSER_MAP.get(broker_name)
                if parser:
                    file_data.seek(0)
                    return parser.parse(file_data, filename), broker_name

        # ── 3. Detección por columnas del encabezado ─────────────
        file_data.seek(0)
        header_row = _find_header_row(file_data, filename)
        file_data.seek(0)
        try:
            if fname_lower.endswith(('.xlsx', '.xls')):
                cols_df = pd.read_excel(file_data, header=header_row, nrows=0, engine='openpyxl')
            else:
                cols_df = pd.read_csv(file_data, header=header_row, nrows=0,
                                      encoding='utf-8', on_bad_lines='skip')
            cols = {c.lower().strip() for c in cols_df.columns}
        except Exception:
            cols = set()

        col_broker_map = [
            ({'realised p&l', 'realized p&l', 't. price'},     'Interactive Brokers'),
            ({'entry cost', 'exit price'},                       'TradeStation'),
            ({'ticker symbol', 'date opened'},                   'Robinhood'),
            ({'net p/l', 'underlying'},                          'Tastytrade'),
            ({'acquisition date', 'fund name'},                  'Vanguard'),
            ({'adjusted cost basis'},                            'E*TRADE'),
            ({'open time', 'close time'},                        'Webull'),
        ]
        for sig_cols, broker_name in col_broker_map:
            if sig_cols & cols:
                parser = BROKER_PARSER_MAP.get(broker_name)
                if parser:
                    file_data.seek(0)
                    return parser.parse(file_data, filename), broker_name

        raise ValueError("Broker no reconocido. Se usará el parser universal.")

