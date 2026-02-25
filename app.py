import streamlit as st
import pandas as pd
import pdfplumber          # type: ignore[import-untyped]
import re
from io import BytesIO
import pytesseract         # type: ignore[import-untyped]
from pdf2image import convert_from_bytes  # type: ignore[import-untyped]
import time
from functools import lru_cache
import numpy as np
from broker_parsers import BrokerDetector
from universal_parser import UniversalBrokerParser

# --- CONSTANTES DE RENDIMIENTO ---
BATCH_UPDATE_SIZE = 10

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="PDF to 8949 Suite | PRO-TAX|", layout="wide")
st.title("📄 PDF CONVERTER PRO>>TAX")


# --- INICIALIZACIÓN, REGEX Y FUNCIONES ---
if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False
if 'analysis_in_progress' not in st.session_state: st.session_state.analysis_in_progress = False
if 'file_info' not in st.session_state: st.session_state.file_info = None
if 'final_df' not in st.session_state: st.session_state.final_df = pd.DataFrame()
if 'raw_lines_df' not in st.session_state: st.session_state.raw_lines_df = pd.DataFrame()
if 'audit_log_df' not in st.session_state: st.session_state.audit_log_df = pd.DataFrame()
if 'file_type' not in st.session_state: st.session_state.file_type = None
if 'broker_detected' not in st.session_state: st.session_state.broker_detected = None

# ── Regex para cazar líneas de DESCRIPCIÓN en PDFs 1099-B ─────────────────
# Detecta líneas que terminan con CUSIP / Symbol / ISIN  OR  que comienzan
# con un ticker corto seguido de texto largo (descripción del instrumento).
# Excluye líneas que empiezan con fecha o número (son datos, no cabeceras).
DESC_RE = re.compile(
    r'(?i)(?<!\d)'
    r'(cusip[:\s]|symbol[:\s]|isin[:\s]|'
    r'(?:^[A-Z]{1,6}(?:[/ ][A-Z0-9]{1,10}){0,3}\s+.{10,}))'
    r'|(?:CUSIP|ISIN|Stock Description)',
    re.MULTILINE
)

# ── Colección de patrones para extraer filas de datos de distintos formatos
# de PDF de brokers.  Se prueban en orden y se toma el primero que coincida.
PDF_PATTERNS = [
    # Patrón 1: date_sold qty proceeds date_acquired cost_basis [adj] gain_loss
    re.compile(
        r'^(?P<date_sold>\d{1,2}/\d{1,2}/\d{2,4})\s+'
        r'(?P<quantity>[\d,\.]+)\s+'
        r'(?P<proceeds>-?[\d,]+\.\d{2})\s+'
        r'(?P<date_acquired>Various|\d{1,2}/\d{1,2}/\d{2,4})\s+'
        r'(?P<cost_basis>-?[\d,]+\.\d{2})\s+'
        r'(?:\.{2,}\s+)?(?P<gain_loss>-?[\d,]+\.\d{2})',
        re.IGNORECASE),
    # Patrón 2: symbol – date_acquired – date_sold – qty – proceeds – cost – gain
    re.compile(
        r'^(?P<description>[A-Z]{1,6})\s+'
        r'(?P<date_acquired>Various|\d{1,2}/\d{1,2}/\d{2,4})\s+'
        r'(?P<date_sold>\d{1,2}/\d{1,2}/\d{2,4})\s+'
        r'(?P<quantity>[\d,\.]+)\s+'
        r'(?P<proceeds>-?[\d,]+\.\d{2})\s+'
        r'(?P<cost_basis>-?[\d,]+\.\d{2})\s+'
        r'(?P<gain_loss>-?[\d,]+\.\d{2})',
        re.IGNORECASE),
    # Patrón 3: date_acquired – date_sold – description – qty – proceeds – cost – gain
    re.compile(
        r'^(?P<date_acquired>Various|\d{1,2}/\d{1,2}/\d{2,4})\s+'
        r'(?P<date_sold>\d{1,2}/\d{1,2}/\d{2,4})\s+'
        r'(?P<description>[\w\s/\-\.]{3,40}?)\s+'
        r'(?P<quantity>[\d,\.]+)\s+'
        r'(?P<proceeds>-?[\d,]+\.\d{2})\s+'
        r'(?P<cost_basis>-?[\d,]+\.\d{2})\s+'
        r'(?P<gain_loss>-?[\d,]+\.\d{2})',
        re.IGNORECASE),
    # Patrón 4: Schwab / Fidelity 1099-B estilo tabla
    # date_sold – description – date_acquired – qty – proceeds – cost – gain
    re.compile(
        r'^(?P<date_sold>\d{1,2}/\d{1,2}/\d{2,4})\s+'
        r'(?P<description>[A-Z][\w\s/\-\.]{2,35})\s+'
        r'(?P<date_acquired>Various|\d{1,2}/\d{1,2}/\d{2,4})\s+'
        r'(?P<quantity>[\d,\.]+)\s+'
        r'(?P<proceeds>-?[\d,]+\.\d{2})\s+'
        r'(?P<cost_basis>-?[\d,]+\.\d{2})\s+'
        r'(?P<gain_loss>-?[\d,]+\.\d{2})',
        re.IGNORECASE),
    # Patrón 5: Solo montos (sin descripción inline) – fallback ligero
    re.compile(
        r'^(?P<date_sold>\d{1,2}/\d{1,2}/\d{2,4})\s+'
        r'(?P<date_acquired>Various|\d{1,2}/\d{1,2}/\d{2,4})\s+'
        r'(?P<proceeds>-?[\d,]+\.\d{2})\s+'
        r'(?P<cost_basis>-?[\d,]+\.\d{2})\s+'
        r'(?P<gain_loss>-?[\d,]+\.\d{2})',
        re.IGNORECASE),
]
# Para compatibilidad con código que referencia DATA_RE_UNIVERSAL
DATA_RE_UNIVERSAL = PDF_PATTERNS[0]

@lru_cache(maxsize=256)
def clean_value_cached(value_str):
    """Cachea valores limpios frecuentes"""
    try:
        return float(value_str.replace(',', '').replace('$', '').strip())
    except:
        return 0.0

def clean_value(value):
    """Limpia valores numéricos eficientemente"""
    if not value: return 0.0
    return clean_value_cached(str(value))

@lru_cache(maxsize=256)
def parse_date_cached(date_str):
    """Cachea fechas parseadas comunes"""
    if date_str is None or 'various' in date_str.lower(): return 'Various'
    try:
        return pd.to_datetime(date_str, errors='coerce').strftime('%m/%d/%Y')
    except:
        return 'Invalid Date'

def parse_date(date_str):
    return parse_date_cached(date_str)

def format_time(seconds):
    if seconds < 60: return f"{int(seconds)} seg"
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes} min {seconds} seg"

def format_decisions_for_log(page_lines):
    if not page_lines: return "No se encontraron coincidencias."
    return "\n".join([f"{line['type']}: {line['content']}" for line in page_lines])


def match_pdf_data_line(line_text: str):
    """Prueba todos los patrones PDF y devuelve (match, pattern_idx) o (None, -1)."""
    for i, pattern in enumerate(PDF_PATTERNS):
        m = pattern.match(line_text)
        if m:
            return m, i
    return None, -1

# --- PROCESAMIENTO DE BROKERS CSV/EXCEL ---
def process_broker_file(file_data: BytesIO, filename: str):
    """Procesa archivos CSV/Excel de brokers usando parser universal"""
    try:
        # Primero intentar detección específica de broker conocido
        df, broker_name = None, None
        warnings = []
        
        try:
            df, broker_name = BrokerDetector.detect_and_parse(file_data, filename)
            return df, broker_name, None, warnings
        except:
            pass
        
        # Si no funciona, usar parser universal
        file_data.seek(0)
        result = UniversalBrokerParser.parse(file_data, filename)
        
        # El parser retorna tupla (df, warnings) o solo df dependiendo de la versión
        if isinstance(result, tuple):
            df, warnings = result
        else:
            df = result
            warnings = []
        
        return df, 'universal', None, warnings
        
    except Exception as e:
        return None, None, str(e), []

# --- PROCESAMIENTO DE PDFs ---
@st.cache_data(show_spinner=False)
def process_single_page_v43(_pdf_bytes, page_number, force_ocr):
    """Optimizado para mejor eficiencia de memoria y velocidad"""
    ocr_was_used, page_potential_lines, text = False, [], ""
    
    try:
        with pdfplumber.open(BytesIO(_pdf_bytes)) as pdf:
            page = pdf.pages[page_number - 1]
            text = page.extract_text(x_tolerance=1, y_tolerance=3, layout=True) or ""
    except Exception as e:
        text = f"[Error extrayendo texto: {str(e)}]"
    
    # Si el texto extraído es muy corto o se fuerza OCR, usar OCR
    if force_ocr or len(text) < 150:
        ocr_was_used = True
        try:
            images = convert_from_bytes(_pdf_bytes, dpi=300, first_page=page_number, last_page=page_number)
            if images:
                text = pytesseract.image_to_string(images[0])
        except Exception as e:
            text = f"[Error durante el OCR: {str(e)}]"
    
    # Procesar líneas con múltiples patrones
    lines = text.split('\n')
    for line_text in lines:
        clean_line = line_text.strip()
        if not clean_line or len(clean_line) < 6:
            continue
        # No procesar como DESC si empieza con dígito/fecha (es fila de datos)
        starts_with_digit = bool(re.match(r'^\d', clean_line))
        if not starts_with_digit and DESC_RE.search(clean_line):
            page_potential_lines.append({'type': 'DESC', 'content': f"P.{page_number}: {clean_line}"})
        else:
            m, _ = match_pdf_data_line(clean_line)
            if m:
                # Guardar solo el texto; re-matcheamos al ensamblar para evitar
                # problemas de serialización de objetos regex en DataFrames.
                page_potential_lines.append({'type': 'DATA', 'content': clean_line})
    
    return page_potential_lines, ocr_was_used, text

def _holding_period(date_acquired: str, date_sold: str) -> str:
    """Determina si una posicón es Short-term (≤12 meses) o Long-term (>12 meses)."""
    if date_acquired in ('Various', 'Invalid Date', '') or date_sold in ('Invalid Date', ''):
        return 'Unknown'
    try:
        dt_acq = pd.to_datetime(date_acquired, errors='coerce')
        dt_sold = pd.to_datetime(date_sold, errors='coerce')
        if pd.isna(dt_acq) or pd.isna(dt_sold):
            return 'Unknown'
        days = (dt_sold - dt_acq).days
        return 'Long-term' if days > 365 else 'Short-term'
    except Exception:
        return 'Unknown'


def assemble_records(lines_df):
    """
    Ensambla registros usando los múltiples patrones PDF.
    SIEMPRE re-matchea desde el texto (los objetos regex no sobreviven en DataFrame).
    """
    if lines_df.empty:
        return pd.DataFrame()

    final_records = []
    last_seen_desc = None

    for _, row in lines_df.iterrows():
        if row['type'] == 'DESC':
            last_seen_desc = row['content']
        elif row['type'] == 'DATA':
            # Re-match siempre desde el string guardado
            m, _ = match_pdf_data_line(str(row['content']))
            if m is None:
                continue

            data = m.groupdict()
            # Descripción: inline en el patrón (2/3/4) o de la línea DESC previa
            desc = data.get('description') or (last_seen_desc or '')
            desc = ' '.join(str(desc).split())
            if not desc:
                continue

            date_acq  = parse_date(data.get('date_acquired', 'Various'))
            date_sold = parse_date(data.get('date_sold', ''))
            proceeds  = clean_value(data.get('proceeds', '0'))
            cost      = clean_value(data.get('cost_basis', '0'))
            gain      = clean_value(data.get('gain_loss', '0'))

            # Recalcular gain/loss si no estaba en el patrón o es cero
            if gain == 0.0:
                gain = proceeds - cost

            final_records.append({
                'Description': desc,
                'Date Acquired': date_acq,
                'Date Sold': date_sold,
                'Proceeds': proceeds,
                'Cost Basis': cost,
                'Gain or (loss)': gain,
                'Term': _holding_period(date_acq, date_sold),
            })
            # Limpiar desc contextual solo cuando vino de línea previa
            if data.get('description') is None:
                last_seen_desc = None

    if not final_records:
        return pd.DataFrame()
    return pd.DataFrame(final_records)

IRS_8949_COLUMNS = [
    '(a) Description of Property',
    '(b) Date Acquired',
    '(c) Date Sold or Disposed',
    '(d) Proceeds (Sales Price)',
    '(e) Cost or Other Basis',
    '(f) Code(s)',
    '(g) Adjustment Amount',
    '(h) Gain or (Loss)',
]

def generate_8949_output(df: pd.DataFrame) -> tuple:
    """
    Genera DataFrames separados para IRS Form 8949:
      - Part I  : transacciones Short-term (≤12 meses)
      - Part II : transacciones Long-term  (>12 meses)
    Retorna (part1_df, part2_df, combined_df)
    """
    if df.empty:
        empty = pd.DataFrame(columns=IRS_8949_COLUMNS)
        return empty, empty, empty

    df = df.copy()

    # Calcular Term si no existe
    if 'Term' not in df.columns:
        df['Term'] = df.apply(
            lambda r: _holding_period(r.get('Date Acquired', ''), r.get('Date Sold', '')), axis=1
        )

    # Asegurar Gain or (loss)
    if 'Gain or (loss)' not in df.columns:
        df['Gain or (loss)'] = df.get('Proceeds', pd.Series(0)) - df.get('Cost Basis', pd.Series(0))

    def _to_8949(subset: pd.DataFrame) -> pd.DataFrame:
        def _col(name: str, default) -> pd.Series:
            return subset[name] if name in subset.columns else pd.Series([default] * len(subset), dtype=object)
        out = pd.DataFrame(index=subset.index)
        out['(a) Description of Property']   = _col('Description', '')
        out['(b) Date Acquired']              = _col('Date Acquired', 'Various')
        out['(c) Date Sold or Disposed']      = _col('Date Sold', '')
        out['(d) Proceeds (Sales Price)']     = pd.to_numeric(_col('Proceeds', 0), errors='coerce').fillna(0)
        out['(e) Cost or Other Basis']        = pd.to_numeric(_col('Cost Basis', 0), errors='coerce').fillna(0)
        out['(f) Code(s)']                    = ''
        out['(g) Adjustment Amount']          = 0.0
        out['(h) Gain or (Loss)']             = pd.to_numeric(_col('Gain or (loss)', 0), errors='coerce').fillna(0)
        return out.reset_index(drop=True)

    short = df[df['Term'].isin(['Short-term', 'Unknown'])]
    long_ = df[df['Term'] == 'Long-term']

    part1  = _to_8949(short)
    part2  = _to_8949(long_)
    combined = _to_8949(df)
    return part1, part2, combined

# --- LÓGICA PRINCIPAL DE LA APLICACIÓN ---
st.header("1️⃣ Cargar Archivo")

# Pestañas para diferentes tipos de entrada
tab1, tab2 = st.tabs(["📊 CSV/Excel de Broker", "📄 PDF de Broker"])

with tab1:
    st.subheader("Carga archivos CSV o Excel de tu broker")
    st.markdown("""
    **Brokers reconocidos automáticamente:**

    | Broker | Formatos |
    |---|---|
    | Interactive Brokers | CSV, Flex Query |
    | TD Ameritrade / thinkorswim | CSV |
    | Charles Schwab | CSV, Excel |
    | Fidelity | CSV, Excel |
    | Robinhood | CSV |
    | Webull | CSV |
    | E*TRADE | CSV |
    | TradeStation | CSV |
    | Tastytrade | CSV |
    | Merrill Edge | CSV, Excel |
    | Vanguard | CSV, Excel |
    | Apex Clearing | CSV |

    **¿Tu broker no está en la lista?**
    El *parser universal* detecta cualquier formato automáticamente.
    """)
    
    uploaded_broker_file = st.file_uploader(
        "Sube tu archivo CSV o Excel",
        type=["csv", "xlsx", "xls"],
        key="broker_uploader"
    )
    
    if uploaded_broker_file:
        if not st.session_state.file_info or uploaded_broker_file.file_id != st.session_state.file_info.get('id'):
            file_bytes = uploaded_broker_file.getvalue()
            file_info = {
                'name': uploaded_broker_file.name,
                'bytes': file_bytes,
                'id': uploaded_broker_file.file_id,
                'type': 'broker'
            }
            st.session_state.file_info = file_info
            st.session_state.processing_complete = False
            st.session_state.final_df = pd.DataFrame()
            st.session_state.file_type = 'broker'
            st.rerun()
        
        if st.session_state.file_info and st.session_state.file_info.get('type') == 'broker':
            st.success(f"✅ Archivo cargado: **{st.session_state.file_info['name']}**")
            
            if st.button("🚀 Procesar Broker File", type="primary", key="process_broker"):
                st.session_state.processing_complete = False
                file_bytes = st.session_state.file_info['bytes']
                filename = st.session_state.file_info['name']
                
                with st.spinner("Procesando archivo..."):
                    file_data = BytesIO(file_bytes)
                    df, broker_name, error, warnings = process_broker_file(file_data, filename)
                    
                    if error or df is None:
                        st.error(f"❌ Error procesando archivo: {error or 'No se pudo procesar el archivo.'}")
                    else:
                        # Mostrar advertencias si las hay
                        if warnings:
                            st.warning("⚠️ Se detectaron inconsistencias en los datos:")
                            with st.expander("Ver detalles de advertencias"):
                                for w in warnings:
                                    st.write(w)

                        # Asegurar columna Term para clasificación
                        if 'Term' not in df.columns:
                            df['Term'] = df.apply(
                                lambda r: _holding_period(
                                    r.get('Date Acquired', ''), r.get('Date Sold', '')), axis=1)

                        # Asegurar todas las columnas necesarias
                        required_columns = ['Description', 'Date Acquired', 'Date Sold',
                                            'Proceeds', 'Cost Basis', 'Gain or (loss)']
                        for col in required_columns:
                            if col not in df.columns:
                                if col == 'Gain or (loss)' and 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
                                    df[col] = df['Proceeds'] - df['Cost Basis']
                                else:
                                    df[col] = ''

                        st.session_state.final_df = df
                        st.session_state.broker_detected = broker_name
                        st.session_state.processing_complete = True

                        if broker_name == 'universal':
                            st.success("✅ Procesado con **Parser Universal Inteligente** — columnas detectadas automáticamente")
                        else:
                            st.success(f"✅ Broker detectado: **{broker_name}** — archivo procesado correctamente")

                        st.rerun()

with tab2:
    st.subheader("Carga un PDF de tu broker (Formato 1099-B)")
    
    uploaded_pdf = st.file_uploader(
        "Sube tu PDF",
        type=["pdf"],
        key="pdf_uploader"
    )
    
    if uploaded_pdf:
        if not st.session_state.file_info or uploaded_pdf.file_id != st.session_state.file_info.get('id'):
            st.session_state.file_info = {
                'name': uploaded_pdf.name,
                'bytes': uploaded_pdf.getvalue(),
                'id': uploaded_pdf.file_id,
                'type': 'pdf'
            }
            st.session_state.processing_complete = False
            st.session_state.file_type = 'pdf'
            st.rerun()
        
        if st.session_state.file_info and st.session_state.file_info.get('type') == 'pdf':
            st.success(f"✅ Archivo cargado: **{st.session_state.file_info['name']}**")
            st.markdown("---")
            
            st.header("2️⃣ Configurar Análisis")
            force_ocr_checkbox = st.checkbox(
                "Forzar OCR en todas las páginas",
                help="Útil para PDFs escaneados. Es más lento pero más preciso."
            )
            
            if st.button("🚀 Iniciar Análisis", type="primary", key="start_pdf_analysis"):
                st.session_state.processing_complete = False
                st.session_state.analysis_in_progress = True
                st.rerun()
            
            if st.session_state.analysis_in_progress and not st.session_state.processing_complete:
                pdf_bytes = st.session_state.file_info['bytes']
                
                with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                    total_pages = len(pdf.pages)
                
                st.info(f"Analizando **{st.session_state.file_info['name']}** ({total_pages} páginas)...")
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                with st.expander("🔬 Detalles del Procesamiento", expanded=True):
                    log_placeholder = st.empty()
                
                all_potential_lines = []
                audit_log = []
                start_time = time.time()
                
                for i in range(total_pages):
                    current_page_num = i + 1
                    page_lines, ocr_used, raw_text = process_single_page_v43(
                        pdf_bytes, current_page_num, force_ocr_checkbox
                    )
                    all_potential_lines.extend(page_lines)
                    audit_log.append({
                        'Page': current_page_num,
                        'OCR_Used': ocr_used,
                        'Matches': len(page_lines),
                        'Summary': format_decisions_for_log(page_lines)
                    })
                    
                    if current_page_num % BATCH_UPDATE_SIZE == 0 or current_page_num == total_pages:
                        progress_pct = current_page_num / total_pages
                        progress_bar.progress(progress_pct)
                        elapsed = time.time() - start_time
                        eta = (elapsed / current_page_num) * (total_pages - current_page_num)
                        status_placeholder.info(
                            f"📊 Páginas: {current_page_num}/{total_pages} | ⏱️ ETA: {format_time(eta)}"
                        )
                
                progress_bar.empty()
                status_placeholder.empty()
                
                with log_placeholder.container():
                    st.success("✅ Escaneo completado")
                
                st.session_state.audit_log_df = pd.DataFrame(audit_log)
                st.session_state.raw_lines_df = pd.DataFrame(all_potential_lines)
                st.session_state.final_df = assemble_records(st.session_state.raw_lines_df)
                st.session_state.analysis_in_progress = False
                st.session_state.processing_complete = True
                st.rerun()

# --- SECCIÓN 3: RESULTADOS Y DESCARGAS ---
if st.session_state.processing_complete and not st.session_state.final_df.empty:
    st.markdown("---")
    st.header("3️⃣ Resultados y Descargas")
    
    df = st.session_state.final_df
    file_info = st.session_state.file_info or {}
    base_filename = str(file_info.get('name', 'output')).rsplit('.', 1)[0]
    
    # Estadísticas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Transacciones", len(df))
    with col2:
        # Calcular ganancia/pérdida si no existe
        if 'Gain or (loss)' in df.columns:
            total_gain_loss = df['Gain or (loss)'].sum()
        elif 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
            total_gain_loss = (df['Proceeds'] - df['Cost Basis']).sum()
        else:
            total_gain_loss = 0
        st.metric("Ganancia/(Pérdida) Total", f"${total_gain_loss:,.2f}")
    with col3:
        if 'Proceeds' in df.columns:
            total_proceeds = df['Proceeds'].sum()
        else:
            total_proceeds = 0
        st.metric("Ingresos Totales", f"${total_proceeds:,.2f}")
    
    # Generar Form 8949 con clasificación short/long term
    part1, part2, combined_8949 = generate_8949_output(df)

    # ── Tabla de transacciones con tabs Part I / Part II ──────────────────────
    st.subheader("📋 Transacciones Extraídas")
    tab_all, tab_short, tab_long = st.tabs([
        f"Todas ({len(df)})",
        f"Part I – Short-term ({len(part1)})",
        f"Part II – Long-term ({len(part2)})",
    ])
    with tab_all:
        st.dataframe(df, use_container_width=True)
    with tab_short:
        st.caption("Posiciones mantenidas ≤ 12 meses (IRS Form 8949, Part I)")
        if not part1.empty:
            st.dataframe(part1, use_container_width=True)
        else:
            st.info("No hay transacciones short-term.")
    with tab_long:
        st.caption("Posiciones mantenidas > 12 meses (IRS Form 8949, Part II)")
        if not part2.empty:
            st.dataframe(part2, use_container_width=True)
        else:
            st.info("No hay transacciones long-term.")

    # ── Descargas ─────────────────────────────────────────────────────────────
    st.subheader("⬇️ Descargar Resultados")

    csv_8949 = combined_8949.to_csv(index=False).encode('utf-8')
    csv_full  = df.to_csv(index=False).encode('utf-8')

    excel_output = BytesIO()
    with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='All_Transactions')
        combined_8949.to_excel(writer, index=False, sheet_name='Form_8949_Combined')
        if not part1.empty:
            part1.to_excel(writer, index=False, sheet_name='Part_I_ShortTerm')
        if not part2.empty:
            part2.to_excel(writer, index=False, sheet_name='Part_II_LongTerm')
    excel_output.seek(0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📥 Form 8949 CSV (combinado)",
            csv_8949,
            f"{base_filename}_Form8949.csv",
            "text/csv",
            key="csv_8949"
        )
    with col2:
        st.download_button(
            "📊 Excel Completo (Part I + Part II)",
            excel_output.getvalue(),
            f"{base_filename}_Form8949.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="xlsx"
        )
    with col3:
        st.download_button(
            "📄 CSV Completo (datos originales)",
            csv_full,
            f"{base_filename}_Complete.csv",
            "text/csv",
            key="csv_full"
        )
    
    # Auditoría
    if not st.session_state.audit_log_df.empty and st.session_state.file_type == 'pdf':
        st.subheader("🔍 Log de Auditoría")
        st.dataframe(st.session_state.audit_log_df, use_container_width=True)
        
        audit_excel = BytesIO()
        with pd.ExcelWriter(audit_excel, engine='openpyxl') as writer:
            st.session_state.audit_log_df.to_excel(writer, index=False)
        
        st.download_button(
            "📥 Descargar Log de Auditoría",
            audit_excel.getvalue(),
            f"{base_filename}_Audit.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="audit_dl"
        )

elif st.session_state.processing_complete and st.session_state.final_df.empty:
    st.warning("⚠️ No se encontraron transacciones válidas en el archivo.")
