import streamlit as st
import pandas as pd
import pdfplumber
import re
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes
import time
from functools import lru_cache
import numpy as np
from broker_parsers import BrokerDetector

# --- CONSTANTES DE RENDIMIENTO ---
BATCH_UPDATE_SIZE = 10

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="PDF to 8949 Suite | PRO-TAX|", layout="wide")
st.title("📄 PDF CONVERTER PRO-TAX")


# --- INICIALIZACIÓN, REGEX Y FUNCIONES ---
if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False
if 'analysis_in_progress' not in st.session_state: st.session_state.analysis_in_progress = False
if 'file_info' not in st.session_state: st.session_state.file_info = None
if 'final_df' not in st.session_state: st.session_state.final_df = pd.DataFrame()
if 'raw_lines_df' not in st.session_state: st.session_state.raw_lines_df = pd.DataFrame()
if 'audit_log_df' not in st.session_state: st.session_state.audit_log_df = pd.DataFrame()
if 'file_type' not in st.session_state: st.session_state.file_type = None
if 'broker_detected' not in st.session_state: st.session_state.broker_detected = None

# Compilar regex una sola vez (global)
DESC_RE = re.compile(r"/ CUSIP: / Symbol:", re.IGNORECASE)
DATA_RE_UNIVERSAL = re.compile(r"^(?P<date_sold>\d{1,2}/\d{1,2}/\d{2,4})\s+(?P<quantity>[\d\.]+)\s+(?P<proceeds>-?[\d,]+\.\d{2})\s+(?P<date_acquired>Various|\d{1,2}/\d{1,2}/\d{2,4})\s+(?P<cost_basis>-?[\d,]+\.\d{2})\s+(?:\.\.\.\s+)?(?P<gain_loss>-?[\d,]+\.\d{2})", re.IGNORECASE)

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

# --- PROCESAMIENTO DE BROKERS CSV/EXCEL ---
def process_broker_file(file_data: BytesIO, filename: str):
    """Procesa archivos CSV/Excel de brokers"""
    try:
        df, broker_name = BrokerDetector.detect_and_parse(file_data, filename)
        return df, broker_name, None
    except Exception as e:
        return None, None, str(e)

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
    
    # Procesar líneas eficientemente
    lines = text.split('\n')
    for line_text in lines:
        clean_line = line_text.strip()
        if not clean_line:  # Skip empty lines
            continue
        if DESC_RE.search(clean_line):
            page_potential_lines.append({'type': 'DESC', 'content': f"P.{page_number}: {clean_line}"})
        elif DATA_RE_UNIVERSAL.match(clean_line):
            page_potential_lines.append({'type': 'DATA', 'content': clean_line})
    
    return page_potential_lines, ocr_was_used, text

def assemble_records(lines_df):
    """Versión vectorizada y optimizada para mejor rendimiento"""
    if lines_df.empty:
        return pd.DataFrame()
    
    # Separa DESC y DATA de forma más eficiente
    desc_rows = lines_df[lines_df['type'] == 'DESC'].copy()
    data_rows = lines_df[lines_df['type'] == 'DATA'].copy()
    
    if desc_rows.empty or data_rows.empty:
        return pd.DataFrame()
    
    final_records = []
    last_seen_desc = None
    
    for idx, row in lines_df.iterrows():
        if row['type'] == 'DESC':
            last_seen_desc = row['content']
        elif row['type'] == 'DATA' and last_seen_desc is not None:
            data_match = DATA_RE_UNIVERSAL.match(row['content'])
            if data_match:
                data = data_match.groupdict()
                final_records.append({
                    'Description': ' '.join(last_seen_desc.split()),
                    'Date Acquired': parse_date(data['date_acquired']),
                    'Date Sold': parse_date(data['date_sold']),
                    'Proceeds': clean_value(data['proceeds']),
                    'Cost Basis': clean_value(data['cost_basis']),
                    'Gain or (loss)': clean_value(data['gain_loss']),
                })
                last_seen_desc = None
    
    return pd.DataFrame(final_records)

def generate_8949_output(df):
    """Genera DataFrame con columnas de Formulario 8949"""
    if df.empty:
        return pd.DataFrame()
    
    # Asegurar que todas las columnas necesarias existen
    required_cols = ['Description', 'Date Acquired', 'Date Sold', 'Proceeds', 'Cost Basis', 'Gain or (loss)']
    
    # Verificar que existen, sino agregar con valores por defecto
    for col in required_cols:
        if col not in df.columns:
            if col == 'Gain or (loss)' and 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
                df[col] = df['Proceeds'] - df['Cost Basis']
            else:
                df[col] = ''
    
    df_8949 = df[required_cols].copy()
    df_8949.insert(5, '(1f) Code(s) from instructions', '')
    df_8949.insert(6, '(1g) Amount of adjustment', '')
    return df_8949

# --- LÓGICA PRINCIPAL DE LA APLICACIÓN ---
st.header("1️⃣ Cargar Archivo")

# Pestañas para diferentes tipos de entrada
tab1, tab2 = st.tabs(["📊 CSV/Excel de Broker", "📄 PDF de Broker"])

with tab1:
    st.subheader("Carga archivos CSV o Excel de tu broker")
    st.markdown("""
    **Brokers soportados:**
    - Interactive Brokers
    - TD Ameritrade (thinkorswim)
    - Fidelity
    - Charles Schwab
    - TradeStation
    - Robinhood
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
                    df, broker_name, error = process_broker_file(file_data, filename)
                    
                    if error:
                        st.error(f"❌ Error procesando archivo: {error}")
                    else:
                        # Asegurar que el DataFrame tenga todas las columnas necesarias
                        required_columns = ['Description', 'Date Acquired', 'Date Sold', 'Proceeds', 'Cost Basis', 'Gain or (loss)']
                        for col in required_columns:
                            if col not in df.columns:
                                if col == 'Gain or (loss)' and 'Proceeds' in df.columns and 'Cost Basis' in df.columns:
                                    df[col] = df['Proceeds'] - df['Cost Basis']
                                else:
                                    df[col] = ''
                        
                        st.session_state.final_df = df
                        st.session_state.broker_detected = broker_name
                        st.session_state.processing_complete = True
                        st.success(f"✅ Archivo procesado exitosamente. Broker detectado: **{broker_name.replace('_', ' ').title()}**")
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
    base_filename = st.session_state.file_info['name'].rsplit('.', 1)[0]
    
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
    
    # Tabla de transacciones
    st.subheader("📋 Transacciones Extraídas")
    st.dataframe(df, use_container_width=True)
    
    # Descargas
    st.subheader("⬇️ Descargar Resultados")
    
    # Preparar outputs
    df_8949 = generate_8949_output(df)
    csv_8949 = df_8949.to_csv(index=False).encode('utf-8')
    
    csv_full = df.to_csv(index=False).encode('utf-8')
    
    excel_output = BytesIO()
    with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Transactions')
        df_8949.to_excel(writer, index=False, sheet_name='Form_8949')
    excel_output.seek(0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📥 Formato 8949 (CSV)",
            csv_8949,
            f"{base_filename}_Form8949.csv",
            "text/csv",
            key="csv_8949"
        )
    with col2:
        st.download_button(
            "📊 Excel Completo",
            excel_output.getvalue(),
            f"{base_filename}_Transactions.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="xlsx"
        )
    with col3:
        st.download_button(
            "📄 CSV Completo",
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
