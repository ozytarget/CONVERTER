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

# --- CONSTANTES DE RENDIMIENTO ---
BATCH_UPDATE_SIZE = 10

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="PDF to 8949 Suite | PRO-TAX|", layout="wide")
st.title("📄 PDF CONVERTER PRO-TAX")


# --- INICIALIZACIÓN, REGEX Y FUNCIONES ---
if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False
if 'analysis_in_progress' not in st.session_state: st.session_state.analysis_in_progress = False
if 'pdf_info' not in st.session_state: st.session_state.pdf_info = None
if 'final_df' not in st.session_state: st.session_state.final_df = pd.DataFrame()
if 'raw_lines_df' not in st.session_state: st.session_state.raw_lines_df = pd.DataFrame()
if 'audit_log_df' not in st.session_state: st.session_state.audit_log_df = pd.DataFrame()

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

# --- LÓGICA PRINCIPAL DE LA APLICACIÓN ---
st.header("1. Cargar Archivo")
uploaded_file = st.file_uploader("Sube tu PDF para comenzar", type=["pdf"], key="main_uploader")

if uploaded_file:
    if not st.session_state.pdf_info or uploaded_file.file_id != st.session_state.pdf_info['id']:
        st.session_state.pdf_info = {'name': uploaded_file.name, 'bytes': uploaded_file.getvalue(), 'id': uploaded_file.file_id}
        st.session_state.processing_complete = False
        st.session_state.analysis_in_progress = False
        st.session_state.raw_lines_df, st.session_state.final_df, st.session_state.audit_log_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        st.cache_data.clear()
        st.rerun()

if st.session_state.pdf_info:
    st.success(f"Archivo cargado: **{st.session_state.pdf_info['name']}**")
    st.markdown("---")
    
    st.header("2. Iniciar Análisis")
    force_ocr_checkbox = st.checkbox("Forzar escaneo profundo (OCR) en todas las páginas", help="Útil para PDFs escaneados. Es mucho más lento.")
    
    if st.button("🚀 Iniciar Análisis Completo", type="primary"):
        st.session_state.processing_complete = False
        st.session_state.analysis_in_progress = True
        st.rerun() # Forzamos un rerun para que el expander se abra antes de empezar el bucle largo

    # Este bloque solo se ejecuta si el análisis ha sido iniciado
    if st.session_state.analysis_in_progress and not st.session_state.processing_complete:
        pdf_bytes = st.session_state.pdf_info['bytes']
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf: total_pages = len(pdf.pages)
        st.info(f"Iniciando análisis para **{st.session_state.pdf_info['name']}** ({total_pages} páginas)...")
        progress_bar, status_placeholder = st.progress(0), st.empty()
        
        # El expander ahora se abre dinámicamente
        with st.expander("🔬 Monitor", expanded=st.session_state.analysis_in_progress):
            log_placeholder = st.empty()
            log_placeholder.info("El escaneo en vivo aparecerá aquí...")
        
        all_potential_lines, audit_log = [], []
        start_time = time.time()
        
        for i in range(total_pages):
            current_page_num = i + 1
            page_lines, ocr_used, raw_text = process_single_page_v43(pdf_bytes, current_page_num, force_ocr_checkbox)
            all_potential_lines.extend(page_lines)
            audit_log.append({'Page_Number': current_page_num, 'OCR_Used': ocr_used, 'Extracted_Text': raw_text, 'Scanner_Decisions': format_decisions_for_log(page_lines)})
            
            if current_page_num % BATCH_UPDATE_SIZE == 0 or current_page_num == total_pages:
                progress_percentage = current_page_num / total_pages; progress_bar.progress(progress_percentage)
                elapsed_time = time.time() - start_time; eta = (elapsed_time / current_page_num) * (total_pages - current_page_num)
                status_placeholder.markdown(f'<div style="padding: 10px; border-radius: 5px; background-color: #262730;"><b>Estado:</b> Procesando...<br><b>Páginas:</b> {current_page_num} / {total_pages} | <b>ETA:</b> {format_time(eta)}</div>', unsafe_allow_html=True)
            
            with log_placeholder.container():
                st.subheader(f"Página {current_page_num} de {total_pages}")
                if ocr_used: st.warning(f"  - Se utilizó OCR en esta página.")
                st.markdown("**Texto Crudo Extraído:**"); st.code(raw_text, language='text')
                st.markdown("**Decisiones del Escáner:**")
                if not page_lines: st.write("  - No se encontraron coincidencias.")
                for line in page_lines: st.success(f"  - **{line['type']}**: `{line['content']}`")
        
        status_placeholder.empty(); progress_bar.empty()
        log_placeholder.success("✅ Escaneo en vivo completado. Revisa el Log de Auditoría descargable para un resumen completo.")
        
        st.session_state.audit_log_df = pd.DataFrame(audit_log)
        st.session_state.raw_lines_df = pd.DataFrame(all_potential_lines)
        st.session_state.final_df = assemble_records(st.session_state.raw_lines_df)
        st.session_state.analysis_in_progress = False # Terminamos el análisis
        st.session_state.processing_complete = True
        st.rerun()

# --- SECCIÓN 3: RESULTADOS Y DESCARGAS ---
if st.session_state.processing_complete:
    st.markdown("---")
    st.header("3. Resultados y Descargas")
    df = st.session_state.final_df
    base_filename = st.session_state.pdf_info['name'].rsplit('.', 1)[0]
    
    if not df.empty:
        # Cálculo vectorizado eficiente
        total_gain_loss = df['Gain or (loss)'].sum()
        st.success(f"¡Proceso exitoso! Se ensamblaron **{len(df)}** transacciones.")
        st.metric(label="Ganancia / (Pérdida) Neta Total", value=f"${total_gain_loss:,.2f}")
        st.dataframe(df)
        
        st.subheader("⬇️ Opciones de Descarga de Transacciones")
        # Crear dataframe 8949 de forma más eficiente
        df_8949 = df[['Description', 'Date Acquired', 'Date Sold', 'Proceeds', 'Cost Basis', 'Gain or (loss)']].copy()
        df_8949.insert(5, '(1f) Code(s) from instructions', '')
        df_8949.insert(6, '(1g) Amount of adjustment', '')
        
        csv_8949_output = df_8949.to_csv(index=False).encode('utf-8')
        
        # Generar Excel una sola vez
        excel_output = BytesIO()
        with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Transactions')
        excel_output.seek(0)
        
        txt_output = df.to_csv(index=False, sep='\t').encode('utf-8')
        
        col1, col2, col3 = st.columns(3)
        with col1: st.download_button("📥 Formato 8949 (.csv)", csv_8949_output, f"{base_filename}_Form8949.csv", "text/csv", key="csv_dl")
        with col2: st.download_button("📄 Excel (.xlsx)", excel_output.getvalue(), f"{base_filename}_Transactions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="xlsx_dl")
        with col3: st.download_button("📝 Texto Plano (.txt)", txt_output, f"{base_filename}_Transactions.txt", "text/plain", key="txt_dl")
    else:
        st.warning("No se ensamblaron transacciones válidas.")

    st.subheader("🔎 Auditoría y Avanzadas")
    col_audit1, col_audit2 = st.columns(2)
    with col_audit1:
        if not st.session_state.audit_log_df.empty:
            audit_log_excel_output = BytesIO()
            with pd.ExcelWriter(audit_log_excel_output, engine='openpyxl') as writer: st.session_state.audit_log_df.to_excel(writer, index=False, sheet_name='Audit_Log')
            st.download_button("📥 Descargar Log de Transparencia (Auditoría)", audit_log_excel_output.getvalue(), f"{base_filename}_Audit_Log.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", help="Guarda un registro detallado del escaneo página por página.", key="audit_log_dl")
    with col_audit2:
        if not st.session_state.raw_lines_df.empty:
            reprocessable_log_output = BytesIO()
            with pd.ExcelWriter(reprocessable_log_output, engine='openpyxl') as writer: st.session_state.raw_lines_df.to_excel(writer, index=False, sheet_name='Reprocessable_Log')
            st.download_button("⚙️ Descargar A Ecxel y Descar Nuevamente a TAX-8949", reprocessable_log_output.getvalue(), f"{base_filename}_Reprocessable_Log.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", help="Guarda solo las líneas encontradas para un futuro re-ensamblaje.", key="repro_log_dl")
