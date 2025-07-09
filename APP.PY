import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

# Configuración de la página
st.set_page_config(page_title="PDF to Form 8949 (Unlimited)", layout="wide")
st.title("📄 Convertidor PDF 1099-B → Formulario IRS 8949 (Páginas ilimitadas)")
st.markdown("Sube tu PDF de transacciones y convierte todo automáticamente al Formulario 8949.")

# Plantilla vacía descargable
blank = pd.DataFrame(columns=[
    'Description','Date Acquired','Date Sold',
    'Proceeds','Cost Basis','Code','Adjustment Amount','Gain or (loss)'
])
csv_blank = blank.to_csv(index=False).encode('utf-8')
st.download_button(
    "Descargar Plantilla CSV Vacía (8949)",
    data=csv_blank,
    file_name="form_8949_template.csv",
    mime="text/csv"
)

# Subida de archivo PDF
uploaded_file = st.file_uploader("📤 Subir PDF 1099-B", type=["pdf"])

# Regex para descripción y líneas de transacción
desc_re = re.compile(r"^([A-Z]+ .*CALL .*|[A-Z]+ .*PUT .*)$")
data_re = re.compile(
    r"^(\d{2}/\d{2}/\d{2})\s+([\d\.]+)\s+(-?[\d\.]+)\s+(\d{2}/\d{2}/\d{2})\s+(-?[\d\.]+).*?\.\.\.\s+(-?[\d\.]+)"
)

# Función para extraer las transacciones de TODO el PDF
def extract_all_records(pdf_bytes):
    records = []
    last_desc = ''
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ''
            for line in text.splitlines():
                line = line.strip()
                if desc_re.match(line):
                    last_desc = line
                    continue
                m = data_re.match(line)
                if m and last_desc:
                    dsold, qty, proceeds, dacq, cost, gain = m.groups()
                    records.append({
                        'Description': last_desc,
                        'Date Acquired': dacq,
                        'Date Sold': dsold,
                        'Proceeds': proceeds,
                        'Cost Basis': cost,
                        'Code': '',
                        'Adjustment Amount': '',
                        'Gain or (loss)': gain
                    })
                    last_desc = ''
    return records

# Acción al presionar botón
if uploaded_file:
    pdf_bytes = uploaded_file.read()
    with st.spinner("Procesando todo el PDF..."):
        records = extract_all_records(pdf_bytes)
        if not records:
            st.error("⚠️ No se detectaron transacciones en el PDF.")
        else:
            df = pd.DataFrame(records)
            df['Date Acquired'] = pd.to_datetime(df['Date Acquired'], errors='coerce').dt.strftime('%Y-%m-%d')
            df['Date Sold'] = pd.to_datetime(df['Date Sold'], errors='coerce').dt.strftime('%Y-%m-%d')
            df['Gain or (loss)'] = df['Gain or (loss)'].astype(float)

            st.success(f"✅ {len(df)} transacciones extraídas de {len(pdfplumber.open(BytesIO(pdf_bytes)).pages)} páginas.")
            st.dataframe(df)

            # Texto plano IRS
            txt_header = "Form 8949 - Department of the Treasury - Internal Revenue Service\n"
            txt_content = txt_header + df.to_csv(index=False)
            st.download_button(
                "📥 Descargar .TXT para IRS",
                data=txt_content.encode('utf-8'),
                file_name="form_8949.txt",
                mime="text/plain"
            )

            # CSV para importar/guardar
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar CSV 8949",
                data=csv_output,
                file_name="form_8949.csv",
                mime="text/csv"
            )
