import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

# Configure page
st.set_page_config(page_title="1099-B to 8949 Converter", layout="wide")
st.title("📄 1099-B to IRS Form 8949 Converter")

st.markdown("Extrae transacciones de PDF 1099-B y genera Formulario 8949 con descarga de TXT y CSV.")

# Blank template download
blank = pd.DataFrame(columns=[
    'Description','Date Acquired','Date Sold',
    'Proceeds','Cost Basis','Code','Adjustment Amount','Gain or (loss)'
])
csv_blank = blank.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download Blank Form 8949 Template CSV",
    data=csv_blank,
    file_name="form_8949_template.csv",
    mime="text/csv",
    key="download_blank"
)

# File uploader and pages input
uploaded_file = st.file_uploader("Upload 1099-B PDF", type=["pdf"], key="uploader")
pages_input = st.text_input("Pages (e.g. 1,2-3). Leave blank = all.", key="pages_input")

# Regex patterns
desc_re = re.compile(r"^([A-Z]+ .*CALL .*|[A-Z]+ .*PUT .*)$")
data_re = re.compile(
    r"^(\d{2}/\d{2}/\d{2})\s+([\d\.]+)\s+(-?[\d\.]+)\s+(\d{2}/\d{2}/\d{2})\s+(-?[\d\.]+).*?\.\.\.\s+(-?[\d\.]+)"
)

# Parse pages
@st.cache_data
def parse_pages(text: str):
    if not text:
        return None
    parts = [p.strip() for p in text.split(',')]
    pages = []
    for part in parts:
        if '-' in part:
            a, b = part.split('-')
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))

# Process button triggers extraction once
def extract_records(pdf_bytes, pages):
    records = []
    last_desc = ''
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            if pages and idx not in pages:
                continue
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

# When Process button clicked
def process_action():
    pdf_bytes = st.session_state.uploader.read()
    pages = parse_pages(st.session_state.pages_input)
    records = extract_records(pdf_bytes, pages)
    if not records:
        st.error("No transactions detected.")
        return
    df = pd.DataFrame(records)
    df['Date Acquired'] = pd.to_datetime(df['Date Acquired'], errors='coerce').dt.strftime('%Y-%m-%d')
    df['Date Sold'] = pd.to_datetime(df['Date Sold'], errors='coerce').dt.strftime('%Y-%m-%d')
    df['Gain or (loss)'] = df['Gain or (loss)'].astype(float)
    st.session_state['df_out'] = df

if uploaded_file:
    st.button("Process PDF", on_click=process_action, key="process_btn")

# Display results and download
if 'df_out' in st.session_state:
    df = st.session_state.df_out
    st.subheader("Extracted Transactions")
    st.dataframe(df)
    # Download text file
    header = "Form 8949\nDepartment of the Treasury...\n"
    combined = header + df.to_csv(index=False)
    st.download_button(
        "Download Form 8949 Text File",
        data=combined.encode('utf-8'),
        file_name="form_8949.txt",
        mime="text/plain",
        key="download_txt"
    )
    # Download CSV file
    df_csv = df[[
        'Description','Date Acquired','Date Sold',
        'Proceeds','Cost Basis','Code','Adjustment Amount','Gain or (loss)'
    ]]
    csv_out = df_csv.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Form 8949 CSV",
        data=csv_out,
        file_name="form_8949.csv",
        mime="text/csv",
        key="download_csv"
    )
else:
    st.info("Upload a PDF and click ‘Process PDF’ to extract transactions.")

