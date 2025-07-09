# 1099-B to IRS Form 8949 Converter

Este proyecto es una aplicación **Streamlit** que extrae transacciones de un PDF de 1099-B y genera:

1. Un **CSV** con las columnas (a)–(h) listo para importar en el Formulario 8949.  
2. Un **archivo de texto** (`.txt`) con el encabezado completo del Formulario 8949 y tus transacciones.

---

## Cómo usarlo (local, paso a paso)

1. **Clona o descarga** este repositorio (puedes descargar el ZIP desde GitHub o, si más adelante usas Git, `git clone ...`).  
2. Abre una **consola** o **PowerShell** dentro de la carpeta donde guardaste estos archivos.  
3. Crea y activa un **entorno virtual** (recomendado):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
