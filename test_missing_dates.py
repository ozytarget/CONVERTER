"""
Script de prueba para verificar el relleno inteligente de fechas faltantes
"""

import pandas as pd
from io import StringIO, BytesIO
from universal_parser import UniversalBrokerParser

# Crear un archivo CSV de prueba con fechas faltantes
test_data = """Description,Quantity,Date Acquired,Date Sold,Proceeds,Cost Basis,Gain or Loss
SPXU 01/07,1,########,11/9/2021,174.98,190,-15.02
SPXU 01/07,1,########,########,101.98,117,-15.02
SPXU 01/07,1,12/10/2021,12/14/2021,94.98,64,30.98
SPXU 01/07,1,########,########,49.98,49,0.98
SQOO 01/07,1,########,########,122.98,125,-2.02
TWTR 09/10,1,8/18/2021,8/23/2021,451.98,403,48.98"""

print("="*70)
print("PRUEBA: RELLENO INTELIGENTE DE FECHAS FALTANTES")
print("="*70)

# Convertir string a BytesIO
csv_bytes = test_data.encode('utf-8')
file_data = BytesIO(csv_bytes)

# Parsear con el parser universal
df, warnings = UniversalBrokerParser.parse(file_data, "test_dates.csv")

print("\nResultados del parsing:")
print(f"Total de transacciones: {len(df)}")

print("\n" + "="*70)
print("DATOS PROCESADOS:")
print("="*70)

# Mostrar las transacciones SPXU (que tenían fechas faltantes)
spxu_data = df[df['Description'].str.contains('SPXU', na=False)]
print("\nTransacciones SPXU (tenían fechas faltantes):")
print(spxu_data[['Description', 'Date Acquired', 'Date Sold', 'Proceeds', 'Cost Basis']].to_string())

print("\nTransacciones SQOO (tenían ambas fechas faltantes):")
sqoo_data = df[df['Description'].str.contains('SQOO', na=False)]
print(sqoo_data[['Description', 'Date Acquired', 'Date Sold', 'Proceeds', 'Cost Basis']].to_string())

print("\n" + "="*70)
print("VERIFICACION:")
print("="*70)
print("¿Todas las fechas están rellenas?")
for idx, row in df.iterrows():
    if row['Date Acquired'] == 'Invalid Date' or row['Date Acquired'] == 'Various':
        print(f"  Fila {idx}: Date Acquired = {row['Date Acquired']} (no rellenada)")
    if row['Date Sold'] == 'Invalid Date' or row['Date Sold'] == 'Various':
        print(f"  Fila {idx}: Date Sold = {row['Date Sold']} (no rellenada)")

print("\n✅ Si no hay mensajes arriba, todas las fechas fueron rellenadas exitosamente!")

if warnings:
    print(f"\nAdvertencias encontradas: {len(warnings)}")
    for w in warnings:
        print(f"  - {w}")
