"""
Script de prueba para verificar el parser universal con el archivo de Robinhood
"""

import pandas as pd
from universal_parser import UniversalBrokerParser
from io import BytesIO

# Leer el archivo
file_path = r"c:\Users\urbin\OneDrive\Documents\RH2021 EXPORT ECXEL.csv"

print("=" * 80)
print("PROBANDO PARSER UNIVERSAL CON ARCHIVO ROBINHOOD")
print("=" * 80)

try:
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    
    file_data = BytesIO(file_bytes)
    df = UniversalBrokerParser.parse(file_data, "RH2021 EXPORT ECXEL.csv")
    
    print(f"\n✅ Archivo parseado exitosamente")
    print(f"Total de transacciones: {len(df)}")
    print(f"\nColumnas detectadas: {list(df.columns)}")
    
    print("\n" + "=" * 80)
    print("PRIMERAS 5 TRANSACCIONES:")
    print("=" * 80)
    print(df.head().to_string())
    
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS:")
    print("=" * 80)
    print(f"Total Proceeds: ${df['Proceeds'].sum():,.2f}")
    print(f"Total Cost Basis: ${df['Cost Basis'].sum():,.2f}")
    print(f"Total Gain/Loss: ${df['Gain or (loss)'].sum():,.2f}")
    
    print("\n" + "=" * 80)
    print("VERIFICACIÓN DE CÁLCULOS:")
    print("=" * 80)
    calculated_gain_loss = df['Proceeds'] - df['Cost Basis']
    matches = (abs(df['Gain or (loss)'] - calculated_gain_loss) < 0.01).sum()
    print(f"Transacciones donde Gain/Loss = Proceeds - Cost Basis: {matches}/{len(df)}")
    
    if matches == len(df):
        print("✅ Todos los cálculos son correctos")
    else:
        print(f"⚠️ {len(df) - matches} transacciones tienen cálculos incorrectos")
        print("\nTransacciones con discrepancias:")
        discrepancies = df[abs(df['Gain or (loss)'] - calculated_gain_loss) >= 0.01]
        print(discrepancies[['Description', 'Proceeds', 'Cost Basis', 'Gain or (loss)']].to_string())
    
    print("\n" + "=" * 80)
    print("ÚLTIMAS 5 TRANSACCIONES:")
    print("=" * 80)
    print(df.tail().to_string())
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
