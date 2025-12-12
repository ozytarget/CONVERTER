# Multi-Broker Support - TAX Converter

## 📊 Brokers Soportados

La aplicación ahora puede procesar archivos de los siguientes brokers:

### 1. **Interactive Brokers**
- **Formato:** CSV
- **Cómo obtener:** Reportes → Trades → Closed Positions Export
- **Columnas esperadas:** Open Date, Close Date, Quantity, T. Price, Proceeds, Basis, Realised P&L, Symbol

### 2. **TD Ameritrade (thinkorswim)**
- **Formato:** CSV
- **Cómo obtener:** Account Statement → Positions → Export to CSV
- **Columnas esperadas:** Open Date, Close Date, Qty, Proceeds, Basis, Gain/Loss, Symbol

### 3. **Fidelity**
- **Formato:** CSV o Excel
- **Cómo obtener:** Portfolio → Download as CSV/Excel
- **Columnas esperadas:** Open Date, Sell Date, Quantity, Price, Proceeds, Cost Basis, Gain/Loss, Symbol

### 4. **Charles Schwab**
- **Formato:** CSV
- **Cómo obtener:** Account → Positions → Export
- **Columnas esperadas:** Open Date, Close Date, Quantity, Price, Proceeds, Cost, Gain/Loss, Symbol

### 5. **TradeStation**
- **Formato:** CSV
- **Cómo obtener:** Tools → Trade Positions → Export
- **Columnas esperadas:** Entry Date, Exit Date, Qty, Exit Price, Proceeds, Entry Cost, P&L $, Symbol

## 🔄 Cómo Usar

### Con CSV/Excel
1. Descarga el reporte de tu broker
2. Ve a la pestaña "📊 CSV/Excel de Broker"
3. Sube el archivo
4. Haz clic en "Procesar"
5. Descarga el resultado en formato 8949

### Con PDF (1099-B)
1. Descarga el documento 1099-B en PDF
2. Ve a la pestaña "📄 PDF de Broker"
3. Sube el PDF
4. Haz clic en "Iniciar Análisis"
5. Descarga el resultado en formato 8949

## 📋 Formato de Salida

La aplicación genera automáticamente:

- **Form 8949 (CSV):** Formato listo para importar en el IRS Form 8949
- **Excel Completo:** Todas las transacciones con detalles completos
- **CSV Completo:** Versión CSV de todos los datos

### Columnas Estándar de Salida
```
Description (Símbolo/Descripción)
Date Acquired (Fecha de Compra)
Date Sold (Fecha de Venta)
Proceeds (Ingresos de la Venta)
Cost Basis (Costo Base)
Gain or (loss) (Ganancia o Pérdida)
(1f) Code(s) from instructions
(1g) Amount of adjustment
```

## 🔍 Detección Automática

La aplicación intenta detectar automáticamente el broker basado en:
1. **Nombre del archivo** - Busca palabras clave como "interactive", "fidelity", "schwab", etc.
2. **Estructura de columnas** - Analiza los nombres de las columnas
3. **Contenido** - Si los métodos anteriores fallan

## ⚠️ Troubleshooting

### "No se pudo detectar el formato del broker"
- Verifica que el archivo tenga las columnas esperadas
- Renombra el archivo incluye el nombre del broker (ej: "interactive_brokers_trades.csv")
- Asegúrate de descargar el reporte correcto

### Columnas no reconocidas
- Algunos brokers pueden tener nombres de columnas ligeramente diferentes
- Si es necesario, edita el archivo CSV antes de subirlo para que coincida con los nombres esperados

### Fechas mal parseadas
- Asegúrate que las fechas estén en formato MM/DD/YYYY o similar
- Si el problema persiste, convierte las fechas antes de subirlas

## 🛠️ Agregar Nuevo Broker

Para agregar soporte a un nuevo broker:

1. Abre `broker_parsers.py`
2. Crea una nueva clase heredando de `BrokerParser`
3. Implementa el método `parse()`
4. Agrega la detección en `BrokerDetector.detect_and_parse()`

Ejemplo:
```python
class MyBrokerParser(BrokerParser):
    @staticmethod
    def parse(file_data: BytesIO, filename: str) -> pd.DataFrame:
        df = pd.read_csv(file_data, encoding='utf-8')
        # Mapear columnas
        # Limpiar datos
        return df
```

## 📞 Soporte

Si tienes problemas con un broker específico:
1. Comparte el nombre de las columnas del archivo
2. Comparte un ejemplo (sin datos sensibles)
3. Se agregarán nuevos brokers según sea necesario
