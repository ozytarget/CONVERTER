# 🧠 Parser Universal Inteligente

## ¿Qué es?

El **Parser Universal Inteligente** es un sistema de detección automática que puede leer y convertir **CUALQUIER formato de broker** a formato IRS Form 8949, sin necesidad de crear un parser específico.

## 🔍 Cómo Funciona

### 1. **Detección de Columnas**
El parser analiza todas las columnas del archivo y usa **fuzzy matching** para identificar qué representa cada una:

- **Date Acquired** - Busca palabras como "open date", "purchase date", "fecha compra", etc.
- **Date Sold** - Busca "close date", "sale date", "fecha venta", etc.
- **Proceeds** - Busca "sale proceeds", "sale amount", "ingresos", etc.
- **Cost Basis** - Busca "cost basis", "amount invested", "costo base", etc.
- **Gain/Loss** - Busca "gain", "loss", "p&l", "ganancia", etc.
- **Description** - Busca "symbol", "ticker", "description", etc.

### 2. **Mapeo Automático**
Una vez detectadas las columnas, el parser automáticamente:
- Renombra las columnas a formato estándar
- Limpia los valores numéricos (elimina $, %, comas, espacios)
- Normaliza las fechas al formato MM/DD/YYYY
- Calcula "Gain or (loss)" si no existe
- Maneja múltiples encodings de archivo

### 3. **Tolerancia a Variaciones**
El parser es flexible y puede manejar:
- Diferentes idiomas (español, inglés, etc.)
- Diferentes símbolos ($, €, ¥, etc.)
- Diferentes separadores y encodings
- Columnas en diferente orden
- Columnas adicionales que no necesita

## 📊 Ejemplo de Funcionamiento

### Entrada (archivo desconocido):
```
Ticker | Open Date | Close Date | Cantidad | Precio Entrada | Precio Salida | PnL
AAPL   | 2024-01-15 | 2024-06-20 | 100 | $150.00 | $180.00 | $3,000.00
```

### Detección:
- `Ticker` → Description (detecta palabra clave "ticker")
- `Open Date` → Date Acquired (detecta "open date")
- `Close Date` → Date Sold (detecta "close date")
- `Cantidad` → Quantity (detecta "cantidad")
- `Precio Entrada` → Cost Basis (fuzzy match con "entrada" = entrada/compra)
- `Precio Salida` → Proceeds (fuzzy match con "salida" = venta)
- `PnL` → Gain or (loss) (detecta "PnL")

### Salida (formato estándar):
```
Description | Date Acquired | Date Sold | Proceeds | Cost Basis | Gain or (loss)
AAPL        | 01/15/2024    | 06/20/2024 | 18000.00 | 15000.00 | 3000.00
```

## ✨ Características

### ✅ Ventajas
- No necesita configuración manual
- Funciona con cualquier broker
- Maneja múltiples idiomas
- Tolerante a errores y variaciones
- Automáticamente calcula valores faltantes

### ⚠️ Limitaciones
- Necesita al menos las columnas básicas (fechas, montos)
- Si las columnas son muy ambiguas, podría confundirse
- Requiere que los datos estén en formato tabular (CSV, Excel)

## 🛠️ Cómo Usar

1. **Descarga tu reporte del broker** en formato CSV o Excel
2. **Sube el archivo** a la app
3. **La app automáticamente:**
   - Detecta el broker (si es conocido)
   - O usa el parser universal (si es desconocido)
   - Convierte a formato 8949
4. **Descarga el resultado**

## 📝 Palabras Clave Soportadas

### Fechas de Compra
`date acquired`, `open date`, `purchase date`, `buy date`, `entry date`, `date opened`, `acquisition date`, `fecha compra`, `fecha adquisición`

### Fechas de Venta
`date sold`, `close date`, `sale date`, `sell date`, `exit date`, `fecha venta`, `fecha cierre`, `fecha salida`

### Ingresos
`proceeds`, `sale proceeds`, `proceeds amount`, `sale amount`, `monto venta`, `ingresos`, `total proceeds`

### Costo Base
`cost basis`, `basis`, `cost`, `amount invested`, `entry cost`, `total cost`, `costo base`, `costo`, `inversión`, `purchase price`

### Ganancia/Pérdida
`gain`, `loss`, `gain or loss`, `gain/loss`, `p&l`, `profit loss`, `return`, `ganancia`, `pérdida`, `total return`, `realized gain`

### Descripción
`symbol`, `ticker`, `description`, `security`, `instrument`, `product`, `name`, `símbolo`

### Cantidad
`quantity`, `shares`, `qty`, `amount`, `units`, `cantidad`, `acciones`

## 🔄 Cómo Agregar un Nuevo Broker

Si el parser universal no funciona bien con tu broker:

1. Ve a `BROKERS_GUIDE.md`
2. Reporta el nombre del broker y las columnas que usa
3. Se creará un parser específico para mejor precisión

## 📞 Soporte

Si tienes problemas:

1. **Verifica que el archivo tenga las columnas básicas:**
   - Una columna de fecha de compra
   - Una columna de fecha de venta
   - Una columna de ingresos de venta
   - Una columna de costo base

2. **Si falta alguna columna:**
   - Agrega la columna manualmente antes de subir
   - O descarga un reporte diferente de tu broker

3. **Si las columnas tienen nombres muy raros:**
   - Renombralas a nombres más estándar antes de subir
   - O crea un issue reportando el broker para soporte específico
