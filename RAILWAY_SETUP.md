# 🚀 Configuración de Railway + GitHub Actions

## Pasos para Configurar el Deploy Automático

### 1️⃣ En Railway

#### Crear un nuevo proyecto
1. Ve a [railway.app](https://railway.app)
2. Haz clic en "New Project"
3. Selecciona "GitHub Repo"
4. Autoriza GitHub y selecciona tu repositorio `ozytarget/CONVERTER`

#### Obtener el Railway Token
1. Ve a tu cuenta en Railway (esquina superior derecha → Settings)
2. Ve a "Tokens"
3. Crea un nuevo token llamado "GITHUB_DEPLOY"
4. **Copia el token completo**

### 2️⃣ En GitHub

#### Agregar el Secret
1. Ve a tu repositorio `ozytarget/CONVERTER`
2. Settings → Secrets and variables → Actions
3. Haz clic en "New repository secret"
4. **Name:** `RAILWAY_TOKEN`
5. **Value:** Pega el token de Railway que copiaste
6. Haz clic en "Add secret"

### 3️⃣ Verificar la Configuración

Los archivos necesarios ya están en tu repositorio:
- ✅ `.github/workflows/deploy.yml` - Workflow de GitHub Actions
- ✅ `Procfile` - Instrucciones para Railway
- ✅ `railway.json` - Configuración de Railway
- ✅ `.railwayignore` - Archivos a ignorar en Railway

### 4️⃣ Hacer Deploy

Simplemente haz un push a `main`:

```bash
git push origin main
```

GitHub Actions automáticamente:
1. Descargará el código
2. Validará que Python funcione correctamente
3. Hará deploy a Railway

### 5️⃣ Monitorear el Deploy

#### En GitHub
1. Ve a tu repositorio
2. Haz clic en "Actions"
3. Verás el workflow ejecutándose
4. Espera a que termine (debería tardar ~2-3 minutos)

#### En Railway
1. Ve a [railway.app](https://railway.app)
2. Selecciona tu proyecto "CONVERTER"
3. Verás los logs en tiempo real
4. Cuando esté listo, verás un enlace público para acceder a tu app

## 🔗 URL de tu App

Una vez que Railway termine el deploy, tendrás una URL como:
```
https://converter-production-xxxx.railway.app
```

Esta URL se actualizará automáticamente cada vez que hagas push a `main`.

## ⚙️ Configuración Adicional en Railway (Opcional)

Si quieres agregar variables de entorno en Railway:

1. Ve a tu proyecto en Railway
2. Variables → Nuevo
3. Por ejemplo, puedes agregar:
   - `PYTESSERACT_PATH` (si usas Windows en producción)
   - `PDF_UPLOAD_LIMIT` (límite de tamaño de PDF)

## 🛑 Troubleshooting

### "GitHub Actions fails"
- Verifica que el token de Railway está correctamente configurado en GitHub Secrets
- Revisa los logs en GitHub Actions para ver el error exacto

### "Railway deployment fails"
- Verifica los logs en Railway
- Asegúrate que `requirements.txt` tiene todas las dependencias
- Comprueba que no hay errores de sintaxis en Python

### "Pytesseract no funciona en Railway"
- Railway es una plataforma Linux, pytesseract necesita tesseract instalado
- Agrega `apt-get install tesseract-ocr` al build process si es necesario

## 📝 Próximas Veces

Para los próximos deploys, solo necesitas:

```bash
# Haz cambios
nano app.py

# Commit y push
git add -A
git commit -m "Descripción de cambios"
git push origin main

# ¡Listo! GitHub Actions y Railway se encargán del resto
```

## 🔄 Diferencia entre GitHub Actions y Railway

- **GitHub Actions:** Valida el código y ejecuta pruebas
- **Railway:** Hospeda y ejecuta tu aplicación Streamlit en la nube

Ambas están conectadas automáticamente.
