# ============================================================
# Dockerfile - API de prediccion de pago a tiempo
# Construir:  docker build -t api-creditos .
# Ejecutar :  docker run -p 8000:8000 api-creditos
# Probar   :  http://localhost:8000/docs
# ============================================================

# 1. Imagen base: Python liviano
FROM python:3.13-slim

# 2. Variables de entorno: evita archivos .pyc y muestra los logs al instante
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Carpeta de trabajo dentro del contenedor
WORKDIR /app

# 4. Instalar dependencias (se copia primero requirements.txt para aprovechar la cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el codigo y el modelo (el .dockerignore excluye lo que no hace falta)
COPY mlops_pipeline/ ./mlops_pipeline/

# 6. Ejecutar como usuario normal (buena practica de seguridad: no usar root)
RUN useradd --create-home appuser
USER appuser

# 7. Puerto en el que escucha la API
EXPOSE 8000

# 8. Comando de arranque: uvicorn sirve la app "app" del archivo model_deploy.py
CMD ["uvicorn", "model_deploy:app", "--app-dir", "mlops_pipeline", "--host", "0.0.0.0", "--port", "8000"]
