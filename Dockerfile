
# Dockerfile.api
ARG PYTHON_VERSION="3.12"
FROM python:${PYTHON_VERSION}-slim

LABEL maintainer="Matoki"
LABEL description="API with FastAPI"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Installer les dépendances
COPY requirements-prod.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-prod.txt

# Copier le code
COPY app.py .
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

