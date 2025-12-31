# FROM python:3.10-slim

# WORKDIR /app

# # System deps (for scikit-learn)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential gcc g++ \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# # Create data dirs
# RUN mkdir -p /app/data /app/models

# ENV PYTHONUNBUFFERED=1

# # Render uses PORT env
# CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:$PORT", "app:app"]

FROM python:3.10-slim

WORKDIR /app

# System deps (needed for llama-cpp + numpy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/models

ENV PYTHONUNBUFFERED=1

# ✅ FIX: increase gunicorn timeout to prevent GGUF load kill
CMD gunicorn -w 1 -k sync --timeout 180 -b 0.0.0.0:$PORT app:app
