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

# System deps for sklearn + llama
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/models

ENV PYTHONUNBUFFERED=1

CMD gunicorn -w 1 -b 0.0.0.0:$PORT app:app
