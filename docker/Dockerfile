# ─────────────────────────────────────────────────────────────
# Stage 1 — Base image
# Using slim variant to keep image size small
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ─────────────────────────────────────────────────────────────
# System dependencies
# MediaPipe needs libGL and libgthread (OpenCV headless still
# needs some system libs on Linux)
# ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────
# Working directory inside container
# ─────────────────────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────────────────────
# Install Python dependencies first (cached layer)
# Copy requirements before code so Docker caches this layer
# ─────────────────────────────────────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────
# Copy backend source code
# ─────────────────────────────────────────────────────────────
COPY backend/ .

# ─────────────────────────────────────────────────────────────
# Expose FastAPI port
# ─────────────────────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────────────────────
# Run the server
# host=0.0.0.0 makes it accessible outside the container
# ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]