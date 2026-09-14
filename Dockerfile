# ==============================================================================
# Multi-Stage Dockerfile for BioVision Full-Stack Deployment
# Compatible with: Hugging Face Spaces, Render, Railway, Fly.io, Cloud Run, VPS
# ==============================================================================

# --- Stage 1: Build React Frontend ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
# Build production bundle to /app/frontend/dist
RUN npm run build

# --- Stage 2: Production Python Inference & Web Server ---
FROM python:3.10-slim AS runner

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Install system dependencies for OpenCV and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-optimized wheels first (keeps image small, ~700MB vs 5GB CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install backend production dependencies
COPY backend/requirements_prod.txt ./backend/requirements_prod.txt
RUN pip install --no-cache-dir -r ./backend/requirements_prod.txt

# Copy application code and artifacts
COPY backend/ ./backend/
COPY results/ ./results/

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create uploads and cache directories
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Expose port (7860 is default for Hugging Face Spaces, overridable by PORT env)
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch FastAPI app with Uvicorn
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
