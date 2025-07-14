# ConMED-RL Docker Image
# Multi-stage build for optimized production image

# Build stage
FROM python:3.10-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY ../requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim as production

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Create non-root user early
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH \
    FLASK_APP=CDM-Software/web_application_demo.py \
    FLASK_ENV=production \
    WORKERS=2 \
    PORT=5000 \
    LOG_LEVEL=INFO

# Copy application files with proper ownership
COPY --chown=appuser:appuser ../ConMedRL ./ConMedRL/
COPY --chown=appuser:appuser ../CDM-Software ./CDM-Software/
COPY --chown=appuser:appuser ../Software_FQE_models ./Software_FQE_models/
COPY --chown=appuser:appuser ../image ./image/
COPY --chown=appuser:appuser ../Data ./Data/
COPY --chown=appuser:appuser ../Experiment\ Notebook ./Experiment\ Notebook/

# Create necessary directories
RUN mkdir -p /app/logs /app/temp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Change to application directory
WORKDIR /app/CDM-Software

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers ${WORKERS} --timeout 300 --log-level ${LOG_LEVEL} --access-logfile - --error-logfile - web_application_demo:create_app"] 