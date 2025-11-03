# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml ./
# Copy lock file if it exists (optional)
COPY uv.lock* ./

# Install Python dependencies using uv
# uv will automatically use lock file if available
RUN uv sync --no-dev

# Copy application code
COPY app.py ./
COPY static/ ./static/
COPY catdog_model.h5 ./

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run the application
CMD ["uv", "run", "python", "app.py"]

