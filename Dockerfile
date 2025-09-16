# Dockerfile
# Use Python 3.11 (stable for FastAPI, Streamlit, pandas, faiss-cpu, etc.)
FROM python:3.11-slim

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies (needed for building some packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker cache efficiency
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command (FastAPI API server)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
