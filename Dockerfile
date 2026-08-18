FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8502)
EXPOSE 8000 8502

# Default command starts FastAPI app
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
