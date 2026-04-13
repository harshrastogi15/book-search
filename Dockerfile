# Use slim python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU only version first
# This is much smaller than default PyTorch which includes CUDA (GPU support)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install rest of dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports
EXPOSE 8000

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]