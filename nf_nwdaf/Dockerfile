# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpcap-dev \
    net-tools \
    iputils-ping \
    iproute2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make sure the script is executable
RUN chmod +x realtime_fl_detector.py

# Create directory for flow data
RUN mkdir -p /app/flow_data

# Default command (will be overridden by docker-compose)
CMD ["python3", "realtime_fl_detector.py", "--iface", "br-free5gc", "--slice_id", "1"]
