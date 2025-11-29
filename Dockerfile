# Use python 3.10 slim buster as requested
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for scipy/numpy/etc)
# build-essential is often needed for compiling python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY .streamlit/ .streamlit/

# Copy main app entry point if it's in src/app.py
# The instruction says "configure entry point to execute Streamlit application (src/app.py)"

# Expose Streamlit port
EXPOSE 8501

# Entrypoint
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
