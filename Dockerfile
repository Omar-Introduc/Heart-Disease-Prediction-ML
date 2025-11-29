# Use official Python 3.10 slim image as base
FROM python:3.10-slim-buster

# Set working directory in the container
WORKDIR /app

# Install system dependencies (needed for pyarrow/numpy/etc if wheels are missing, though usually safe)
# apt-get update && apt-get install -y build-essential ...
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to keep image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY .pre-commit-config.yaml .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the entrypoint to run the Streamlit app
# Address binding to 0.0.0.0 is crucial for Docker
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
