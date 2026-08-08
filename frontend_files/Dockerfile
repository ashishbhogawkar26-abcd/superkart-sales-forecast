# Use Python 3.12 slim as base image
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY streamlit_app.py .

# Expose port 8501 (default for Streamlit)
EXPOSE 8501

# Run the Streamlit application.
# XSRF and CORS checks are disabled because this container runs behind the
# GitHub Codespaces reverse proxy, which rewrites Host/Or
