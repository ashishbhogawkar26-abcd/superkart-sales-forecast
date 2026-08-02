# Use Python 3.12 slim as base image
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the serialized model
COPY app.py .
COPY superkart_model.joblib .

# Expose port 7860
EXPOSE 7860

# Run the Flask application
CMD ["python", "app.py"]
