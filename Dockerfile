FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install nltk punkt tokenizer
RUN python -c "import nltk; nltk.download('punkt')"

# Copy model files and application code
COPY app.py .
COPY robust_model_trainer.py .
COPY templates/ ./templates/
COPY model_dir/ ./model_dir/

# Create a volume mount point for potential data persistence
VOLUME /app/data

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]