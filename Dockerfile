# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install nltk punkt data (required for tokenization)
RUN python -m nltk.downloader punkt

# Copy the application code and other necessary files
COPY app.py hybrid_classifier.py run_classification.py ./
COPY templates/ ./templates/
COPY customer_classification_model_lr.pkl customer_classification_model_svm.pkl ./
COPY consumer_complaints.csv ./

# Define environment variable
ENV FLASK_APP=app.py

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]