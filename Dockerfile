# Use the official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir huggingface_hub transformers langchain uvicorn

# Download models using huggingface-cli (one model per command for better caching)
RUN huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --repo-type model --revision main

RUN huggingface-cli download typeform/distilbert-base-uncased-mnli \
    --repo-type model --revision main

RUN huggingface-cli download google/flan-t5-large \
    --repo-type model --revision main

# Copy the requirements file and install application-specific dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# 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
