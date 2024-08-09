# Use the official Python base image
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the rest of the application code
COPY . /app

# # Copy the requirements file
# COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which FastAPI will run
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]