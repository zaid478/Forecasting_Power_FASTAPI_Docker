# Forecasting Power Using Machine Learning - FastAPI and Docker Setup

This guide will help you pull and run a Docker image for a FastAPI application that serves an inference model for forecasting active power. The application is exposed via FastAPI, and you can interact with it through the Swagger UI.

## Prerequisites

Before you begin, ensure that you have the following installed on your system:

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/) if you don't have it installed.

## Step 1: Pull the Docker Image

First, you'll need to pull the Docker image from Docker Hub. Replace `<your-username>` and `<your-image-name>` with the appropriate values.

```bash 
docker pull zaid478/forecasting_v2emax:latest
```


## Step 2: Run the Docker Container

Once the image is pulled, you can run it using the following command. The container will expose the FastAPI application on port 8000 by default
```bash
docker run -d -p 8000:8000 zaid478/forecasting_v2emax:latest
```
## Explanation of the Command

- -d: Run the container in detached mode (in the background).
- -p: 8000:8000: Map port 8000 of the host to port 8000 of the container.

## Step 3: Access the FastAPI Application

After running the container, you can access the FastAPI application and its Swagger UI by navigating to:
```bash
http://localhost:8000/docs
```
The Swagger UI will provide you with a user-friendly interface to interact with the API endpoints, send requests, and view responses.

## Step 4: Submit a POST Request for Inference

You can submit a POST request to the FastAPI endpoint via the Swagger UI. Follow these steps:
- Go to http://localhost:8000/docs.
- Locate the endpoint for inference (e.g., /predict)
- Click on the "Try it out" button.
- Enter the required data in the provided fields.
- Click "Execute" to run the inference.

## Step 5: Stop the Docker Container (Optional)

If you want to stop the Docker container, you can do so by running:

```bash
docker stop <container_id>
```
To find the container ID, you can list all running containers:
```bash
docker ps