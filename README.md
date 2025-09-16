# Liver Disease Prediction API

This project implements an API for predicting liver disease using a machine learning model. It's built with FastAPI, containerized with Docker, and orchestrated with Kubernetes.

## Overview

The project consists of the following main components:

*   **`app/main.py`:** The FastAPI application that exposes the API endpoints for health check and prediction. It loads a pre-trained machine learning model and a scaler for data preprocessing.
*   **`models/liver_model.pkl`:** The pre-trained machine learning model for liver disease prediction (RandomForestClassifier).
*   **`models/data.pkl`:** The scaler used for preprocessing input data.
*   **`train.py`:** A script to train the machine learning model and save it along with the scaler.
*   **`app/Dockerfile`:** Defines the steps to build a Docker image for the API.
*   **`k8s-deploy.yml`:** A Kubernetes deployment file for deploying the API to a Kubernetes cluster.
*   **`index.html` & `style.css`:** A simple portfolio website showcasing the project and other skills.

## Project Structure


## Setup and Deployment

### Prerequisites

*   Python 3.9+
*   Docker
*   Minikube (or a Kubernetes cluster)
*   kubectl

### Steps

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd gundlapallilokeswarraju.disease_pred.github.io
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    cd app
    pip install -r requirements.txt
    cd ..
    ```

4.  **Train the model (optional):**

    *   If you want to retrain the model, run:

        ```bash
        python train.py
        ```

        *   This will generate `liver_model.pkl` and `data.pkl` files in the `models/` directory.

5.  **Build the Docker image:**

    ```bash
    docker build -t <dockerhub-username>/liver:latest ./app
    ```

    *   Replace `<dockerhub-username>` with your Docker Hub username.

6.  **Push the Docker image to Docker Hub:**

    ```bash
    docker push <dockerhub-username>/liver:latest
    ```

7.  **Deploy to Kubernetes (Minikube):**

    *   Start Minikube:

        ```bash
        minikube start --profile liver-mlops
        ```

    *   Apply the Kubernetes deployment configuration:

        ```bash
        kubectl apply -f k8s-deploy.yml
        ```

8.  **Access the API:**

    *   Get the service URL:

        ```bash
        minikube service liver-api-service --profile liver-mlops --url
        ```

    *   The API will be accessible at the provided URL.

## API Endpoints

*   **`/` (GET):** Health check endpoint. Returns 
`{"message": "Liver Disease prediction API is up!"}`.
*   **`/predict` (POST):** Prediction endpoint. Accepts patient data as a JSON payload and returns the prediction.

### Request Body for `/predict`

```json
{
  "Age": 62.0,
  "Total_Bilirubin": 1.2,
  "Direct_Bilirubin": 0.4,
  "Alkaline_Phosphatase": 220.0,
  "Alanine_Aminotransferase": 21.0,
  "Aspartate_Aminotransferase": 31.0,
  "Total_Proteins": 7.0,
  "Albumin": 4.0,
  "Albumin_Globulin_Ratio": 1.2,
  "Gender": 1
}