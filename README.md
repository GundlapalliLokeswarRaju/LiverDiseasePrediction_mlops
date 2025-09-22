# Liver Disease Prediction API

This project implements an API for predicting liver disease using a machine learning model. It's built with FastAPI, containerized with Docker, and orchestrated with Kubernetes. It also features monitoring using Prometheus and Grafana.

## Overview

The project consists of the following main components:

* **`app/main.py`:** The FastAPI application that exposes API endpoints for health checks and predictions. It loads a pre-trained machine learning model and a scaler for data preprocessing.
* **`models/liver_model.pkl`:** The pre-trained machine learning model for liver disease prediction (RandomForestClassifier).
* **`models/data.pkl`:** The scaler used for preprocessing input data.
* **`train.py`:** A script to train the machine learning model and save it along with the scaler.
* **`app/Dockerfile`:** Defines the steps to build a Docker image for the API.
* **`k8s-deploy.yml`:** A Kubernetes deployment file to deploy the API on a Kubernetes cluster.
* **`prometheus-config.yaml`:** A Prometheus configuration file to enable scraping of the `/metrics` endpoint.
  

## Setup and Deployment

### Prerequisites

* Python 3.9+
* Docker
* Minikube (or another Kubernetes cluster)
* kubectl
* (Optional) Helm for Kubernetes package management

### Steps

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd LiverDiseasePrediction_mlops
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    # On Linux/macOS
    source venv/bin/activate
    # On Windows
    venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    cd app
    pip install -r requirements.txt
    cd ..
    ```

4. **Train the model (optional):**

    If you want to retrain the model, run:

    ```bash
    python train.py
    ```

    This will generate/update `models/liver_model.pkl` and `models/data.pkl`.

5. **Build the Docker image:**

    ```bash
    docker build -t <dockerhub-username>/liver:latest ./app
    ```

    Replace `<dockerhub-username>` with your Docker Hub username.

6. **Push the Docker image to Docker Hub:**

    ```bash
    docker push <dockerhub-username>/liver:latest
    ```

7. **Deploy to Kubernetes (Minikube):**

    * Start Minikube:

      ```bash
      minikube start --profile liver-mlops
      ```

    * Apply the Kubernetes deployment configuration:

      ```bash
      kubectl apply -f k8s-deploy.yml
      ```

8. **Access the API:**

    * Get the service URL:

      ```bash
      minikube service liver-api-service --profile liver-mlops --url
      ```

    * The API will be accessible at the provided URL.

## API Endpoints

* **`/` (GET):** Health check endpoint. Returns:
  
  ```json
  {"message": "Liver Disease prediction API is up!"}
  ```

* **`/predict` (POST):** Prediction endpoint. Accepts patient data as a JSON payload and returns the prediction.

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
```

* **`/metrics` (GET):** Exposes Prometheus metrics for monitoring the API.

## Monitoring with Prometheus and Grafana

This project includes monitoring support to scrape API metrics and visualize them with Grafana.

### Prometheus Setup

1. **Deploy Prometheus:**

   You can deploy Prometheus to your Kubernetes cluster using either Helm or a static YAML configuration.

   **Using Helm:**

   ```bash
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm repo update
   helm install prometheus prometheus-community/prometheus --namespace monitoring --create-namespace
   ```

   **Using YAML files:**

   Apply your custom Prometheus configuration (ensure your `prometheus-config.yaml` is correctly referenced):

   ```bash
   kubectl apply -f prometheus-config.yaml
   ```

2. **Scrape Configuration:**

   In `prometheus-config.yaml`, ensure that the target is set correctly. For example, if you need to scrape metrics from a different port (like NodePort 1010), update the target accordingly:

   ```yaml
   scrape_configs:
     - job_name: 'liver-api'
       static_configs:
         - targets: ['liver-api-service.default.svc.cluster.local:1010']
   ```

3. **Verify Prometheus:**

   Access the Prometheus dashboard and check the "Targets" page to ensure that the `/metrics` endpoint is being scraped properly.

### Grafana Setup

1. **Deploy Grafana:**

   You can deploy Grafana in your cluster using Helm.

   ```bash
   helm repo add grafana https://grafana.github.io/helm-charts
   helm repo update
   helm install grafana grafana/grafana --namespace monitoring --create-namespace
   ```

2. **Configure Grafana:**

   - Retrieve the Grafana admin password:

     ```bash
     kubectl get secret --namespace monitoring grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
     ```

   - Port-forward to access the Grafana dashboard:

     ```bash
     kubectl port-forward --namespace monitoring service/grafana 3000:80
     ```

   - Open your browser and navigate to `http://localhost:3000`, then log in using `admin` as the username and the retrieved password.

3. **Add Prometheus as a Data Source:**

   In the Grafana UI:
   - Navigate to **Configuration** > **Data Sources**.
   - Click **Add data source** and select **Prometheus**.
   - Set the URL to the Prometheus server (for example, `http://prometheus-server.monitoring.svc.cluster.local:80` if using the default service name from Helm).
   - Click **Save & Test** to confirm it connects.

4. **Create Dashboards:**

   Import or create dashboards to visualize the metrics scraped from your API.

## Troubleshooting

- **Metrics Not Updating in Prometheus/Grafana:**
  - Ensure the Kubernetes Service correctly maps the container port (8000) to the service port.
  - Verify DNS and network connectivity between Prometheus, Grafana, and your API.
  - Check the Prometheus "Targets" page for any scrape errors.

- **Container Issues:**
  - Check pod logs with:

    ```bash
    kubectl logs <pod-name>
    ```

Happy coding!

# integrating MLflow

1. mkdir -p ~/mlfow_artifacts
2. mlflow server ^
  --backend-store-uri sqlite:///C:/Users/lokes/OneDrive/Desktop/liver/mlflow.db ^
  --default-artifact-root file:///C:/Users/lokes/OneDrive/Desktop/liver/mlflow_artifacts ^
  --host 0.0.0.0 --port 5000
3. set MLFLOW_TRACKING_URI=http://localhost:5000
4. 