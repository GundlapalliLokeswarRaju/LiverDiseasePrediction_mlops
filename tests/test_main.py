import pytest
from httpx import AsyncClient
from main import app

# Sample patient data for testing
sample_input = {
    "Age": 45,
    "Gender": 1,
    "Total_Bilirubin": 1.2,
    "Direct_Bilirubin": 0.3,
    "Alkaline_Phosphatase": 210,
    "Alanine_Aminotransferase": 35,
    "Aspartate_Aminotransferase": 45,
    "Total_Proteins": 6.5,
    "Albumin": 3.2,
    "Albumin_Globulin_Ratio": 1.0
}

@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert "mlflow_uri" in data
    assert "model_uri" in data

@pytest.mark.asyncio
async def test_predict():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict", json=sample_input)
    if response.status_code == 503:
        # Model not loaded; acceptable in testing CI
        assert response.json()["detail"] == "Model not loaded."
    else:
        assert response.status_code == 200
        assert "prediction" in response.json()
