
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metadata():
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert data["sequence_length"] == 30
    assert len(data["required_features"]) > 0


def test_prediction_with_valid_sequence():
    metadata = client.get("/metadata").json()
    features = metadata["required_features"]

    sequence = [
        {feature: 0.0 for feature in features}
        for _ in range(30)
    ]

    response = client.post(
        "/predict",
        json={"sequence": sequence}
    )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_RUL_cycles" in result
    assert result["predicted_RUL_cycles"] >= 0


def test_prediction_rejects_short_sequence():
    metadata = client.get("/metadata").json()
    features = metadata["required_features"]

    sequence = [
        {feature: 0.0 for feature in features}
        for _ in range(29)
    ]

    response = client.post(
        "/predict",
        json={"sequence": sequence}
    )

    assert response.status_code == 422
