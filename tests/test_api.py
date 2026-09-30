from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)

# -----------------------------------------------------------------------------
# 1. Cas nominal : prédiction correcte avec [1.0, 2.0, 3.0]:
# -----------------------------------------------------------------------------
def test_predict_success_with_new_values():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "predictions": [2.0, 4.0, 6.0]
    }


# -----------------------------------------------------------------------------
# 2. Cas de prédiction incorrecte : résultat volontairement faux
# -----------------------------------------------------------------------------
def test_predict_incorrect_result():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]},
    )

    # Résultat volontairement faux.
    # Le test vérifie que l'API ne retourne pas cette mauvaise prédiction.

    assert response.status_code == 200
    assert response.json() != {
        "predictions": [7.0, 7.0, 7.0]
    }

# -----------------------------------------------------------------------------
# 3. Cas invalide : JSON incorrect --> "features" est absent
# -----------------------------------------------------------------------------
def test_predict_invalid_json():
    response = client.post(
        "/predict",
        json=[3.5, 1.2, 4.9],
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Input should be a valid dictionary or object to extract fields from"
