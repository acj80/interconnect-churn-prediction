from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_CUSTOMER = {
    "Type": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 19.65,
    "TotalCharges": 67.55,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "InternetService": "No internet service",
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
    "MultipleLines": "No",
    "HasInternet": 0,
    "HasPhone": 1,
    "InternetAddOnCount": 0,
    "StreamingCount": 0,
    "HasTechProtection": 0,
    "AutomaticPayment": 0,
}


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "CatBoostClassifier + One-Hot Encoding"
    assert data["threshold"] == 0.5
    assert data["features_expected"] == 23


def test_predict_valid_customer():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 0
    assert data["threshold"] == 0.5
    assert data["risk_level"] == "low"

    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_probability_matches_reference():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    probability = response.json()["churn_probability"]

    expected_probability = 0.26085351452633293

    assert abs(
        probability - expected_probability
    ) < 1e-12


def test_invalid_phone_combination():
    invalid_customer = VALID_CUSTOMER.copy()

    invalid_customer["MultipleLines"] = "No phone service"

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422


def test_negative_monthly_charges():
    invalid_customer = VALID_CUSTOMER.copy()

    invalid_customer["MonthlyCharges"] = -10

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422


def test_missing_required_field():
    invalid_customer = VALID_CUSTOMER.copy()

    del invalid_customer["MonthlyCharges"]

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422


def test_invalid_senior_citizen_value():
    invalid_customer = VALID_CUSTOMER.copy()

    invalid_customer["SeniorCitizen"] = 2

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422


def test_invalid_internet_combination():
    invalid_customer = VALID_CUSTOMER.copy()

    invalid_customer["HasInternet"] = 0
    invalid_customer["InternetService"] = "Fiber optic"

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422


def test_prediction_response_structure():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    data = response.json()

    expected_keys = {
        "prediction",
        "churn_probability",
        "risk_level",
        "threshold",
    }

    assert set(data.keys()) == expected_keys


def test_prediction_types():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    data = response.json()

    assert isinstance(data["prediction"], int)
    assert isinstance(data["churn_probability"], float)
    assert isinstance(data["risk_level"], str)
    assert isinstance(data["threshold"], float)
