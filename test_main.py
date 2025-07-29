import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_login_success():
    response = client.post("/token", data={"username": "atm_ops", "password": "password"})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_failure():
    response = client.post("/token", data={"username": "atm_ops", "password": "wrong"})
    assert response.status_code == 400

def test_train_model_file_not_found():
    response = client.get("/train-model/", params={"csv_path": "nonexistent.csv"})
    assert response.status_code == 400

def test_train_model_invalid_file(tmp_path):
    file = tmp_path / "empty.csv"
    file.write_text("")
    response = client.get("/train-model/", params={"csv_path": str(file)})
    assert response.status_code == 400

def test_refill_request_lifecycle():
    # Login as ATM Operations Staff
    login_resp = client.post("/token", data={"username": "atm_ops", "password": "password"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create refill request
    create_resp = client.post("/refill-requests/", json={
        "atm_id": "ATM123",
        "requested_amount": 10000,
        "comment": "Need refill soon"
    }, headers=headers)
    assert create_resp.status_code == 201
    request_id = create_resp.json()["request_id"]

    # List refill requests as ATM Ops (should see own request)
    list_resp = client.get("/refill-requests/", headers=headers)
    assert list_resp.status_code == 200
    assert any(r["request_id"] == request_id for r in list_resp.json())

    # Login as Branch Operations Manager
    login_mgr = client.post("/token", data={"username": "branch_mgr", "password": "password"})
    token_mgr = login_mgr.json()["access_token"]
    headers_mgr = {"Authorization": f"Bearer {token_mgr}"}

    # List pending requests as Branch Manager (should see the request)
    list_mgr_resp = client.get("/refill-requests/", headers=headers_mgr)
    assert list_mgr_resp.status_code == 200
    assert any(r["request_id"] == request_id for r in list_mgr_resp.json())

    # Approve the refill request
    action_resp = client.post(f"/refill-requests/{request_id}/action", json={"action": "approve"}, headers=headers_mgr)
    assert action_resp.status_code == 200
    assert "approved" in action_resp.json()["message"]

    # Get audit trail as Branch Manager
    audit_resp = client.get(f"/refill-requests/{request_id}/audit", headers=headers_mgr)
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert any(a["action"] == "approved" for a in audit_data)

def test_predict_refill_without_training():
    login_resp = client.post("/token", data={"username": "atm_ops", "password": "password"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/predict-refill/", params={"current_date": "2023-01-01", "days": 5}, headers=headers)
    assert response.status_code == 400
    assert "Model not trained" in response.json()["detail"]
