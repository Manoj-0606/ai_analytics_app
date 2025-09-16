# tests/test_app.py
import sys
import os
import pytest
from fastapi.testclient import TestClient

# --- ensure project root is on sys.path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.services import calculate_kpis, load_data

client = TestClient(app)


def test_root():
    """Root endpoint should return hello message"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Hello from AI Analytics App!"}


def test_load_data():
    """ETL: load_data should return DataFrame and warnings list"""
    df, warnings = load_data()
    assert isinstance(warnings, list)
    for col in ["month", "service", "cost"]:
        assert col in df.columns


def test_calculate_kpis():
    """KPI calculation should return expected keys"""
    result = calculate_kpis()
    assert "total_spend" in result
    assert "highest_service" in result
    assert "lowest_service" in result
    assert "warnings" in result


def test_kpi_endpoint():
    """API: /kpi endpoint should return KPIs"""
    response = client.get("/kpi")
    assert response.status_code == 200
    data = response.json()
    assert "total_spend" in data
    assert "highest_service" in data
    assert "lowest_service" in data


def test_services_endpoint():
    """API: /services endpoint should return services dict"""
    response = client.get("/services")
    assert response.status_code == 200
    data = response.json()
    assert "services" in data
