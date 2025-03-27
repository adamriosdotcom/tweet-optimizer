"""
Tests for the API layer.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from tweet_api import app
from config import settings

# Create test client
client = TestClient(app)

# Test API key
TEST_API_KEY = "test_api_key"
settings.API_KEY = TEST_API_KEY

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "version" in response.json()

def test_predict_tweet_unauthorized():
    """Test tweet prediction without API key"""
    response = client.post(
        "/predict",
        json={"text": "Test tweet"}
    )
    assert response.status_code == 403

def test_predict_tweet():
    """Test tweet prediction with valid API key"""
    response = client.post(
        "/predict",
        json={"text": "Test tweet"},
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "predicted_engagement" in data
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1

def test_optimize_tweet():
    """Test tweet optimization"""
    response = client.post(
        "/optimize",
        json={
            "text": "Test tweet for optimization",
            "iterations": 2,
            "variations": 3
        },
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert "original_text" in data
    assert "optimized_text" in data
    assert "improvement" in data
    assert "processing_time" in data
    assert data["processing_time"] > 0

def test_rate_limiting():
    """Test rate limiting"""
    # Make multiple requests quickly
    for _ in range(61):  # Should hit rate limit
        response = client.post(
            "/predict",
            json={"text": "Test tweet"},
            headers={"X-API-Key": TEST_API_KEY}
        )
    
    # Next request should be rate limited
    response = client.post(
        "/predict",
        json={"text": "Test tweet"},
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 429

def test_invalid_input():
    """Test invalid input handling"""
    # Test empty text
    response = client.post(
        "/predict",
        json={"text": ""},
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 422
    
    # Test text too long
    response = client.post(
        "/predict",
        json={"text": "a" * 281},  # Twitter's limit is 280
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 422
    
    # Test invalid iterations
    response = client.post(
        "/optimize",
        json={
            "text": "Test tweet",
            "iterations": 11,  # Max is 10
            "variations": 5
        },
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 422

def test_error_handling():
    """Test error handling"""
    # Test with invalid JSON
    response = client.post(
        "/predict",
        data="invalid json",
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 422
    
    # Test with missing required field
    response = client.post(
        "/predict",
        json={},
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 422 