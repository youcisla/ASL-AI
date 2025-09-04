import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch
from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the health check endpoint"""
    with patch('app.database.get_database') as mock_db:
        # Mock database ping
        mock_db.return_value.command = AsyncMock()
        
        with patch('app.ml_service.ml_service') as mock_ml:
            mock_ml.is_ready.return_value = True
            mock_ml.get_num_labels.return_value = 26
            
            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.get("/health")
                
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["model_loaded"] is True
            assert data["num_labels"] == 26
            assert data["db"] == "connected"


@pytest.mark.asyncio 
async def test_health_endpoint_model_not_loaded():
    """Test health endpoint when model is not loaded"""
    with patch('app.database.get_database') as mock_db:
        mock_db.return_value.command = AsyncMock()
        
        with patch('app.ml_service.ml_service') as mock_ml:
            mock_ml.is_ready.return_value = False
            mock_ml.get_num_labels.return_value = 0
            
            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.get("/health")
                
            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is False
            assert data["num_labels"] == 0


@pytest.mark.asyncio
async def test_predict_model_not_loaded():
    """Test prediction endpoint when model is not loaded"""
    with patch('app.ml_service.ml_service') as mock_ml:
        mock_ml.is_ready.return_value = False
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Create a dummy image file
            files = {"image": ("test.jpg", b"fake_image_data", "image/jpeg")}
            response = await ac.post("/api/predict", files=files)
            
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


@pytest.mark.asyncio
async def test_predict_invalid_file_type():
    """Test prediction endpoint with invalid file type"""
    with patch('app.ml_service.ml_service') as mock_ml:
        mock_ml.is_ready.return_value = True
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Create a dummy text file
            files = {"image": ("test.txt", b"not_an_image", "text/plain")}
            response = await ac.post("/api/predict", files=files)
            
        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_labels():
    """Test get labels endpoint"""
    with patch('app.ml_service.ml_service') as mock_ml:
        mock_labels = ["A", "B", "C", "D", "E"]
        mock_ml.is_ready.return_value = True
        mock_ml.get_labels.return_value = mock_labels
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/api/labels")
            
        assert response.status_code == 200
        assert response.json() == mock_labels


@pytest.mark.asyncio
async def test_get_labels_model_not_loaded():
    """Test get labels endpoint when model is not loaded"""
    with patch('app.ml_service.ml_service') as mock_ml:
        mock_ml.is_ready.return_value = False
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/api/labels")
            
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


@pytest.mark.asyncio
async def test_submit_feedback_invalid_upload_id():
    """Test feedback endpoint with invalid upload ID"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        feedback_data = {
            "upload_id": "invalid_id",
            "is_correct": True
        }
        response = await ac.post("/api/feedback", json=feedback_data)
        
    assert response.status_code == 400
    assert "Invalid upload_id" in response.json()["detail"]


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
        
    assert response.status_code == 200
    data = response.json()
    assert "ASL Classifier API" in data["message"]
    assert "version" in data
    assert "docs" in data
