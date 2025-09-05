#!/usr/bin/env python3
"""
Backend startup script with model validation.
This script tests the models before starting the server.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory (backend) to path so we can import app
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Change working directory to backend so model paths work correctly
os.chdir(backend_dir)

async def validate_setup():
    """Validate that everything is set up correctly"""
    print("🔍 Validating backend setup...")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        # Import and test models
        from app.ml_service import ml_service
        from app.config import settings
        
        print(f"CNN model path: {settings.cnn_model_path}")
        print(f"VGG model path: {settings.vgg_model_path}")
        
        # Load models
        await ml_service.load_models()
        
        # Check model status
        cnn_ready = ml_service.is_ready("cnn")
        vgg_ready = ml_service.is_ready("vgg")
        
        print(f"✅ CNN Model: {'Ready' if cnn_ready else 'Not available'}")
        print(f"✅ VGG16 Model: {'Ready' if vgg_ready else 'Not available'}")
        
        if not (cnn_ready or vgg_ready):
            print("❌ No models are available. Please check your model files.")
            return False
            
        print("✅ Backend validation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Backend validation failed: {e}")
        return False


async def start_server():
    """Start the FastAPI server"""
    import uvicorn
    from app.main import app
    
    # Try different ports if 8000 is busy
    ports_to_try = [8000, 8001, 8002, 8003]
    
    for port in ports_to_try:
        try:
            print(f"🚀 Starting ASL Classifier API on port {port}...")
            print(f"📡 Server will be available at: http://localhost:{port}")
            print(f"📖 API docs will be available at: http://localhost:{port}/docs")
            print(f"❤️  Health check: http://localhost:{port}/health")
            print("\nPress Ctrl+C to stop the server")
            
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=port,
                reload=True,
                reload_dirs=["app"],
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            break  # If we get here, server started successfully
            
        except OSError as e:
            if "address already in use" in str(e).lower() or "10048" in str(e):
                print(f"⚠️  Port {port} is busy, trying next port...")
                continue
            else:
                raise e
    else:
        print("❌ All ports are busy. Please free up a port and try again.")


async def main():
    """Main startup function"""
    print("=" * 60)
    print("🧠 ASL CLASSIFIER BACKEND STARTUP")
    print("=" * 60)
    
    # Validate setup first
    if not await validate_setup():
        print("\n❌ Setup validation failed. Please fix the issues above.")
        return 1
    
    print("\n" + "=" * 60)
    
    # Start the server
    try:
        await start_server()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
