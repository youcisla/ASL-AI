#!/usr/bin/env python3
"""
Simple backend startup on port 8001
"""

import sys
import os
from pathlib import Path
import asyncio
import uvicorn

# Add parent directory (backend) to path so we can import app
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Change working directory to backend so model paths work correctly
os.chdir(backend_dir)

async def main():
    """Start the server on port 8001"""
    from app.main import app
    
    print("🚀 Starting ASL Classifier API on port 8001...")
    print("📡 Server: http://localhost:8001")
    print("📖 Docs: http://localhost:8001/docs")
    print("❤️  Health: http://localhost:8001/health")
    print("\nPress Ctrl+C to stop")
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,  # Disable reload to avoid issues
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
