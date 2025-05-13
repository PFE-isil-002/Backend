import logging
import os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from .api.routes import router as api_router
from .core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    # Create application
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="API for UAV Authentication System",
        version="0.1.0",
    )
    
    # Configure CORS
    app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS] + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    # Ensure required directories exist
    ensure_directories()
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"Starting {settings.PROJECT_NAME}")
        
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info(f"Shutting down {settings.PROJECT_NAME}")
    
    return app


def ensure_directories():
    """Ensure that required directories exist"""
    # Create model directory if it doesn't exist
    if not settings.MODEL_DIR.exists():
        logger.info(f"Creating model directory: {settings.MODEL_DIR}")
        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create dataset directory if it doesn't exist
    if not settings.DATASET_DIR.exists():
        logger.info(f"Creating dataset directory: {settings.DATASET_DIR}")
        settings.DATASET_DIR.mkdir(parents=True, exist_ok=True)


app = create_app()


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "documentation": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
    



from .api.websocket import handle_websocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket)