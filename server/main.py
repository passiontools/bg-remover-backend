"""
AI Background Remover - FastAPI Backend
High-performance, scalable background removal service using rembg
"""

import io
import logging
from typing import Optional
from functools import lru_cache
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from rembg import remove, new_session
from PIL import Image
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="AI Background Remover API",
    description="High-performance API for removing image backgrounds using AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration - Update origins for production
ALLOWED_ORIGINS = [
    "https://ai-image-background-remover.netlify.app",
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp"
}

# Global session for rembg (loaded once for efficiency)
_session = None


def get_session():
    """Get or create rembg session (singleton pattern for performance)"""
    global _session
    if _session is None:
        logger.info("Initializing rembg session with u2net model...")
        _session = new_session("u2netp")  # Smaller model, uses less memory
        logger.info("rembg session initialized successfully")
    return _session


@app.on_event("startup")
async def startup_event():
    """Pre-load the model on startup for faster first request"""
    logger.info("Starting up AI Background Remover API...")
    # Pre-initialize the session
    get_session()
    logger.info("API ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global _session
    if _session:
        del _session
        _session = None
    logger.info("API shutdown complete")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Background Remover API",
        "version": "1.0.0",
        "endpoints": {
            "remove_background": "POST /remove-bg/",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": _session is not None
    }


@app.post("/remove-bg/")
async def remove_background(
    file: UploadFile = File(...),
    request: Request = None
):
    """
    Remove background from uploaded image
    
    Args:
        file: Image file (JPG, PNG, or WEBP)
        
    Returns:
        PNG image with transparent background
        
    Raises:
        HTTPException: For invalid files or processing errors
    """
    client_ip = request.client.host if request else "unknown"
    logger.info(f"Processing request from {client_ip} - File: {file.filename}")
    
    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_CONTENT_TYPES.keys())}"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Validate file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file provided"
            )
        
        logger.info(f"Processing image: {file.filename}, size: {len(content)} bytes")
        
        # Open and validate image
        try:
            input_image = Image.open(io.BytesIO(content))
            # Convert to RGB if necessary (handles RGBA, etc.)
            if input_image.mode not in ('RGB', 'RGBA'):
                input_image = input_image.convert('RGB')
        except Exception as img_error:
            logger.error(f"Invalid image format: {img_error}")
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image file"
            )
        
        # Process with rembg
        try:
            session = get_session()
            output_bytes = remove(content, session=session)
        except Exception as rembg_error:
            logger.error(f"Background removal failed: {rembg_error}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process image. Please try again."
            )
        
        # Validate output
        if not output_bytes or len(output_bytes) == 0:
            raise HTTPException(
                status_code=500,
                detail="Processing produced no output. Please try a different image."
            )
        
        logger.info(f"Successfully processed image: {file.filename}, output size: {len(output_bytes)} bytes")
        
        # Return as streaming response (memory efficient)
        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{file.filename.rsplit(".", 1)[0]}_no_bg.png"',
                "X-Processed-Size": str(len(output_bytes)),
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )


@app.exception_handler(413)
async def request_entity_too_large(request: Request, exc):
    """Custom handler for file size limit exceeded"""
    return JSONResponse(
        status_code=413,
        content={"detail": "File too large. Maximum size is 10MB."}
    )


@app.exception_handler(500)
async def internal_server_error(request: Request, exc):
    """Custom handler for internal server errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."}
    )


# Rate limiting middleware (simple in-memory implementation)
# For production, use Redis-based rate limiting
request_counts = {}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware"""
    # Only apply to POST requests to /remove-bg/
    if request.url.path == "/remove-bg/" and request.method == "POST":
        client_ip = request.client.host if request.client else "unknown"
        
        # Simple rate limiting: 20 requests per minute per IP
        # In production, use Redis with sliding window
        import time
        current_time = int(time.time() / 60)  # Current minute
        
        key = f"{client_ip}:{current_time}"
        request_counts[key] = request_counts.get(key, 0) + 1
        
        # Cleanup old entries (keep last 2 minutes)
        keys_to_delete = [k for k in request_counts.keys() 
                        if int(k.split(":")[1]) < current_time - 1]
        for k in keys_to_delete:
            del request_counts[k]
        
        if request_counts[key] > 20:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait a moment and try again."}
            )
    
    return await call_next(request)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,  # Use multiple workers in production with gunicorn
        log_level="info"
    )
