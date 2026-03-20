"""
AI Background Remover - Mini Service
FastAPI server for background removal using rembg
"""

import io
import logging
import os
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global session for rembg
_session = None
remove_func = None

def get_rembg_session():
    """Lazy load rembg session - only when first request comes in"""
    global _session, remove_func
    if _session is None:
        try:
            from rembg import remove, new_session
            logger.info("Loading rembg model...")
            # u2netp is smaller (4MB vs 176MB) and fits in 512MB free tier RAM
            _session = new_session("u2netp")
            remove_func = remove
            logger.info("rembg model loaded successfully")
        except ImportError:
            logger.warning("rembg not installed, using mock mode")
            _session = "mock"
    return _session

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    logger.info("Starting AI Background Remover service...")
    # Do NOT load model at startup - load lazily on first request to save RAM
    yield
    logger.info("Shutting down AI Background Remover service...")

# FastAPI app
app = FastAPI(
    title="AI Background Remover Service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - Allow all origins for internal service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg": "jpg",
    "image/png": "png", 
    "image/webp": "webp"
}

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "AI Background Remover",
        "version": "1.0.0",
        "rembg_loaded": _session is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "rembg_loaded": _session is not None}

@app.post("/remove-bg/")
async def remove_background(
    file: UploadFile = File(...),
    request: Request = None
):
    """Remove background from uploaded image"""
    logger.info(f"Processing: {file.filename}, type: {file.content_type}")
    
    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {list(ALLOWED_CONTENT_TYPES.keys())}"
        )
    
    try:
        # Read file
        content = await file.read()
        
        # Validate size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Process with rembg
        session = get_rembg_session()
        
        if session == "mock":
            # Mock mode - return original as PNG
            from PIL import Image
            img = Image.open(io.BytesIO(content))
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            output = io.BytesIO()
            img.save(output, format='PNG')
            output_bytes = output.getvalue()
        else:
            # Real rembg processing
            output_bytes = remove_func(content, session=session)
        
        if not output_bytes:
            raise HTTPException(status_code=500, detail="Processing failed")
        
        logger.info(f"Processed successfully: {len(output_bytes)} bytes")
        
        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{file.filename.rsplit(".", 1)[0]}_no_bg.png"',
                "Cache-Control": "no-store"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
