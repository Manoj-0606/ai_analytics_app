# main.py
from dotenv import load_dotenv
load_dotenv()

import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router as kpi_router

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ai-analytics-app")

app = FastAPI(title="AI Analytics App", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} completed_in={process_time:.3f}s status={response.status_code}"
    )
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    return response

@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    return {"message": "Hello from AI Analytics App!"}

app.include_router(kpi_router, prefix="", tags=["kpi"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
