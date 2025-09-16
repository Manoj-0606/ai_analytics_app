# main.py
from dotenv import load_dotenv
load_dotenv()

import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# routes are in app/ folder
from app.routes import router as kpi_router

# --- Logging setup ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ai-analytics-app")

app = FastAPI(title="AI Analytics App", version="0.1")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Middleware for request metrics ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} "
        f"completed_in={process_time:.3f}s "
        f"status={response.status_code}"
    )

    # include latency in response header
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    return response


@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    return {"message": "Hello from AI Analytics App!"}


# include routes
app.include_router(kpi_router, prefix="", tags=["kpi"])

if __name__ == "__main__":
    import uvicorn
    # note: reload=True is for development only
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
