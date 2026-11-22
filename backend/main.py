"""FastAPI application entry point.

    uvicorn backend.main:app --reload --port 8000
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import services
from backend.database import SessionLocal, init_db
from backend.routes import APP_VERSION, router
from backend.security import get_secret_key
from ml.predict import ModelNotFoundError, get_predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("upi_sentinel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_secret_key()  # fail fast if SECRET_KEY is missing
    init_db()
    app.state.model_error = None
    with SessionLocal() as db:
        created = services.bootstrap_users(db)
        if created:
            log.info("Created bootstrap users: %s", ", ".join(created))
        try:
            predictor = get_predictor()  # model is loaded once, at startup
            services.sync_model_performance(db, predictor.metadata)
            log.info("Model loaded: %s", predictor.model_name)
        except ModelNotFoundError as exc:
            app.state.model_error = str(exc)
            log.error("Model not available: %s", exc)
    yield


app = FastAPI(
    title="UPI Sentinel AI - Fraud Detection API",
    description=("Intelligent Digital Payment Fraud Detection & Risk Intelligence.\n\n"
                 "**Academic decision-support system.** Trained on synthetic data; it neither connects to "
                 "banks nor blocks / initiates payments. Authorise in Swagger with the token returned by "
                 "`POST /auth/login`."),
    version=APP_VERSION, lifespan=lifespan)

origins = [o.strip() for o in os.getenv(
    "FRONTEND_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
app.include_router(router)


@app.exception_handler(ModelNotFoundError)
async def model_missing_handler(_: Request, exc: ModelNotFoundError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
