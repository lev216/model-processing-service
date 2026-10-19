import logging
from contextlib import asynccontextmanager

import logging_loki
import structlog
import uvicorn
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.database import close_engine, create_tables
from app.router import router

SERVICE_LOGGER_NAME = "model_processing_service"


def configure_logging():
    root_logger = logging.getLogger()
    existing_service_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "_model_processing_service_handler", False)
    ]
    for handler in existing_service_handlers:
        root_logger.removeHandler(handler)
        handler.close()

    loki_handler = logging_loki.LokiHandler(
        url=f"{settings.loki_url}/loki/api/v1/push",
        tags={"service": SERVICE_LOGGER_NAME},
        version="1",
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer()
            if settings.log_level == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler._model_processing_service_handler = True

    loki_handler._model_processing_service_handler = True
    root_logger.addHandler(handler)
    root_logger.addHandler(loki_handler)
    root_logger.setLevel(logging.getLevelName(settings.log_level))

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log = structlog.get_logger(__name__)

    await create_tables()

    log.info("Model processing service started")
    log.info(f"Models directory: {settings.models_dir}")
    yield
    await close_engine()
    log.info("Model processing service stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Model Processing Service",
        description=("The service to work with models"),
        version="0.1.0",
        lifespan=lifespan,
        root_path="/model",
    )
    app.include_router(router)
    return app


app = create_app()
Instrumentator().instrument(app).expose(app)


def run():
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
