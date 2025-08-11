import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from loguru import logger

from .api import ApiServer
from .config import server_config
from .service import DistributedInferenceService


class ServerManager:
    def __init__(self):
        self.api_server: Optional[ApiServer] = None
        self.inference_service: Optional[DistributedInferenceService] = None
        self.shutdown_event = asyncio.Event()

    async def startup(self, args):
        logger.info("Starting LightX2V server...")

        if hasattr(args, "host") and args.host:
            server_config.host = args.host
        if hasattr(args, "port") and args.port:
            server_config.port = args.port

        if not server_config.validate():
            raise RuntimeError("Invalid server configuration")

        self.inference_service = DistributedInferenceService()
        if not self.inference_service.start_distributed_inference(args):
            raise RuntimeError("Failed to start distributed inference service")

        cache_dir = Path(server_config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.api_server = ApiServer(max_queue_size=server_config.max_queue_size)
        self.api_server.initialize_services(cache_dir, self.inference_service)

        logger.info("Server startup completed successfully")

    async def shutdown(self):
        logger.info("Starting server shutdown...")

        if self.api_server:
            await self.api_server.cleanup()
            logger.info("API server cleaned up")

        if self.inference_service:
            self.inference_service.stop_distributed_inference()
            logger.info("Inference service stopped")

        logger.info("Server shutdown completed")

    def handle_signal(self, sig, frame):
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown())
        self.shutdown_event.set()

    async def run_server(self, args):
        try:
            await self.startup(args)

            assert self.api_server is not None

            app = self.api_server.get_app()

            signal.signal(signal.SIGINT, self.handle_signal)
            signal.signal(signal.SIGTERM, self.handle_signal)

            logger.info(f"Starting server on {server_config.host}:{server_config.port}")
            config = uvicorn.Config(
                app=app,
                host=server_config.host,
                port=server_config.port,
                log_level="info",
            )
            server = uvicorn.Server(config)

            server_task = asyncio.create_task(server.serve())

            await self.shutdown_event.wait()

            server.should_exit = True
            await server_task

        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            await self.shutdown()


def run_server(args):
    inference_service = None
    try:
        logger.info("Starting LightX2V server...")

        if hasattr(args, "host") and args.host:
            server_config.host = args.host
        if hasattr(args, "port") and args.port:
            server_config.port = args.port

        if not server_config.validate():
            raise RuntimeError("Invalid server configuration")

        inference_service = DistributedInferenceService()
        if not inference_service.start_distributed_inference(args):
            raise RuntimeError("Failed to start distributed inference service")
        logger.info("Inference service started successfully")

        cache_dir = Path(server_config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        api_server = ApiServer(max_queue_size=server_config.max_queue_size)
        api_server.initialize_services(cache_dir, inference_service)

        app = api_server.get_app()

        logger.info(f"Starting server on {server_config.host}:{server_config.port}")
        uvicorn.run(app, host=server_config.host, port=server_config.port, log_level="info")

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        if inference_service:
            inference_service.stop_distributed_inference()
    except Exception as e:
        logger.error(f"Server failed: {e}")
        if inference_service:
            inference_service.stop_distributed_inference()
        sys.exit(1)
