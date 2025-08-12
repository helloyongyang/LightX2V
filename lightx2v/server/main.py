import sys
from pathlib import Path

import uvicorn
from loguru import logger

from .api import ApiServer
from .config import server_config
from .service import DistributedInferenceService


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
