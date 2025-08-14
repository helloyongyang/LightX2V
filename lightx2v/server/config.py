import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_queue_size: int = 10

    master_addr: str = "127.0.0.1"
    master_port_range: tuple = (29500, 29600)

    task_timeout: int = 300
    task_history_limit: int = 1000

    http_timeout: int = 30
    http_max_retries: int = 3

    cache_dir: str = str(Path(__file__).parent.parent / "server_cache")
    max_upload_size: int = 500 * 1024 * 1024  # 500MB

    @classmethod
    def from_env(cls) -> "ServerConfig":
        config = cls()

        if env_host := os.environ.get("LIGHTX2V_HOST"):
            config.host = env_host

        if env_port := os.environ.get("LIGHTX2V_PORT"):
            try:
                config.port = int(env_port)
            except ValueError:
                logger.warning(f"Invalid port in environment: {env_port}")

        if env_queue_size := os.environ.get("LIGHTX2V_MAX_QUEUE_SIZE"):
            try:
                config.max_queue_size = int(env_queue_size)
            except ValueError:
                logger.warning(f"Invalid max queue size: {env_queue_size}")

        if env_master_addr := os.environ.get("MASTER_ADDR"):
            config.master_addr = env_master_addr

        if env_cache_dir := os.environ.get("LIGHTX2V_CACHE_DIR"):
            config.cache_dir = env_cache_dir

        return config

    def find_free_master_port(self) -> str:
        import socket

        for port in range(self.master_port_range[0], self.master_port_range[1]):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.master_addr, port))
                    logger.info(f"Found free port for master: {port}")
                    return str(port)
            except OSError:
                continue

        raise RuntimeError(
            f"No free port found for master in range {self.master_port_range[0]}-{self.master_port_range[1] - 1} "
            f"on address {self.master_addr}. Please adjust 'master_port_range' or free an occupied port."
        )

    def validate(self) -> bool:
        valid = True

        if self.max_queue_size <= 0:
            logger.error("max_queue_size must be positive")
            valid = False

        if self.task_timeout <= 0:
            logger.error("task_timeout must be positive")
            valid = False

        return valid


server_config = ServerConfig.from_env()
