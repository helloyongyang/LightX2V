import signal
import sys
import psutil
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import uvicorn
import json
from typing import Optional
from datetime import datetime
import threading

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.infer import init_runner


# =========================
# Signal & Process Control
# =========================


def kill_all_related_processes():
    """Kill the current process and all its child processes"""
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except Exception as e:
            logger.info(f"Failed to kill child process {child.pid}: {e}")
    try:
        current_process.kill()
    except Exception as e:
        logger.info(f"Failed to kill main process: {e}")


def signal_handler(sig, frame):
    logger.info("\nReceived Ctrl+C, shutting down all related processes...")
    kill_all_related_processes()
    sys.exit(0)


# =========================
# FastAPI Related Code
# =========================

runner = None

app = FastAPI()


class Message(BaseModel):
    task_id: str
    task_id_must_unique: bool = False

    prompt: str
    use_prompt_enhancer: bool = False
    negative_prompt: str = ""
    image_path: str = ""
    num_fragments: int = 1
    save_video_path: str

    def get(self, key, default=None):
        return getattr(self, key, default)


class TaskStatusMessage(BaseModel):
    task_id: str


class ServiceStatus:
    _lock = threading.Lock()
    _current_task = None
    _result_store = {}

    @classmethod
    def start_task(cls, message: Message):
        with cls._lock:
            if cls._current_task is not None:
                raise RuntimeError("Service busy")
            if message.task_id_must_unique and message.task_id in cls._result_store:
                raise RuntimeError(f"Task ID {message.task_id} already exists")
            cls._current_task = {"message": message, "start_time": datetime.now()}
            return message.task_id

    @classmethod
    def complete_task(cls, message: Message):
        with cls._lock:
            cls._result_store[message.task_id] = {"success": True, "message": message, "start_time": cls._current_task["start_time"], "completion_time": datetime.now()}
            cls._current_task = None

    @classmethod
    def record_failed_task(cls, message: Message, error: Optional[str] = None):
        """Record a failed task with an error message."""
        with cls._lock:
            cls._result_store[message.task_id] = {"success": False, "message": message, "start_time": cls._current_task["start_time"], "error": error}
            cls._current_task = None

    @classmethod
    def get_status_task_id(cls, task_id: str):
        with cls._lock:
            if cls._current_task and cls._current_task["message"].task_id == task_id:
                return {"task_status": "processing"}
            if task_id in cls._result_store:
                return {"task_status": "completed", **cls._result_store[task_id]}
            return {"task_status": "not_found"}

    @classmethod
    def get_status_service(cls):
        with cls._lock:
            if cls._current_task:
                return {"service_status": "busy", "task_id": cls._current_task["message"].task_id}
            return {"service_status": "idle"}

    @classmethod
    def get_all_tasks(cls):
        with cls._lock:
            return cls._result_store


def local_video_generate(message: Message):
    try:
        global runner
        runner.set_inputs(message)
        logger.info(f"message: {message}")
        runner.run_pipeline()
        ServiceStatus.complete_task(message)
    except Exception as e:
        logger.error(f"task_id {message.task_id} failed: {str(e)}")
        ServiceStatus.record_failed_task(message, error=str(e))


@app.post("/v1/local/video/generate")
async def v1_local_video_generate(message: Message):
    try:
        task_id = ServiceStatus.start_task(message)
        # Use background threads to perform long-running tasks
        threading.Thread(target=local_video_generate, args=(message,), daemon=True).start()
        return {"task_id": task_id, "task_status": "processing", "save_video_path": message.save_video_path}
    except RuntimeError as e:
        return {"error": str(e)}


@app.get("/v1/local/video/generate/service_status")
async def get_service_status():
    return ServiceStatus.get_status_service()


@app.get("/v1/local/video/generate/get_all_tasks")
async def get_all_tasks():
    return ServiceStatus.get_all_tasks()


@app.post("/v1/local/video/generate/task_status")
async def get_task_status(message: TaskStatusMessage):
    return ServiceStatus.get_status_task_id(message.task_id)


# TODO: Implement delete task. Stop the specified task and clean many things.
# @app.delete("/v1/local/video/generate/task_status")
# async def delete_task(message: TaskStatusMessage):


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_causvid", "wan2.1_skyreels_v2_df"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--prompt_enhancer", default=None)

    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Init Server Cost"):
        config = set_config(args)
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = init_runner(config)

    uvicorn.run(app, host="0.0.0.0", port=config.port, reload=False, workers=1)
