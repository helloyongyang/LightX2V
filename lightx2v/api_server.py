import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.infer import init_runner


class Message(BaseModel):
    prompt: str
    negative_prompt: str = ""
    image_path: str = ""
    save_video_path: str

    def get(self, key, default=None):
        return getattr(self, key, default)


async def main(message):
    runner.set_inputs(message)
    runner.run_pipeline()
    return {"response": "finished"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_causal"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"args: {args}")

    with ProfilingContext("Init Server Cost"):
        config = set_config(args)
        print(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = init_runner(config)

    app = FastAPI()

    @app.post("/v1/local/video/generate")
    async def generate_video(message: Message):
        response = await main(message)
        return response

    uvicorn.run(app, host="0.0.0.0", port=config.port)
