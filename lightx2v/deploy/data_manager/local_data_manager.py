import asyncio
import os

from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch_async
from lightx2v.deploy.data_manager import BaseDataManager


class LocalDataManager(BaseDataManager):
    def __init__(self, local_dir):
        self.local_dir = local_dir
        self.name = "local"
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)

    @class_try_catch_async
    async def save_bytes(self, bytes_data, filename):
        out_path = os.path.join(self.local_dir, filename)
        with open(out_path, "wb") as fout:
            fout.write(bytes_data)
            return True

    @class_try_catch_async
    async def load_bytes(self, filename):
        inp_path = os.path.join(self.local_dir, filename)
        with open(inp_path, "rb") as fin:
            return fin.read()

    @class_try_catch_async
    async def delete_bytes(self, filename):
        inp_path = os.path.join(self.local_dir, filename)
        os.remove(inp_path)
        logger.info(f"deleted local file {filename}")
        return True


async def test():
    import torch
    from PIL import Image

    m = LocalDataManager("/data/nvme1/liuliang1/lightx2v/local_data")
    await m.init()

    img = Image.open("/data/nvme1/liuliang1/lightx2v/assets/img_lightx2v.png")
    tensor = torch.Tensor([233, 456, 789]).to(dtype=torch.bfloat16, device="cuda:0")

    await m.save_image(img, "test_img.png")
    print(await m.load_image("test_img.png"))

    await m.save_tensor(tensor, "test_tensor.pt")
    print(await m.load_tensor("test_tensor.pt", "cuda:0"))

    await m.save_object(
        {
            "images": [img, img],
            "tensor": tensor,
            "list": [
                [2, 0, 5, 5],
                {
                    "1": "hello world",
                    "2": "world",
                    "3": img,
                    "t": tensor,
                },
                "0609",
            ],
        },
        "test_object.json",
    )
    print(await m.load_object("test_object.json", "cuda:0"))

    await m.get_delete_func("OBJECT")("test_object.json")
    await m.get_delete_func("TENSOR")("test_tensor.pt")
    await m.get_delete_func("IMAGE")("test_img.png")


if __name__ == "__main__":
    asyncio.run(test())
