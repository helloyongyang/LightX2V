<div align="center" style="font-family: charter;">
  <h1>⚡️ LightX2V:<br> Light Video Generation Inference Framework</h1>

<img alt="logo" src="assets/img_lightx2v.png" width=75%></img>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ModelTC/lightx2v)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://lightx2v-en.readthedocs.io/en/latest)
[![Doc](https://img.shields.io/badge/文档-中文-99cc2)](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest)
[![Papers](https://img.shields.io/badge/论文集-中文-99cc2)](https://lightx2v-papers-zhcn.readthedocs.io/zh-cn/latest)
[![Docker](https://badgen.net/badge/icon/docker?icon=docker&label)](https://hub.docker.com/r/lightx2v/lightx2v/tags)

**\[ English | [中文](README_zh.md) \]**

</div>

--------------------------------------------------------------------------------

**LightX2V** is a lightweight video generation inference framework designed to provide an inference tool that leverages multiple advanced video generation inference techniques. As a unified inference platform, this framework supports various generation tasks such as text-to-video (T2V) and image-to-video (I2V) across different models. **X2V means transforming different input modalities (such as text or images) to video output.**


## 💡 How to Start

Please refer to our documentation: **[English Docs](https://lightx2v-en.readthedocs.io/en/latest/) | [中文文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/)**.


## 🤖 Supported Model List

- ✅ [HunyuanVideo-T2V](https://huggingface.co/tencent/HunyuanVideo)
- ✅ [HunyuanVideo-I2V](https://huggingface.co/tencent/HunyuanVideo-I2V)
- ✅ [Wan2.1-T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- ✅ [Wan2.1-I2V](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
- ✅ [Wan2.1-T2V-StepDistill-CfgDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill) (recommended 🚀🚀🚀)
- ✅ [Wan2.1-T2V-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)
- ✅ [SkyReels-V2-DF](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P)
- ✅ [CogVideoX1.5-5B-T2V](https://huggingface.co/THUDM/CogVideoX1.5-5B)


## 🧾 Contributing Guidelines

We have prepared a pre-commit hook to enforce consistent code formatting across the project.

> [!TIP]
> - Install the required dependencies:
>
> ```shell
> pip install ruff pre-commit
>```
>
> - Then, run the following command before commit:
>
> ```shell
> pre-commit run --all-files
>```


Thank you for your contributions!


## 🤝 Acknowledgments

We built the code for this repository by referencing the code repositories involved in all the models mentioned above.


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ModelTC/lightx2v&type=Timeline)](https://star-history.com/#ModelTC/llmc&Timeline)


## ✏️ Citation

If you find our framework useful to your research, please kindly cite our work:

```
@misc{lightx2v,
 author = {lightx2v contributors},
 title = {LightX2V: Light Video Generation Inference Framework},
 year = {2025},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/ModelTC/lightx2v}},
}
```
