<div align="center" style="font-family: charter;">
  <h1>⚡️ LightX2V：<br>轻量级视频生成推理框架</h1>

<img alt="logo" src="assets/img_lightx2v.png" width=75%></img>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ModelTC/lightx2v)
[![GitHub Stars](https://img.shields.io/github/stars/ModelTC/lightx2v.svg?style=social&label=Star&maxAge=60)](https://github.com/ModelTC/lightx2v)
![visitors](https://komarev.com/ghpvc/?username=lightx2v&label=visitors)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://lightx2v-en.readthedocs.io/en/latest)
[![Doc](https://img.shields.io/badge/文档-中文-99cc2)](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest)
[![Docker](https://badgen.net/badge/icon/docker?icon=docker&label)](https://hub.docker.com/r/lightx2v/lightx2v/tags)

**\[ [English](README.md) | 中文 | [日本語](README_ja.md) \]**
</div>

**LightX2V** 是一款轻量级视频生成推理框架，集成多种先进的视频生成推理技术，统一支持 文本生成视频 (T2V)、图像生成视频 (I2V) 等多种生成任务及模型。**“X2V” 表示将不同输入模态（文本、图像等）转换为视频输出。**

## 💡 快速开始

请参考文档：**[English Docs](https://lightx2v-en.readthedocs.io/en/latest/)** | **[中文文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/)**

## 🤖 支持的模型列表

- ✅ [HunyuanVideo-T2V](https://huggingface.co/tencent/HunyuanVideo)
- ✅ [HunyuanVideo-I2V](https://huggingface.co/tencent/HunyuanVideo-I2V)
- ✅ [Wan2.1-T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- ✅ [Wan2.1-I2V](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
- ✅ [Wan2.1-T2V-StepDistill-CfgDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)
- ✅ [Wan2.1-T2V-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)
- ✅ [SkyReels-V2-DF](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P)
- ✅ [CogVideoX1.5-5B-T2V](https://huggingface.co/THUDM/CogVideoX1.5-5B)

## 🧾 贡献指南

我们使用 `pre-commit` 统一代码格式。

> [!Tip]
> - 下载需要的依赖:
>
> ```shell
> pip install ruff pre-commit
>```
>
> - 然后，再提交前运行下述指令:
>
> ```shell
> pre-commit run --all-files
>```

欢迎贡献！

## 🤝 致谢

本仓库实现参考了以上列出的所有模型对应的代码仓库。

## 🌟 Star 记录

[![Star History Chart](https://api.star-history.com/svg?repos=ModelTC/lightx2v&type=Timeline)](https://star-history.com/#ModelTC/lightx2v&Timeline)

## ✏️ 引用

如果您觉得本框架对您的研究有帮助，请引用：

```bibtex
@misc{lightx2v,
  author = {lightx2v contributors},
  title  = {LightX2V: Light Video Generation Inference Framework},
  year   = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ModelTC/lightx2v}},
}
```
