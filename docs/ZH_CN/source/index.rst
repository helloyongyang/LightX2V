欢迎了解 Lightx2v!
==================

.. figure:: ../../../assets/img_lightx2v.png
  :width: 100%
  :align: center
  :alt: Lightx2v
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>一个轻量级的视频生成推理框架
   </strong>


LightX2V 是一个轻量级的视频生成推理框架，旨在提供一个利用多种先进的视频生成推理技术的推理工具。该框架作为统一的推理平台，支持不同模型的文本到视频（T2V）和图像到视频（I2V）等生成任务。X2V 表示将不同的输入模态（X，如文本或图像）转换（to）为视频输出（V）。

文档列表
-------------

.. toctree::
   :maxdepth: 1
   :caption: 快速入门

   快速入门 <getting_started/quickstart.md>

.. toctree::
   :maxdepth: 1
   :caption: 方法教程

   模型量化 <method_tutorials/quantization.md>
   特征缓存 <method_tutorials/cache.md>
   注意力机制 <method_tutorials/attention.md>
   参数卸载 <method_tutorials/offload.md>
   并行推理 <method_tutorials/parallel.md>
   步数蒸馏 <method_tutorials/step_distill.md>
   自回归蒸馏 <method_tutorials/autoregressive_distill.md>

.. toctree::
   :maxdepth: 1
   :caption: 部署指南

   低延迟场景部署 <deploy_guides/for_low_latency.md>
   低资源场景部署 <deploy_guides/for_low_resource.md>
   Lora模型部署 <deploy_guides/lora_deploy.md>
   服务化部署 <deploy_guides/deploy_service.md>
   gradio部署 <deploy_guides/deploy_gradio.md>
   comfyui部署 <deploy_guides/deploy_comfyui.md>
   本地windows电脑部署 <deploy_guides/deploy_local_windows.md>
