# 基准测试

---

## H200 (~140GB显存)

**软件环境配置:**
- Python 3.11
- PyTorch 2.7.1+cu128
- SageAttention 2.2.0
- vLLM 0.9.2
- sgl-kernel 0.1.8

### 480P 5s视频

**测试配置:**
- **模型**: [Wan2.1-I2V-14B-480P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-Lightx2v)
- **参数**: infer_steps=40, seed=42, enable_cfg=True

#### 性能对比

| 配置 | 模型加载时间(s) | 推理时间(s) | GPU显存占用(GB) | 加速比 | 视频效果 |
|:-----|:---------------:|:----------:|:---------------:|:------:|:--------:|
| Wan2.1 Official(baseline) | 68.26 | 366.04 | 71 | 1.0x | <video src="PATH_TO_BASELINE_480P_VIDEO" width="200px"></video> |
| **LightX2V_1** | 37.28 | 249.54 | 53 | **1.47x** | <video src="PATH_TO_LIGHTX2V_1_480P_VIDEO" width="200px"></video> |
| **LightX2V_2** | 37.24 | 216.16 | 50 | **1.69x** | <video src="PATH_TO_LIGHTX2V_2_480P_VIDEO" width="200px"></video> |
| **LightX2V_3** | 23.62 | 190.73 | 35 | **1.92x** | <video src="PATH_TO_LIGHTX2V_3_480P_VIDEO" width="200px"></video> |
| **LightX2V_4** | 23.62 | 107.19 | 35 | **3.41x** | <video src="PATH_TO_LIGHTX2V_4_480P_VIDEO" width="200px"></video> |

### 720P 5s视频

**测试配置:**
- **模型**: [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v)
- **参数**: infer_steps=40, seed=42, enable_cfg=True

*即将更新...*

---

## RTX 4090 (~24GB显存)

### 480P 5s视频

*即将更新...*

### 720P 5s视频

*即将更新...*

---

## 表格说明

- **Wan2.1 Official(baseline)**: 基于[Wan2.1官方仓库](https://github.com/Wan-Video/Wan2.1)的基线实现
- **LightX2V_1**: 使用SageAttention2替换原生注意力机制，采用DIT BF16+FP32(部分敏感层)混合精度计算，在保持精度的同时提升计算效率
- **LightX2V_2**: 统一使用BF16精度计算，进一步减少显存占用和计算开销，同时保持生成质量
- **LightX2V_3**: 引入FP8量化技术显著减少计算精度要求，结合Tiling VAE技术优化显存使用
- **LightX2V_4**: 在LightX2V_3基础上加入TeaCache(teacache_thresh=0.2)缓存复用技术，通过智能跳过冗余计算实现最大加速
