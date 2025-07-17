# Benchmark

---

## H200 (~140GB VRAM)

**Software Environment:**
- Python 3.11
- PyTorch 2.7.1+cu128
- SageAttention 2.2.0
- vLLM 0.9.2
- sgl-kernel 0.1.8

### 480P 5s Video

**Test Configuration:**
- **Model**: [Wan2.1-I2V-14B-480P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-Lightx2v)
- **Parameters**: infer_steps=40, seed=42, enable_cfg=True

#### Performance Comparison

| Configuration | Model Load Time(s) | Inference Time(s) | GPU Memory(GB) | Speedup | Video Effect |
|:-------------|:------------------:|:-----------------:|:--------------:|:-------:|:------------:|
| Wan2.1 Official(baseline) | 68.26 | 366.04 | 71 | 1.0x | <video src="PATH_TO_BASELINE_480P_VIDEO" width="200px"></video> |
| **LightX2V_1** | 37.28 | 249.54 | 53 | **1.47x** | <video src="PATH_TO_LIGHTX2V_1_480P_VIDEO" width="200px"></video> |
| **LightX2V_2** | 37.24 | 216.16 | 50 | **1.69x** | <video src="PATH_TO_LIGHTX2V_2_480P_VIDEO" width="200px"></video> |
| **LightX2V_3** | 23.62 | 190.73 | 35 | **1.92x** | <video src="PATH_TO_LIGHTX2V_3_480P_VIDEO" width="200px"></video> |
| **LightX2V_4** | 23.62 | 107.19 | 35 | **3.41x** | <video src="PATH_TO_LIGHTX2V_4_480P_VIDEO" width="200px"></video> |
| **LightX2V_4-Distill** | 23.62 | 107.19 | 35 | **3.41x** | <video src="PATH_TO_LIGHTX2V_4_DISTILL_480P_VIDEO" width="200px"></video> |

### 720P 5s Video

**Test Configuration:**
- **Model**: [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v)
- **Parameters**: infer_steps=40, seed=42, enable_cfg=True

*Coming soon...*

---

## RTX 4090 (~24GB VRAM)

### 480P 5s Video

*Coming soon...*

### 720P 5s Video

*Coming soon...*

---

## Table Descriptions

- **Wan2.1 Official(baseline)**: Baseline implementation based on [Wan2.1 official repository](https://github.com/Wan-Video/Wan2.1)
- **LightX2V_1**: Uses SageAttention2 to replace native attention mechanism with DIT BF16+FP32 mixed precision (sensitive layers), improving computational efficiency while maintaining precision
- **LightX2V_2**: Unified BF16 precision computation to further reduce memory usage and computational overhead while maintaining generation quality
- **LightX2V_3**: Quantization optimization introducing FP8 quantization technology to significantly reduce computational precision requirements, combined with Tiling VAE technology to optimize memory usage
- **LightX2V_4**: Ultimate optimization adding TeaCache (teacache_thresh=0.2) caching reuse technology on top of LightX2V_3 to achieve maximum acceleration by intelligently skipping redundant computations
- **LightX2V_4-Distill**: Building on LightX2V_4 with 4-step distilled model ([Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v))
