# 🎯 Attention Type Configuration in DiT Model

The DiT model in `LightX2V` currently uses three types of attention mechanisms. Each type of attention can be configured with a specific backend library.

---

## Attention Usage Locations

1. **Self-Attention on the image**
   - Configuration key: `self_attn_1_type`

2. **Cross-Attention between image and prompt text**
   - Configuration key: `cross_attn_1_type`

3. **Cross-Attention between image and reference image (in I2V mode)**
   - Configuration key: `cross_attn_2_type`

---

## 🚀 Supported Attention Backends

| Name               | Type Identifier   | GitHub Link |
|--------------------|-------------------|-------------|
| Flash Attention 2  | `flash_attn2`     | [flash-attention v2](https://github.com/Dao-AILab/flash-attention) |
| Flash Attention 3  | `flash_attn3`     | [flash-attention v3](https://github.com/Dao-AILab/flash-attention) |
| Sage Attention 2   | `sage_attn2`      | [SageAttention](https://github.com/thu-ml/SageAttention) |
| Radial Attention   | `radial_attn`     | [Radial Attention](https://github.com/mit-han-lab/radial-attention) |
| Sparge Attention   | `sparge_ckpt`     | [Sparge Attention](https://github.com/thu-ml/SpargeAttn) |

---

## 🛠️ Configuration Example

In the `wan_i2v.json` configuration file, you can specify the attention types as follows:

```json
{
  "self_attn_1_type": "radial_attn",
  "cross_attn_1_type": "flash_attn3",
  "cross_attn_2_type": "flash_attn3"
}
```

To use other attention backends, simply replace the values with the appropriate type identifiers listed above.

Tip: Due to the limitations of the sparse algorithm's principle, radial_attn can only be used in self-attention.

---

For Sparge Attention like `wan_t2v_sparge.json` configuration file:

   Sparge Attention need PostTrain weight path

```json
{
  "self_attn_1_type": "flash_attn3",
  "cross_attn_1_type": "flash_attn3",
  "cross_attn_2_type": "flash_attn3"
  "sparge": true,
  "sparge_ckpt": "/path/to/sparge_wan2.1_t2v_1.3B.pt"
}
```

---

For further customization or behavior tuning, please refer to the official documentation of the respective attention libraries.
