# 🎯 DiT 模型中的注意力类型配置说明

当前 DiT 模型在 `LightX2V` 中三个地方使用到了注意力，每个注意力可以分别配置底层注意力库类型。

---

## 使用注意力的位置

1. **图像的自注意力（Self-Attention）**
   - 配置参数：`self_attn_1_type`

2. **图像与提示词（Text）之间的交叉注意力（Cross-Attention）**
   - 配置参数：`cross_attn_1_type`

3. **I2V 模式下图像与参考图（Reference）之间的交叉注意力**
   - 配置参数：`cross_attn_2_type`

---

## 🚀 支持的注意力库（Backend）

| 名称               | 类型名称         | GitHub 链接 |
|--------------------|------------------|-------------|
| Flash Attention 2  | `flash_attn2`    | [flash-attention v2](https://github.com/Dao-AILab/flash-attention) |
| Flash Attention 3  | `flash_attn3`    | [flash-attention v3](https://github.com/Dao-AILab/flash-attention) |
| Sage Attention 2   | `sage_attn2`     | [SageAttention](https://github.com/thu-ml/SageAttention) |
| Radial Attention   | `radial_attn`    | [Radial Attention](https://github.com/mit-han-lab/radial-attention) |
| Sparge Attention   | `sparge_ckpt`     | [Sparge Attention](https://github.com/thu-ml/SpargeAttn) |

---

## 🛠️ 配置示例

在 `wan_i2v.json` 配置文件中，可以通过如下方式指定使用的注意力类型：

```json
{
  "self_attn_1_type": "radial_attn",
  "cross_attn_1_type": "flash_attn3",
  "cross_attn_2_type": "flash_attn3"
}
```

如需更换为其他类型，只需将对应值替换为上述表格中的类型名称即可。

tips: radial_attn因为稀疏算法原理的限制只能用在self attention

---

对于 Sparge Attention 配置参考 `wan_t2v_sparge.json` 文件:

    Sparge Attention是需要后一个训练的权重

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

如需进一步定制注意力机制的行为，请参考各注意力库的官方文档或实现代码。
