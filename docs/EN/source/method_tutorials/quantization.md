# Model Quantization

lightx2v supports quantized inference for linear layers in **Dit**, enabling `w8a8-int8`, `w8a8-fp8`, `w8a8-fp8block`, `w8a8-mxfp8` and `w4a4-nvfp4` matrix multiplication.


## Generating Quantized Models

### Offline Quantization

lightx2v also supports direct loading of pre-quantized weights. For offline model quantization, refer to the [documentation](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme.md).
Configure the [quantization file](https://github.com/ModelTC/lightx2v/tree/main/configs/quantization/wan_i2v_quant_offline.json):
1. Set `dit_quantized_ckpt` to the converted weight path
2. Set `weight_auto_quant` to `false` in `mm_type`


## Quantized Inference

### Automatic Quantization
```shell
bash scripts/run_wan_i2v_quant_auto.sh
```

### Offline Quantization
```shell
bash scripts/run_wan_i2v_quant_offline.sh

```

## Launching Quantization Service


After offline quantization, point `--config_json` to the offline quantization JSON file.

Example modification in `scripts/start_server.sh`:

```shell
export RUNNING_FLAG=infer

python -m lightx2v.api_server \
--model_cls wan2.1 \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/quantization/wan_i2v_quant_offline.json \
--port 8000
```

## Advanced Quantization Features

Refer to the quantization tool [LLMC documentation](https://github.com/ModelTC/llmc/blob/main/docs/en/source/backend/lightx2v.md) for details.
