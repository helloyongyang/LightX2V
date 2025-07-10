# Feature Caching

## Cache Acceleration Algorithm
- Cache reuse is an important acceleration algorithm in the inference process of diffusion models.
- Its core idea is to skip redundant computations at certain time steps by reusing historical cache results to improve inference efficiency.
- The key to the algorithm is how to decide at which time steps to perform cache reuse, usually based on dynamic judgment of model state changes or error thresholds.
- During inference, key content such as intermediate features, residuals, and attention outputs need to be cached. When entering reusable time steps, directly use the cached content and reconstruct the current output through approximation methods like Taylor expansion, thereby reducing repetitive calculations and achieving efficient inference.

### TeaCache
The core idea of `TeaCache` is to accumulate the **relative L1** distance between adjacent time step inputs, and when the cumulative distance reaches a set threshold, determine that the current time step can perform cache reuse.
- Specifically, the algorithm calculates the relative L1 distance between the current input and the previous step input at each inference step, and accumulates it.
- When the cumulative distance exceeds the threshold, indicating that the model state has changed sufficiently, it directly reuses the most recently cached content, skipping some redundant computations. This can significantly reduce the number of forward computations of the model and improve inference speed.

In actual effect, TeaCache achieves significant acceleration while ensuring generation quality. The video comparison before and after acceleration is as follows:

| Before Acceleration | After Acceleration |
|:------:|:------:|
| Single H200 inference time: 58s | Single H200 inference time: 17.9s |
| ![Effect before acceleration](../../../../assets/gifs/1.gif) | ![Effect after acceleration](../../../../assets/gifs/2.gif) |
- Speedup ratio: **3.24**
- config：[wan_t2v_1_3b_tea_480p.json](../../../../configs/caching/teacache/wan_t2v_1_3b_tea_480p.json)
- Reference paper: [https://arxiv.org/abs/2411.19108](https://arxiv.org/abs/2411.19108)

### TaylorSeer Cache
The core of `TaylorSeer Cache` lies in using Taylor formula to recalculate cached content as residual compensation for cache reuse time steps. The specific approach is that at cache reuse time steps, not only simply reuse historical cache, but also approximate reconstruction of current output through Taylor expansion. This can further improve output accuracy while reducing computational load. Taylor expansion can effectively capture subtle changes in model state, compensating for errors brought by cache reuse, thereby ensuring generation quality while accelerating. `TaylorSeer Cache` is suitable for scenarios with high requirements for output precision, and can further improve model inference performance on the basis of cache reuse.

| Before Acceleration | After Acceleration |
|:------:|:------:|
| Single H200 inference time: 57.7s | Single H200 inference time: 41.3s |
| ![Effect before acceleration](../../../../assets/gifs/3.gif) | ![Effect after acceleration](../../../../assets/gifs/4.gif) |
- Speedup ratio: **1.39**
- config：[wan_t2v_taylorseer](../../../../configs/caching/taylorseer/wan_t2v_taylorseer.json)
- Reference paper: [https://arxiv.org/abs/2503.06923](https://arxiv.org/abs/2503.06923)

### AdaCache
The core idea of `AdaCache` is to dynamically adjust the step size of cache reuse based on partial cached content in specified block chunks.
- The algorithm analyzes feature differences between two adjacent time steps within specific blocks, and adaptively decides the next cache reuse time step interval based on the difference magnitude.
- When model state changes are small, the step size automatically increases, reducing cache update frequency; when state changes are large, the step size decreases to ensure output quality.

This allows flexible adjustment of cache strategies based on dynamic changes in the actual inference process, achieving more efficient acceleration and better generation effects. AdaCache is suitable for application scenarios with high requirements for both inference speed and generation quality.

| Before Acceleration | After Acceleration |
|:------:|:------:|
| Single H200 inference time: 227s | Single H200 inference time: 83s |
| ![Effect before acceleration](../../../../assets/gifs/5.gif) | ![Effect after acceleration](../../../../assets/gifs/6.gif) |
- Speedup ratio: **2.73**
- config：[wan_i2v_ada](../../../../configs/caching/adacache/wan_i2v_ada.json)
- Reference paper: [https://arxiv.org/abs/2411.02397](https://arxiv.org/abs/2411.02397)

### CustomCache
`CustomCache` combines the advantages of `TeaCache` and `TaylorSeer Cache`.
- It combines the real-time and rationality of `TeaCache` in cache decision-making, determining when to perform cache reuse through dynamic thresholds.
- At the same time, it utilizes `TaylorSeer`'s Taylor expansion method to make use of cached content.

This not only efficiently determines the timing of cache reuse, but also maximizes the utilization of cached content, improving output accuracy and generation quality. Actual tests show that `CustomCache` generates video quality superior to using `TeaCache`, `TaylorSeer Cache`, or `AdaCache` alone across multiple content generation tasks, making it one of the currently optimal comprehensive performance cache acceleration algorithms.

| Before Acceleration | After Acceleration |
|:------:|:------:|
| Single H200 inference time: 57.9s | Single H200 inference time: 16.6s |
| ![Effect before acceleration](../../../../assets/gifs/7.gif) | ![Effect after acceleration](../../../../assets/gifs/8.gif) |
- Speedup ratio: **3.49**
- config：[wan_t2v_custom_1_3b](../../../../configs/caching/custom/wan_t2v_custom_1_3b.json)


## How to Run

The config files for feature caching are available [here](https://github.com/ModelTC/lightx2v/tree/main/configs/caching)

By specifying --config_json to the specific config file, you can test different cache algorithms.

[Here](https://github.com/ModelTC/lightx2v/tree/main/scripts/cache) are some running scripts for use.
