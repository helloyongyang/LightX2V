# Feature Caching

## Cache Acceleration Algorithms
- In the inference process of diffusion models, cache reuse is an important acceleration algorithm.
- Its core idea is to skip redundant computations at certain time steps and improve inference efficiency by reusing historical cache results.
- The key to the algorithm lies in how to decide at which time steps to perform cache reuse, usually based on dynamic judgment of model state changes or error thresholds.
- During inference, key content such as intermediate features, residuals, and attention outputs need to be cached. When entering a reusable time step, the cached content is directly utilized, and the current output is reconstructed through approximation methods like Taylor expansion, thereby reducing repetitive computations and achieving efficient inference.

### TeaCache
The core idea of `TeaCache` is to accumulate the **relative L1** distance between adjacent time step inputs. When the cumulative distance reaches the set threshold, it determines that the current time step should not use cache reuse; conversely, when the cumulative distance does not reach the set threshold, cache reuse is used to accelerate the inference process.
- Specifically, the algorithm calculates the relative L1 distance between the current input and the previous step's input at each inference step and accumulates it.
- When the cumulative distance does not exceed the threshold, it indicates that the model state change is not obvious, so the most recent cached content is directly reused, skipping some redundant computations. This can significantly reduce the number of forward computations of the model and improve inference speed.

In actual effectiveness, TeaCache achieves significant acceleration while ensuring generation quality. The video comparison before and after acceleration is as follows:

| Before Acceleration | After Acceleration |
|:------:|:------:|
| Single H200 inference time: 58s | Single H200 inference time: 17.9s |
| ![Before acceleration effect](../../../../assets/gifs/1.gif) | ![After acceleration effect](../../../../assets/gifs/2.gif) |
- Speedup ratio: **3.24**
- Config: [wan_t2v_1_3b_tea_480p.json](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/teacache/wan_t2v_1_3b_tea_480p.json)
- Reference paper: [https://arxiv.org/abs/2411.19108](https://arxiv.org/abs/2411.19108)

### TaylorSeer Cache
The core of `TaylorSeer Cache` lies in using Taylor formula to recalculate cached content as residual compensation for cache reuse time steps.
- The specific approach is that at cache reuse time steps, not only simply reusing historical cache, but also approximating reconstruction of current output through Taylor expansion. This can further improve output accuracy while reducing computational load.
- Taylor expansion can effectively capture subtle changes in model state, allowing errors caused by cache reuse to be compensated, thus ensuring generation quality while accelerating.

`TaylorSeer Cache` is suitable for scenarios with high output accuracy requirements and can further improve model inference performance based on cache reuse.

| Before Acceleration | After Acceleration |
|:------:|:------:|
| Single H200 inference time: 57.7s | Single H200 inference time: 41.3s |
| ![Before acceleration effect](../../../../assets/gifs/3.gif) | ![After acceleration effect](../../../../assets/gifs/4.gif) |
- Speedup ratio: **1.39**
- Config: [wan_t2v_taylorseer](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/taylorseer/wan_t2v_taylorseer.json)
- Reference paper: [https://arxiv.org/abs/2503.06923](https://arxiv.org/abs/2503.06923)

### AdaCache
The core idea of `AdaCache` is to dynamically adjust the stride of cache reuse based on partial cached content in specified block chunks.
- The algorithm analyzes feature differences between two adjacent time steps within specific blocks and adaptively determines the next cache reuse time step interval based on the difference magnitude.
- When model state changes are small, the stride automatically increases, reducing cache update frequency; when state changes are large, the stride decreases to ensure output quality.

This allows flexible adjustment of caching strategies based on dynamic changes during actual inference, achieving more efficient acceleration and better generation results. AdaCache is suitable for application scenarios with high requirements for both inference speed and generation quality.

| Before Acceleration | After Acceleration |
|:------:|:------:|
| Single H200 inference time: 227s | Single H200 inference time: 83s |
| ![Before acceleration effect](../../../../assets/gifs/5.gif) | ![After acceleration effect](../../../../assets/gifs/6.gif) |
- Speedup ratio: **2.73**
- Config: [wan_i2v_ada](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/adacache/wan_i2v_ada.json)
- Reference paper: [https://arxiv.org/abs/2411.02397](https://arxiv.org/abs/2411.02397)

### CustomCache
`CustomCache` combines the advantages of `TeaCache` and `TaylorSeer Cache`.
- It combines the real-time and rationality of `TeaCache` in cache decision-making, determining when to perform cache reuse through dynamic thresholds.
- At the same time, it utilizes `TaylorSeer`'s Taylor expansion method to make use of cached content.

This not only efficiently determines the timing of cache reuse but also maximally utilizes cached content to improve output accuracy and generation quality. Actual testing shows that `CustomCache` generates better video quality than solutions using `TeaCache`, `TaylorSeer Cache`, or `AdaCache` alone across multiple content generation tasks, making it one of the currently best-performing comprehensive cache acceleration algorithms.

| Before Acceleration | After Acceleration |
|:------:|:------:|
| Single H200 inference time: 57.9s | Single H200 inference time: 16.6s |
| ![Before acceleration effect](../../../../assets/gifs/7.gif) | ![After acceleration effect](../../../../assets/gifs/8.gif) |
- Speedup ratio: **3.49**
- Config: [wan_t2v_custom_1_3b](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/custom/wan_t2v_custom_1_3b.json)


## Usage

The config files for feature caching are [here](https://github.com/ModelTC/lightx2v/tree/main/configs/caching)

By specifying --config_json to a specific config file, you can test different cache algorithms.

[Here](https://github.com/ModelTC/lightx2v/tree/main/scripts/cache) are some running scripts for use.
