# Lightx2v Parameter Offloading Mechanism Documentation

## 📖 Overview

Lightx2v implements an advanced parameter offloading mechanism designed for large model inference under limited hardware resources. This system provides excellent speed-memory balance through intelligent management of model weights across different memory hierarchies.

**Core Features:**
- **Block/Phase Offloading**: Efficiently manages model weights in block/phase units for optimal memory usage
  - **Block**: Basic computational unit of Transformer models, containing complete Transformer layers (self-attention, cross-attention, feed-forward networks, etc.), serving as larger memory management units
  - **Phase**: Finer-grained computational stages within blocks, containing individual computational components (such as self-attention, cross-attention, feed-forward networks, etc.), providing more precise memory control
- **Multi-level Storage Support**: GPU → CPU → Disk hierarchy with intelligent caching
- **Asynchronous Operations**: Uses CUDA streams to overlap computation and data transfer
- **Disk/NVMe Serialization**: Supports secondary storage when memory is insufficient

## 🎯 Offloading Strategies

### Strategy 1: GPU-CPU Block/Phase Offloading

**Applicable Scenarios**: GPU VRAM insufficient but system memory adequate

**Working Principle**: Manages model weights in block or phase units between GPU and CPU memory, utilizing CUDA streams to overlap computation and data transfer. Blocks contain complete Transformer layers, while phases are individual computational components within blocks.

**Block vs Phase Explanation**:
- **Block Granularity**: Larger memory management units containing complete Transformer layers (self-attention, cross-attention, feed-forward networks, etc.), suitable for memory-sufficient scenarios, reducing management overhead
- **Phase Granularity**: Finer-grained memory management containing individual computational components (such as self-attention, cross-attention, feed-forward networks, etc.), suitable for memory-constrained scenarios, providing more flexible memory control

```
GPU-CPU Block/Phase Offloading Workflow:

╔═════════════════════════════════════════════════════════════════╗
║                        🎯 GPU Memory                            ║
╠═════════════════════════════════════════════════════════════════╣
║                                                               ║
║  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ ║
║  │ 🔄 Current      │    │ ⏳ Prefetch     │    │ 📤 To Offload   │ ║
║  │ block/phase N   │◄──►│ block/phase N+1 │◄──►│ block/phase N-1 │ ║
║  └─────────────────┘    └─────────────────┘    └─────────────────┘ ║
║         │                       │                       │         ║
║         ▼                       ▼                       ▼         ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         ║
║  │ Compute     │    │ GPU Load    │    │ CPU Load    │         ║
║  │ Stream      │    │ Stream      │    │ Stream      │         ║
║  │(priority=-1)│   │ (priority=0) │   │ (priority=0) │         ║
║  └─────────────┘    └─────────────┘    └─────────────┘         ║
╚═════════════════════════════════════════════════════════════════╝
                              ↕
╔═════════════════════════════════════════════════════════════════╗
║                        💾 CPU Memory                            ║
╠═════════════════════════════════════════════════════════════════╣
║                                                               ║
║  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ║
║  │ 📥 Cache    │ │ 📥 Cache    │ │ 📥 Cache    │ │ 📥 Cache    │ ║
║  │ block/phase │ │ block/phase │ │ block/phase │ │ block/phase │ ║
║  │    N-2      │ │    N-1      │ │     N       │ │    N+1      │ ║
║  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ ║
║         ▲               ▲               ▲               ▲         ║
║         │               │               │               │         ║
║  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ║
║  │ CPU Load    │ │ CPU Load    │ │ CPU Load    │ │ CPU Load    │ ║
║  │ Stream      │ │ Stream      │ │ Stream      │ │ Stream      │ ║
║  │(priority=0) │ │(priority=0) │ │(priority=0) │ │(priority=0) │ ║
║  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ ║
║                                                               ║
║  💡 CPU memory stores multiple blocks/phases, forming cache pool ║
║  🔄 GPU load stream prefetches from CPU cache, CPU load stream  ║
║     offloads to CPU cache                                        ║
╚═════════════════════════════════════════════════════════════════╝


╔═════════════════════════════════════════════════════════════════╗
║                        🔄 Swap Operation Flow                   ║
╠═════════════════════════════════════════════════════════════════╣
║                                                               ║
║  Step 1: Parallel Execution Phase                              ║
║  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ ║
║  │ 🔄 Compute      │    │ ⏳ Prefetch     │    │ 📤 Offload      │ ║
║  │ block/phase N   │    │ block/phase N+1 │    │ block/phase N-1 │ ║
║  │ (Compute Stream)│    │ (GPU Load Stream)│   │ (CPU Load Stream)│ ║
║  └─────────────────┘    └─────────────────┘    └─────────────────┘ ║
║                                                               ║
║  Step 2: Swap Rotation Phase                                   ║
║  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ ║
║  │ 🔄 Compute      │    │ ⏳ Prefetch     │    │ 📤 Offload      │ ║
║  │ block/phase N+1 │    │ block/phase N+2 │    │ block/phase N   │ ║
║  │ (Compute Stream)│    │ (GPU Load Stream)│   │ (CPU Load Stream)│ ║
║  └─────────────────┘    └─────────────────┘    └─────────────────┘ ║
║                                                               ║
║  Swap Concept: Achieves continuous computation through position ║
║  rotation, avoiding repeated loading/unloading                  ║
╚═════════════════════════════════════════════════════════════════╝

╔═════════════════════════════════════════════════════════════════╗
║                        💡 Swap Core Concept                     ║
╠═════════════════════════════════════════════════════════════════╣
║                                                               ║
║  🔄 Traditional vs Swap Method Comparison:                     ║
║                                                               ║
║  Traditional Method:                                            ║
║  ┌─────────────┐    ┌──────────┐    ┌─────────┐    ┌────────┐ ║
║  │ Compute N   │───►│ Offload N│───►│ Load N+1│───►│Compute │ ║
║  │             │    │          │    │         │    │N+1     │ ║
║  └─────────────┘    └──────────┘    └─────────┘    └────────┘ ║
║       ❌ Serial execution, waiting time, low efficiency        ║
║                                                               ║
║  Swap Method:                                                  ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         ║
║  │ Compute N   │    │ Prefetch    │    │ Offload     │         ║
║  │(Compute     │    │N+1          │    │N-1          │         ║
║  │ Stream)     │    │(GPU Load    │    │(CPU Load    │         ║
║  └─────────────┘    │ Stream)     │    │ Stream)     │         ║
║                     └─────────────┘    └─────────────┘         ║
║       ✅ Parallel execution, no waiting time, high efficiency  ║
║                                                               ║
║  🎯 Swap Advantages:                                           ║
║  • Avoids repeated loading/unloading of same data              ║
║  • Achieves continuous computation through position rotation   ║
║  • Maximizes GPU utilization                                   ║
║  • Reduces memory fragmentation                                ║
╚════════════════════════════════════════════════════════════════╝
```

**Key Features:**
- **Asynchronous Transfer**: Uses three CUDA streams with different priorities to parallelize computation and transfer
  - Compute Stream (priority=-1): High priority, responsible for current computation
  - GPU Load Stream (priority=0): Medium priority, responsible for prefetching from CPU to GPU
  - CPU Load Stream (priority=0): Medium priority, responsible for offloading from GPU to CPU
- **Prefetch Mechanism**: Preloads the next block/phase to GPU
- **Intelligent Caching**: Maintains weight cache in CPU memory
- **Stream Synchronization**: Ensures correctness of data transfer and computation
- **Swap Operation**: Rotates block/phase positions after computation completion for continuous processing


### Strategy 2: Disk-CPU-GPU Block/Phase Offloading (Lazy Loading)

**Applicable Scenarios**: Both GPU VRAM and system memory insufficient

**Working Principle**: Introduces disk storage on top of Strategy 1, implementing a three-level storage hierarchy (Disk → CPU → GPU). CPU continues as a cache pool but with configurable size, suitable for CPU memory-constrained devices.

```
Disk-CPU-GPU Block/Phase Offloading Workflow:

╔═════════════════════════════════════════════════════════════════╗
║                        💿 SSD/NVMe Storage                     ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ║
║  │ 📁 block_0  │ │ 📁 block_1  │ │ 📁 block_2  │ │ 📁 block_N  │ ║
║  │ .safetensors│ │ .safetensors│ │ .safetensors│ │ .safetensors│ ║
║  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ ║
║         │               │               │               │         ║
║         ▼               ▼               ▼               ▼         ║
║  ┌─────────────────────────────────────────────────────────────┐ ║
║  │                    🎯 Disk Worker Thread Pool               │ ║
║  │                                                             │ ║
║  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │ ║
║  │  │ Disk Thread │ │ Disk Thread │ │ Disk Thread │            │ ║
║  │  │     1       │ │     2       │ │     N       │            │ ║
║  │  │(Async Load) │ │(Async Load) │ │(Async Load) │            │ ║
║  │  └─────────────┘ └─────────────┘ └─────────────┘            │ ║
║  │         │               │               │                   │ ║
║  │         └───────────────┼───────────────┘                   │ ║
║  │                         ▼                                   │ ║
║  │  ┌─────────────────────────────────────────────────────────┐ │ ║
║  │  │                 📋 Priority Task Queue                  │ │ ║
║  │  │              (Manages disk loading task scheduling)     │ │ ║
║  │  └─────────────────────────────────────────────────────────┘ │ ║
║  └─────────────────────────────────────────────────────────────┘ ║
╚═════════════════════════════════════════════════════════════════╝
                             ↓
╔═════════════════════════════════════════════════════════════════╗
║                        💾 CPU Memory Buffer                     ║
╠═════════════════════════════════════════════════════════════════╣
║                                                               ║
║  ┌─────────────────────────────────────────────────────────────┐ ║
║  │                    🎯 FIFO Intelligent Cache                │ ║
║  │                                                             │ ║
║  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ║
║  │  │ 📥 Cache    │ │ 📥 Cache    │ │ 📥 Cache    │ │ 📥 Cache    │ ║
║  │  │ block/phase │ │ block/phase │ │ block/phase │ │ block/phase │ ║
║  │  │    N-2      │ │    N-1      │ │     N       │ │    N+1      │ ║
║  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ ║
║  │         ▲               ▲               ▲               ▲         ║
║  │         │               │               │               │         ║
║  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ║
║  │  │ CPU Load    │ │ CPU Load    │ │ CPU Load    │ │ CPU Load    │ ║
║  │  │ Stream      │ │ Stream      │ │ Stream      │ │ Stream      │ ║
║  │  │(priority=0) │ │(priority=0) │ │(priority=0) │ │(priority=0) │ ║
║  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ ║
║  │                                                             │ ║
║  │  💡 Configurable Size 🎯 FIFO Eviction 🔄 Cache Hit/Miss    │ ║
║  └─────────────────────────────────────────────────────────────┘ ║
╚═════════════════════════════════════════════════════════════════╝
                             ↕
╔═════════════════════════════════════════════════════════════════╗
║                        🎯 GPU Memory                            ║
╠═════════════════════════════════════════════════════════════════╣
║                                                               ║
║  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ ║
║  │ 🔄 Current      │    │ ⏳ Prefetch     │    │ 📤 To Offload   │ ║
║  │ block/phase N   │◄──►│ block/phase N+1 │◄──►│ block/phase N-1 │ ║
║  └─────────────────┘    └─────────────────┘    └─────────────────┘ ║
║         │                       │                       │         ║
║         ▼                       ▼                       ▼         ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         ║
║  │ Compute     │    │ GPU Load    │    │ CPU Load    │         ║
║  │ Stream      │    │ Stream      │    │ Stream      │         ║
║  │(priority=-1)│   │ (priority=0) │   │ (priority=0) │         ║
║  └─────────────┘    └─────────────┘    └─────────────┘         ║
╚═════════════════════════════════════════════════════════════════╝

╔═════════════════════════════════════════════════════════════════╗
║                        🔄 Complete Workflow                     ║
╠═════════════════════════════════════════════════════════════════╣
║                                                               ║
║  Step 1: Cache Miss Handling                                   ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         ║
║  │ 💿 Disk     │───►│ 💾 CPU Cache│───►│ 🎯 GPU      │         ║
║  │ (On-demand  │     │ (FIFO       │    │ Memory      │         ║
║  │  loading)   │     │  Management)│    │ (Compute    │         ║
║  └─────────────┘    └─────────────┘    │ Execution)  │         ║
║                                        └─────────────┘         ║
║                                                               ║
║  Step 2: Cache Hit Handling                                    ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         ║
║  │ 💿 Disk     │    │ 💾 CPU Cache│───►│ 🎯 GPU      │         ║
║  │ (Skip       │     │ (Direct     │    │ Memory      │         ║
║  │  loading)   │     │  Access)    │    │ (Compute    │         ║
║  └─────────────┘    └─────────────┘    │ Execution)  │         ║
║                                        └─────────────┘         ║
║                                                               ║
║  Step 3: Memory Management                                      ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         ║
║  │ 💿 Disk     │    │ 💾 CPU Cache│    │ 🎯 GPU      │         ║
║  │ (Persistent │     │ (FIFO       │    │ Memory      │         ║
║  │  Storage)   │     │  Eviction)  │    │ (Swap       │         ║
║  └─────────────┘    └─────────────┘    │ Rotation)   │         ║
║                                        └─────────────┘         ║
╚═════════════════════════════════════════════════════════════════╝

Work Steps:
1. Disk Storage: Model weights stored by block on SSD/NVMe, one .safetensors file per block
2. Task Scheduling: When a block/phase is needed, priority task queue assigns disk worker threads
3. Async Loading: Multiple disk threads parallelly read weight files from disk to CPU memory buffer
4. Intelligent Caching: CPU memory buffer uses FIFO strategy for cache management with configurable size
5. Cache Hit: If weights are already in cache, directly transfer to GPU without disk reading
6. Prefetch Transfer: Weights in cache asynchronously transfer to GPU memory (using GPU load stream)
7. Compute Execution: Weights on GPU perform computation (using compute stream), while background continues prefetching next block/phase
8. Swap Rotation: After computation completion, rotate block/phase positions for continuous computation
9. Memory Management: When CPU cache is full, automatically evict earliest used weight blocks/phases
```

**Key Features:**
- **Lazy Loading**: Model weights loaded from disk on-demand, avoiding loading entire model at once
- **Intelligent Caching**: CPU memory buffer uses FIFO strategy with configurable size
- **Multi-threaded Prefetching**: Uses multiple disk worker threads for parallel loading
- **Asynchronous Transfer**: Uses CUDA streams to overlap computation and data transfer
- **Swap Rotation**: Achieves continuous computation through position rotation, avoiding repeated loading/unloading



## ⚙️ Configuration Parameters

### GPU-CPU Offloading Configuration

```python
config = {
    "cpu_offload": True,
    "offload_ratio": 1.0,           # Offload ratio (0.0-1.0)
    "offload_granularity": "block", # Offload granularity: "block" or "phase"
    "lazy_load": False,             # Disable lazy loading
}
```

### Disk-CPU-GPU Offloading Configuration

```python
config = {
    "cpu_offload": True,
    "lazy_load": True,              # Enable lazy loading
    "offload_ratio": 1.0,           # Offload ratio
    "offload_granularity": "phase", # Recommended to use phase granularity
    "num_disk_workers": 2,          # Number of disk worker threads
    "offload_to_disk": True,        # Enable disk offloading
    "offload_path": ".",            # Disk offload path
}
```

**Intelligent Cache Key Parameters:**
- `max_memory`: Controls CPU cache size, affects cache hit rate and memory usage
- `num_disk_workers`: Controls number of disk loading threads, affects prefetch speed
- `offload_granularity`: Controls cache granularity (block or phase), affects cache efficiency
  - `"block"`: Cache management in units of complete Transformer layers
  - `"phase"`: Cache management in units of individual computational components

Detailed configuration files can be referenced at [config](https://github.com/ModelTC/lightx2v/tree/main/configs/offload)

## 🎯 Usage Recommendations

```
╔═════════════════════════════════════════════════════════════════╗
║                        📋 Configuration Guide                   ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  🔄 GPU-CPU Block/Phase Offloading:                            ║
║        Suitable for insufficient GPU VRAM (RTX 3090/4090 24G)  ║
║        but adequate system memory (>64/128G)                   ║
║  💾 Disk-CPU-GPU Block/Phase Offloading:                       ║
║        Suitable for insufficient GPU VRAM (RTX 3060/4090 8G)   ║
║        and system memory (16/32G)                              ║
║  🚫 No Offload: Suitable for high-end hardware configurations, ║
║        pursuing optimal performance                             ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝
```

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **Disk I/O Bottleneck**
   ```
   Solution: Use NVMe SSD, increase num_disk_workers
   ```

2. **Memory Buffer Overflow**
   ```
   Solution: Increase max_memory or decrease num_disk_workers
   ```

3. **Loading Timeout**
   ```
   Solution: Check disk performance, optimize file system
   ```

**Note**: This offloading mechanism is specifically designed for Lightx2v, fully utilizing modern hardware's asynchronous computing capabilities, significantly reducing the hardware threshold for large model inference.
