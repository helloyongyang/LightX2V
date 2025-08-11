# LightX2V Server

## Overview

The LightX2V server is a distributed video generation service built with FastAPI that processes image-to-video tasks using a multi-process architecture with GPU support. It implements a sophisticated task queue system with distributed inference capabilities for high-throughput video generation workloads.

## Architecture

### System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Client[HTTP Client]
    end
    
    subgraph "API Layer"
        FastAPI[FastAPI Application]
        ApiServer[ApiServer]
        Router1[Tasks Router<br/>/v1/tasks]
        Router2[Files Router<br/>/v1/files]
        Router3[Service Router<br/>/v1/service]
    end
    
    subgraph "Service Layer"
        TaskManager[TaskManager<br/>Thread-safe Task Queue]
        FileService[FileService<br/>File I/O & Downloads]
        VideoService[VideoGenerationService]
    end
    
    subgraph "Processing Layer"
        Thread[Processing Thread<br/>Sequential Task Loop]
    end
    
    subgraph "Distributed Inference Layer"
        DistService[DistributedInferenceService]
        SharedData[(Shared Data<br/>mp.Manager.dict)]
        TaskEvent[Task Event<br/>mp.Manager.Event]
        ResultEvent[Result Event<br/>mp.Manager.Event]
        
        subgraph "Worker Processes"
            W0[Worker 0<br/>Master/Rank 0]
            W1[Worker 1<br/>Rank 1]
            WN[Worker N<br/>Rank N]
        end
    end
    
    subgraph "Resource Management"
        GPUManager[GPUManager<br/>GPU Detection & Allocation]
        DistManager[DistributedManager<br/>PyTorch Distributed]
        Config[ServerConfig<br/>Configuration]
    end
    
    Client -->|HTTP Request| FastAPI
    FastAPI --> ApiServer
    ApiServer --> Router1
    ApiServer --> Router2
    ApiServer --> Router3
    
    Router1 -->|Create/Manage Tasks| TaskManager
    Router1 -->|Process Tasks| Thread
    Router2 -->|File Operations| FileService
    Router3 -->|Service Status| TaskManager
    
    Thread -->|Get Pending Tasks| TaskManager
    Thread -->|Generate Video| VideoService
    
    VideoService -->|Download Images| FileService
    VideoService -->|Submit Task| DistService
    
    DistService -->|Update| SharedData
    DistService -->|Signal| TaskEvent
    TaskEvent -->|Notify| W0
    W0 -->|Broadcast| W1
    W0 -->|Broadcast| WN
    
    W0 -->|Update Result| SharedData
    W0 -->|Signal| ResultEvent
    ResultEvent -->|Notify| DistService
    
    W0 -.->|Uses| GPUManager
    W1 -.->|Uses| GPUManager
    WN -.->|Uses| GPUManager
    
    W0 -.->|Setup| DistManager
    W1 -.->|Setup| DistManager
    WN -.->|Setup| DistManager
    
    DistService -.->|Reads| Config
    ApiServer -.->|Reads| Config
```

### Components

#### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **ServerManager** | `main.py` | Orchestrates server lifecycle, startup/shutdown sequences |
| **ApiServer** | `api.py` | FastAPI application manager with route registration |
| **TaskManager** | `task_manager.py` | Thread-safe task queue and lifecycle management |
| **FileService** | `service.py` | File I/O, HTTP downloads with retry logic |
| **VideoGenerationService** | `service.py` | Video generation workflow orchestration |
| **DistributedInferenceService** | `service.py` | Multi-process inference management |
| **GPUManager** | `gpu_manager.py` | GPU detection, allocation, and memory management |
| **DistributedManager** | `distributed_utils.py` | PyTorch distributed communication setup |
| **ServerConfig** | `config.py` | Centralized configuration with environment variable support |

## Task Processing Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as API Server
    participant TM as TaskManager
    participant PT as Processing Thread
    participant VS as VideoService
    participant FS as FileService
    participant DIS as Distributed<br/>Inference Service
    participant W0 as Worker 0<br/>(Master)
    participant W1 as Worker 1..N
    
    C->>API: POST /v1/tasks<br/>(Create Task)
    API->>TM: create_task()
    TM->>TM: Generate task_id
    TM->>TM: Add to queue<br/>(status: PENDING)
    API->>PT: ensure_processing_thread()
    API-->>C: TaskResponse<br/>(task_id, status: pending)
    
    Note over PT: Processing Loop
    PT->>TM: get_next_pending_task()
    TM-->>PT: task_id
    
    PT->>TM: acquire_processing_lock()
    PT->>TM: start_task()<br/>(status: PROCESSING)
    
    PT->>VS: generate_video_with_stop_event()
    
    alt Image is URL
        VS->>FS: download_image()
        FS->>FS: HTTP download<br/>with retry
        FS-->>VS: image_path
    else Image is Base64
        VS->>FS: save_base64_image()
        FS-->>VS: image_path
    else Image is Upload
        VS->>FS: validate_file()
        FS-->>VS: image_path
    end
    
    VS->>DIS: submit_task(task_data)
    DIS->>DIS: shared_data["current_task"] = task_data
    DIS->>DIS: task_event.set()
    
    Note over W0,W1: Distributed Processing
    W0->>W0: task_event.wait()
    W0->>W0: Get task from shared_data
    W0->>W1: broadcast_task_data()
    
    par Parallel Inference
        W0->>W0: run_pipeline()
    and
        W1->>W1: run_pipeline()
    end
    
    W0->>W0: barrier() for sync
    W0->>W0: shared_data["result"] = result
    W0->>DIS: result_event.set()
    
    DIS->>DIS: result_event.wait()
    DIS->>VS: return result
    VS-->>PT: TaskResponse
    
    PT->>TM: complete_task()<br/>(status: COMPLETED)
    PT->>TM: release_processing_lock()
    
    Note over C: Client Polling
    C->>API: GET /v1/tasks/{task_id}/status
    API->>TM: get_task_status()
    TM-->>API: status info
    API-->>C: Task Status
    
    C->>API: GET /v1/tasks/{task_id}/result
    API->>TM: get_task_status()
    API->>FS: stream_file_response()
    FS-->>API: Video Stream
    API-->>C: Video File
```

## Task States

```mermaid
stateDiagram-v2
    [*] --> PENDING: create_task()
    PENDING --> PROCESSING: start_task()
    PROCESSING --> COMPLETED: complete_task()
    PROCESSING --> FAILED: fail_task()
    PENDING --> CANCELLED: cancel_task()
    PROCESSING --> CANCELLED: cancel_task()
    COMPLETED --> [*]
    FAILED --> [*]
    CANCELLED --> [*]
```

## API Endpoints

see `{base_url}/docs`

### Task Management

- POST `/v1/tasks/`
- POST `/v1/tasks/form`
- GET `/v1/tasks/{task_id}/status`
- GET `/v1/tasks/{task_id}/result`
- DELETE `/v1/tasks/{task_id}`
- DELETE `/v1/tasks/all/running`
- GET `/v1/tasks/`
- GET `/v1/tasks/queue/status`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LIGHTX2V_HOST` | Server host address | `0.0.0.0` |
| `LIGHTX2V_PORT` | Server port | `8000` |
| `LIGHTX2V_MAX_QUEUE_SIZE` | Maximum task queue size | `100` |
| `LIGHTX2V_CACHE_DIR` | File cache directory | `/tmp/lightx2v_cache` |
| `LIGHTX2V_TASK_TIMEOUT` | Task processing timeout (seconds) | `600` |
| `LIGHTX2V_HTTP_TIMEOUT` | HTTP download timeout (seconds) | `30` |
| `LIGHTX2V_HTTP_MAX_RETRIES` | HTTP download max retries | `3` |
| `LIGHTX2V_MAX_UPLOAD_SIZE` | Maximum upload file size (bytes) | `100MB` |

### Command Line Arguments

```bash
python -m lightx2v.server.main \
    --model_path /path/to/model \
    --model_cls wan2.1_distill \
    --task i2v \
    --host 0.0.0.0 \
    --port 8000 \
    --config_json /path/to/xxx_config.json
```

```bash
python -m lightx2v.server.main \
    --model_path /path/to/model \
    --model_cls wan2.1_distill \
    --task i2v \
    --host 0.0.0.0 \
    --port 8000 \
    --config_json /path/to/xxx_dist_config.json \
    --nproc_per_node 2
```


## Key Features

### 1. Distributed Processing

- **Multi-process architecture** for GPU parallelization
- **Master-worker pattern** with rank 0 as coordinator
- **PyTorch distributed** backend (NCCL for GPU, Gloo for CPU)
- **Automatic GPU allocation** across processes
- **Task broadcasting** with chunked pickle serialization

### 2. Task Queue Management

- **Thread-safe** task queue with locks
- **Sequential processing** with single processing thread
- **Configurable queue limits** with overflow protection
- **Task prioritization** (FIFO)
- **Automatic cleanup** of old completed tasks
- **Cancellation support** for pending and running tasks

### 3. File Management

- **Multiple input formats**: URL, base64, file upload
- **HTTP downloads** with exponential backoff retry
- **Streaming responses** for large video files
- **Cache management** with automatic cleanup
- **File validation** and format detection

### 4. Resilient Architecture

- **Graceful shutdown** with signal handling
- **Process failure recovery** mechanisms
- **Connection pooling** for HTTP clients
- **Timeout protection** at multiple levels
- **Comprehensive error handling** throughout

### 5. Resource Management

- **GPU memory management** with cache clearing
- **Process lifecycle management**
- **Connection pooling** for efficiency
- **Memory-efficient** streaming for large files
- **Automatic resource cleanup** on shutdown

## Performance Considerations

1. **Single Task Processing**: Tasks are processed sequentially to manage GPU memory effectively
2. **Multi-GPU Support**: Distributes inference across available GPUs for parallelization
3. **Connection Pooling**: Reuses HTTP connections to reduce overhead
4. **Streaming Responses**: Large files are streamed to avoid memory issues
5. **Queue Management**: Automatic task cleanup prevents memory leaks
6. **Process Isolation**: Distributed workers run in separate processes for stability

## Usage Examples

### Client Usage

```python
import httpx
import base64

# Create a task with URL image
response = httpx.post(
    "http://localhost:8000/v1/tasks/",
    json={
        "prompt": "A cat playing piano",
        "image_path": "https://example.com/image.jpg",
        "use_prompt_enhancer": True,
        "seed": 42
    }
)
task_id = response.json()["task_id"]

# Create a task with base64 image
with open("image.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()
response = httpx.post(
    "http://localhost:8000/v1/tasks/",
    json={
        "prompt": "A dog dancing",
        "image_path": f"data:image/png;base64,{image_base64}"
    }
)

# Check task status
status = httpx.get(f"http://localhost:8000/v1/tasks/{task_id}/status")
print(status.json())

# Download result when completed
if status.json()["status"] == "completed":
    video = httpx.get(f"http://localhost:8000/v1/tasks/{task_id}/result")
    with open("output.mp4", "wb") as f:
        f.write(video.content)
```

## Monitoring and Debugging

### Logging

The server uses `loguru` for structured logging. Logs include:

- Request/response details
- Task lifecycle events
- Worker process status
- Error traces with context

### Health Checks

- `/v1/service/status` - Overall service health
- `/v1/tasks/queue/status` - Queue status and processing state
- Process monitoring via system tools (htop, nvidia-smi)

### Common Issues

1. **GPU Out of Memory**: Reduce `nproc_per_node` or adjust model batch size
2. **Task Timeout**: Increase `LIGHTX2V_TASK_TIMEOUT` for longer videos
3. **Queue Full**: Increase `LIGHTX2V_MAX_QUEUE_SIZE` or add rate limiting
4. **Port Conflicts**: Change `LIGHTX2V_PORT` or `MASTER_PORT` range

## Security Considerations

1. **Input Validation**: All inputs validated with Pydantic schemas
2. **File Access**: Restricted to cache directory
3. **Resource Limits**: Configurable queue and file size limits
4. **Process Isolation**: Worker processes run with limited permissions
5. **HTTP Security**: Support for proxy and authentication headers

## License

See the main project LICENSE file for licensing information.
