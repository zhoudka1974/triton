# 11 - `kernel.run` 阶段分析

> 分析 `CompiledKernel` 的加载和执行过程

---

## 一、整体流程

```
Compile() 返回 CompiledKernel
    │
    ├─→ JITFunction.run() 获取 kernel
    │   ├─→ kernel = kernel_cache[key]
    │   └─→ (首次) kernel = _do_compile() → compile() → CompiledKernel
    │
    ├─→ 预热模式 (warmup=True): 编译后直接返回，不执行
    │
    ├─→ 执行模式 (warmup=False):
    │   ├─→ grid = grid(bound_args)   ← 计算 grid 维度
    │   ├─→ kernel.result()           ← 异步编译等待完成
    │   ├─→ kernel._init_handles()    ← 惰性初始化 CUDA 句柄
    │   │   ├─→ 加载 cubin 到 CUDA module
    │   │   ├─→ 获取 kernel function 句柄
    │   │   └─→ 检查 shared memory 是否超出限制
    │   │
    │   └─→ kernel.run(grid_0, grid_1, grid_2, stream,
    │                  function, packed_metadata, ...)
    │       └─→ CUDA: cuLaunchKernel
    │
输出: GPU 内核执行
```

---

## 二、涉及的 Python 代码

### 2.1 `CompiledKernel` 类

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/compiler/compiler.py` | 404-.. | `CompiledKernel` — 编译结果封装 |
| `python/triton/compiler/compiler.py` | 406-434 | `__init__()` — 从 metadata 和 asm 文件恢复编译结果 |
| `python/triton/compiler/compiler.py` | 436-463 | `_init_handles()` — 惰性初始化 CUDA/HIP module 和 function 句柄 |
| `python/triton/compiler/compiler.py` | 452 | `driver.active.launcher_cls(...)` — 创建启动器对象 |

### 2.2 `CompiledKernel` 的启动接口

`CompiledKernel` 关键属性：

```python
class CompiledKernel:
    asm = {
        "ttir": "...",      # TTIR 文本
        "ttgir": "...",     # TTGIR 文本
        "llir": "...",      # LLVM IR 文本
        "ptx": "...",       # PTX 汇编 (NVIDIA)
        "cubin": b"..."     # CUDA 二进制 (NVIDIA)
        "amdgcn": "...",    # AMD GCN 汇编
        "hsaco": b"..."     # AMD 二进制
    }
    metadata.target         # GPUTarget
    metadata.num_warps      # warp 数
    metadata.shared         # 共享内存大小
    metadata.name           # 内核名
    packed_metadata         # 启动参数元组
    function                # CUDA/HIP 函数句柄
    kernel                  # 二进制数据 (cubin/hsaco)
```

### 2.3 执行路径 (JITFunction.run)

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/runtime/jit.py` | 731-745 | 执行逻辑：grid 规范化 → kernel 启动 |
| `python/triton/runtime/jit.py` | 740-741 | `kernel.result()` — 等待异步编译完成 |
| `python/triton/runtime/jit.py` | 744-745 | `kernel.run(grid_0, grid_1, grid_2, stream, ...)` — 启动 GPU 内核 |

### 2.4 Launcher

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/nvidia/driver.py` | - | `CudaLauncher` — NVIDIA CUDA 内核启动器 |

---

## 三、涉及的 C++ 代码

| Python 调用 | 功能 |
|-------------|------|
| `driver.active.launcher_cls(...)` | 创建 CUDA/HIP 启动器，内部调用 `cuModuleLoadData` / `cuModuleGetFunction` |
| `kernel.run(...)` | 最终调用 `cuLaunchKernel` / `hipLaunchKernel` 执行 GPU 内核 |

---

## 四、涉及的 MLIR 代码

本阶段不涉及 MLIR。

---

## 五、进入条件、参数和输出

### 进入条件
- `JITFunction.run()` 中缓存命中 或 编译完成
- `warmup=False`（预热模式仅编译不执行）

### 参数

| 参数 | 来源 | 说明 |
|------|------|------|
| `grid_0, grid_1, grid_2` | `grid(bound_args)` | grid 各维度大小 |
| `stream` | `driver.active.get_current_stream(device)` | CUDA stream |
| `kernel.function` | `_init_handles()` | CUDA/HIP 函数句柄 |
| `kernel.packed_metadata` | `backend.pack_metadata()` | 启动参数（num_warps, shared 等） |

### 输出
GPU 内核在 GPU 上执行。Python 侧无返回值。