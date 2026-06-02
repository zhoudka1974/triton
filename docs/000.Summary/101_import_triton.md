# 01 - import triton 阶段分析

> 分析 `import triton`, `import triton.language as tl` 时触发的所有初始化过程

---

## 一、整体流程

```
用户执行: import triton
    │
    ├─→ triton/__init__.py 导入
    │   ├─→ from .runtime import ...      → runtime/__init__.py
    │   │   ├─→ from .driver import driver → 创建 DriverConfig 对象
    │   │   │   └─→ 首次访问 driver.active 触发:
    │   │   │       _create_driver()
    │   │   │       → 遍历 backends, 调用 is_active()
    │   │   │       → 激活当前 GPU 对应的 driver (CudaDriver 或 HIPDriver)
    │   │   ├─→ from .jit import ...      → 加载 JITFunction, KernelInterface
    │   │   └─→ from .cache import ...     → 缓存后端
    │   ├─→ from .compiler import compile  → compiler/compiler.py
    │   ├─→ from . import language         → triton.language (TL DSL)
    │   │   └─→ import triton.language as tl
    │   └─→ from ._C.libtriton import ...  → 加载 C++ 扩展 libtriton.so
    │       ├─→ ir 子模块:  context, pass_manager, builder, load_dialects
    │       ├─→ passes 子模块: common, ttir, ttgpuir, convert, llvmir, gluon
    │       ├─→ llvm 子模块:  LLVM IR 转换工具
    │       └─→ nvidia/amd 子模块:  后端 dialect 注册
    │
    └─→ triton/backends/__init__.py
        └─→ _discover_backends()
            ├─→ 扫描 triton/backends/ 子目录
            ├─→ nvidia/: CUDABackend + CudaDriver
            └─→ amd/:    HIPBackend + HIPDriver
```

---

## 二、涉及的 Python 代码

### 2.1 包入口

| 文件 | 功能 |
|------|------|
| `python/triton/__init__.py` | 包入口，导入所有核心子模块，导出 `jit`, `compile`, `language`, `testing` 等 |
| `python/triton/runtime/__init__.py` | Runtime 子模块入口，导入 driver, jit, cache, autotuner 等 |

### 2.2 Backend 发现与 Driver 检测

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/__init__.py` | 38-66 | `_discover_backends()` 扫描子目录注册后端 |
| `python/triton/backends/__init__.py` | 19-29 | `_find_concrete_subclasses()` 查找导出类 |
| `python/triton/backends/compiler.py` | 8-14 | `GPUTarget` 数据类定义 |
| `python/triton/backends/compiler.py` | 23-28 | `BaseBackend` 基类 |
| `python/triton/runtime/driver.py` | 6-10 | `_create_driver()` 激活当前 GPU 的 driver |
| `python/triton/runtime/driver.py` | 13-38 | `DriverConfig` 单例管理 |

### 2.3 NVIDIA Backend

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/nvidia/compiler.py` | 150-155 | `CUDABackend` 类定义 |
| `python/triton/backends/nvidia/compiler.py` | 154-155 | `CUDABackend.supports_target()` — 判断 target.backend == 'cuda' |
| `python/triton/backends/nvidia/driver.py` | 717-729 | `CudaDriver` 类 |
| `python/triton/backends/nvidia/driver.py` | 724-729 | `get_current_target()` — 获取 GPU target |
| `python/triton/backends/nvidia/driver.py` | 739-744 | `is_active()` — 判断当前环境是否有 NVIDIA GPU |

### 2.4 AMD Backend

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/amd/compiler.py` | 102-108 | `HIPBackend` 类定义 |
| `python/triton/backends/amd/compiler.py` | 106-108 | `HIPBackend.supports_target()` — 判断 target.backend == 'hip' |
| `python/triton/backends/amd/driver.py` | 842-848 | `is_active()` — 判断当前环境是否有 AMD GPU |
| `python/triton/backends/amd/driver.py` | 853-858 | `get_current_target()` — 获取 GPU target |

### 2.5 `import triton.language as tl`

| 文件 | 功能 |
|------|------|
| `python/triton/language/__init__.py` | TL DSL 入口，导出所有语言原语 |

---

## 三、涉及的 C++ 代码

### 3.1 pybind11 绑定入口

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/src/main.cc` | 50-62 | `PYBIND11_MODULE(libtriton)` — 初始化所有 C++ 子模块 |

### 3.2 核心绑定模块

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/src/ir.cc` | 1-.. | `ir` 子模块绑定：`context`, `pass_manager`, `builder`, `load_dialects`, `module` |
| `python/src/ir.cc` | 363-378 | `load_dialects()` — 注册 Triton + MLIR dialect 到 MLIRContext |
| `python/src/passes.cc` | 28-37 | `init_triton_passes_common()` — 绑定通用 MLIR pass |
| `python/src/passes.cc` | 39-53 | `init_triton_passes_ttir()` — 绑定 TTIR pass |
| `python/src/passes.cc` | 55-97 | `init_triton_passes_ttgpuir()` — 绑定 TTGIR pass |
| `python/src/passes.cc` | 99-106 | `init_triton_passes_convert()` — 绑定转换 pass |
| `python/src/passes.cc` | 108-112 | `init_triton_passes_llvmir()` — 绑定 LLVM debug pass |
| `python/src/passes.cc` | 114-123 | `init_gluon_passes()` — 绑定 Gluon pass |
| `python/src/passes.cc` | 125-133 | `init_triton_passes()` — 组装所有 pass 子模块 |
| `python/src/passes.h` | 1-44 | `ADD_PASS_WRAPPER_N` 宏定义，简化 pass 绑定 |

### 3.3 NVIDIA C++ 扩展

| 文件 | 行号 | 功能 |
|------|------|------|
| `third_party/nvidia/triton_nvidia.cc` | 51-79 | 注册 NVIDIA 专属 pass 子模块 |

### 3.4 AMD C++ 扩展

| 文件 | 行号 | 功能 |
|------|------|------|
| `third_party/amd/python/triton_amd.cc` | 49-108 | 注册 AMD 专属 pass 子模块 |

---

## 四、涉及的 MLIR 代码

在本阶段（import），MLIR 代码**尚未执行**，只是通过 pybind11 绑定了 pass 工厂函数。实际注册到 `mlir::MLIRContext` 的 dialect 包括：

| Dialect | C++ 类 | 归属 |
|---------|--------|------|
| `triton` (tt) | `mlir::triton::TritonDialect` | Triton 核心 |
| `triton::gpu` (ttg) | `mlir::triton::gpu::TritonGPUDialect` | Triton 核心 |
| `triton::nvidia_gpu` (ttng) | `mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect` | Triton 核心 |
| `triton::instrument` (tti) | `mlir::triton::instrument::TritonInstrumentDialect` | Triton 核心 |
| `gluon` | `mlir::triton::gluon::GluonDialect` | Triton 核心 (实验性) |
| `arith` | `mlir::arith::ArithDialect` | 上游 MLIR |
| `math` | `mlir::math::MathDialect` | 上游 MLIR |
| `scf` | `mlir::scf::SCFDialect` | 上游 MLIR |
| `cf` | `mlir::cf::ControlFlowDialect` | 上游 MLIR |
| `gpu` | `mlir::gpu::GPUDialect` | 上游 MLIR |
| `llvm` | `mlir::LLVM::LLVMDialect` | 上游 MLIR |
| `ub` | `mlir::ub::UBDialect` | 上游 MLIR |
| `nvgpu` | `mlir::triton::nvgpu::NVGPUDialect` | NVIDIA 扩展 |
| `nvws` | `mlir::triton::nvws::NVWSDialect` | NVIDIA 扩展 |
| `ttamd` | `mlir::triton::amd::TritonAMDGPUDialect` | AMD 扩展 |

这些 dialect 的 C++ 实现位于：

| 目录 | 内容 |
|------|------|
| `lib/Dialect/Triton/IR/` | Triton dialect 实现 |
| `lib/Dialect/TritonGPU/IR/` | TritonGPU dialect 实现 |
| `lib/Dialect/TritonNvidiaGPU/IR/` | TritonNvidiaGPU dialect 实现 |
| `third_party/nvidia/include/Dialect/NVGPU/IR/` | NVGPU dialect |
| `third_party/amd/include/Dialect/TritonAMDGPU/IR/` | TritonAMDGPU dialect |

---

## 五、关键数据流

```
import triton 完成时:
    backends["nvidia"] = Backend(CUDABackend, CudaDriver)
    backends["amd"]    = Backend(HIPBackend,  HIPDriver)
    driver.active      = CudaDriver 或 HIPDriver (取决于当前 GPU)
    libtriton.so 已加载:
        ir.pass_manager, ir.context, ir.builder, ir.load_dialects
        passes.common, passes.ttir, passes.ttgpuir, passes.convert, ...
        llvm.to_module, llvm.translate_to_asm, llvm.optimize_module
```