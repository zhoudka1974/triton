# 08 - `make_llir` 阶段分析

> 分析 LLVM IR 生成阶段：TritonGPU → LLVM IR 的 lowering

---

## 一、整体流程

```
输入:  ttg dialect ModuleOp
    │
    ├─→ allocate_warp_groups       ← 分配 warp group
    ├─→ scf_to_cf                  ← 控制流降级
    ├─→ allocate_shared_memory      ← 共享内存分配 (NVIDIA 或 AMD 版)
    ├─→ to_llvmir                  ← 主 lowering: TritonGPU → LLVM
    ├─→ [nvgpu_to_llvm]            ← NVIDIA: NVGPU → LLVM
    ├─→ [warp_specialize_to_llvm]  ← NVIDIA: warp 特化 → LLVM
    ├─→ canonicalizer / cse / symbol_dce
    ├─→ nvvm_to_llvm / cf_to_llvmir / arith_to_llvmir
    └─→ LLVM optimize_module(O3)   ← LLVM 优化器
    │
输出: LLVM IR 文本字符串
```

---

## 二、涉及的 Python 代码

### 2.1 NVIDIA `make_llir()`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/nvidia/compiler.py` | 339-433 | `CUDABackend.make_llir()` — NVIDIA LLIR 管道（约 95 行） |

### 2.2 AMD `make_llir()`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/amd/compiler.py` | 285-440 | `HIPBackend.make_llir()` — AMD LLIR 管道（约 155 行） |

---

## 三、涉及的 C++ 代码

### 3.1 NVIDIA Pipeline 中的 Pass

| Python 调用 | C++ 工厂 | 实现文件 |
|-------------|----------|---------|
| `nvidia.passes.ttgpuir.add_to_llvmir(pm, cap, ptx)` | `createConvertTritonGPUToLLVMPass` | `third_party/nvidia/lib/Conversion/TritonGPUToLLVM/` |
| `nvidia.passes.ttgpuir.add_allocate_shared_memory_nv` | `createAllocateSharedMemoryNvPass` | `third_party/nvidia/lib/Dialect/NVGPU/` |
| `nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm` | `createConvertNVGPUToLLVM` | `third_party/nvidia/lib/Conversion/NVGPUToLLVM/` |
| `nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm` | `createConvertWarpSpecializeToLLVM` | `third_party/nvidia/lib/Conversion/NVWS/` |
| `nvidia.passes.ttnvgpuir.add_proxy_fence_insertion` | `createTritonGPUProxyFenceInsertion` | `lib/Dialect/TritonNvidiaGPU/Transforms/ProxFenceInsertion.cpp` |

### 3.2 AMD Pipeline 中的 Pass

| Python 调用 | C++ 工厂 | 实现文件 |
|-------------|----------|---------|
| `amd.passes.ttgpuir.add_to_llvmir(pm, arch, ftz)` | `createConvertTritonAMDGPUToLLVMPass` | `third_party/amd/lib/TritonAMDGPUToLLVM/` |
| `amd.passes.ttgpuir.add_allocate_shared_memory` | `createAllocateAMDGPUSharedMemory` | `third_party/amd/lib/` |
| `amd.passes.ttgpuir.add_optimize_lds_usage` | `createOptimizeLDSUsagePass` | `third_party/amd/lib/` |
| `amd.passes.ttgpuir.add_builtin_func_to_llvmir` | `createConvertBuiltinFuncToLLVMPass` | `third_party/amd/lib/` |

### 3.3 共享 Pass

| Python 调用 | C++ 工厂 | 来源 |
|-------------|----------|------|
| `passes.ttgpuir.add_allocate_warp_groups` | `createTritonGPUAllocateWarpGroups` | Triton 核心 |
| `passes.convert.add_scf_to_cf` | `createSCFToControlFlowPass` | 上游 MLIR |
| `passes.convert.add_cf_to_llvmir` | `createConvertControlFlowToLLVMPass` | 上游 MLIR |
| `passes.convert.add_arith_to_llvmir` | `createArithToLLVMConversionPass` | 上游 MLIR |
| `passes.convert.add_nvvm_to_llvm` | `createConvertNVVMToLLVMPass` | 上游 MLIR |
| `passes.common.add_canonicalizer` | `createCanonicalizerPass` | 上游 MLIR |
| `passes.common.add_cse` | `createCSEPass` | 上游 MLIR |
| `passes.common.add_symbol_dce` | `createSymbolDCEPass` | 上游 MLIR |

### 3.4 LLVM 工具调用

| Python 调用 | 功能 |
|-------------|------|
| `llvm.init_targets()` | 初始化 LLVM 目标 |
| `llvm.to_module(mod, context)` | MLIR LLVM dialect → LLVM IR Module |
| `llvm.attach_datalayout(...)` | 设置数据布局 |
| `llvm.optimize_module(llvm_mod, OPTIMIZE_O3)` | LLVM O3 优化 |

---

## 四、涉及的 MLIR 代码

| 目录 | 内容 |
|------|------|
| `lib/Conversion/TritonGPUToLLVM/*.cpp` | TritonGPU → LLVM 的各种 op lowering（约 20+ 文件） |
| `third_party/nvidia/lib/Conversion/TritonGPUToLLVM/` | NVIDIA 版 lowering |
| `third_party/nvidia/lib/Conversion/NVGPUToLLVM/` | NVGPU → LLVM |
| `third_party/nvidia/lib/Conversion/NVWS/` | Warp specialization → LLVM |
| `third_party/amd/lib/TritonAMDGPUToLLVM/` | AMD 版 lowering |

---

## 五、进入条件、参数和输出

### 进入条件
`compile()` 中 `stages["llir"]` stage 被执行。

### 参数 (NVIDIA)

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `ir.module` | TTGIR 优化后的 `mlir::ModuleOp` |
| `metadata` | dict | 编译元数据 |
| `options` | CUDAOptions | 编译选项 |
| `capability` | int | GPU 计算能力 |

### 输出

| 输出 | 说明 |
|------|------|
| `str` | LLVM IR 文本字符串 |
| `metadata["num_warps"]` | 实际使用的 warp 数 |
| `metadata["shared"]` | 共享内存大小 |
| `metadata["tmem_size"]` | Tensor memory 大小 (SM100+) |