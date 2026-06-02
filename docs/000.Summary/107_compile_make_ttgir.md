# 07 - `make_ttgir` 阶段分析

> 分析 TTGIR 阶段：TTIR → TTGIR 转换 + GPU 专属优化

---

## 一、整体流程

```
输入:  tt dialect ModuleOp
    │
    ├─→ 1. TTIR → TTGIR 转换
    │   convert_to_ttgpuir → tt → ttg (引入 GPU layout)
    │
    ├─→ 2. 通用 TTGIR 优化 (两后端共享)
    │   coalesce → f32_dot_tc → remove_layout_conversions
    │   → optimize_thread_locality → accelerate_matmul
    │   → optimize_dot_operands → prefetch → reduce_data_duplication
    │   → fuse_nested_loops → combine_tensor_select_and_if
    │   → reorder_instructions → ...
    │
    ├─→ 3. 架构专属优化
    │   ├─ NVIDIA SM80/90:  hopper_warpspec → assign_latencies
    │   │                   → schedule_loops → pipeline
    │   ├─ NVIDIA SM100+:   promote_lhs_to_tmem → warp_specialize
    │   │                   → schedule_loops → pipeline → hoist_tmem_alloc
    │   └─ AMD:             schedule_loops_amd → pipeline_amd
    │                       → block_pingpong → convert_to_buffer_ops
    │
    └─→ 4. 收尾优化
        fence_insertion → lower_mma → sccp → cse → canonicalizer
    │
输出: ttg dialect ModuleOp (含完整 GPU layout 信息)
```

---

## 二、涉及的 Python 代码

### 2.1 NVIDIA `make_ttgir()`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/nvidia/compiler.py` | 246-318 | `CUDABackend.make_ttgir()` — NVIDIA TTGIR 管道（约 70 行） |

### 2.2 AMD `make_ttgir()`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/amd/compiler.py` | 205-266 | `HIPBackend.make_ttgir()` — AMD TTGIR 管道（约 60 行） |

---

## 三、涉及的 C++ 代码

### 3.1 转换 Pass

| Python 调用 | C++ 工厂 | 实现文件 |
|-------------|----------|---------|
| `passes.ttir.add_convert_to_ttgpuir(pm, ...)` | `createConvertTritonToTritonGPU` | `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` |

### 3.2 通用 TTGIR Pass (两后端共享)

| Python 调用 | C++ 工厂 | 实现文件 |
|-------------|----------|---------|
| `passes.ttgpuir.add_coalesce` | `createTritonGPUCoalesce` | `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` |
| `passes.ttgpuir.add_f32_dot_tc` | `createTritonGPUF32DotTC` | `lib/Dialect/TritonGPU/Transforms/F32DotTC.cpp` |
| `passes.ttgpuir.add_remove_layout_conversions` | `createTritonGPURemoveLayoutConversions` | `lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp` |
| `passes.ttgpuir.add_optimize_thread_locality` | `createTritonGPUOptimizeThreadLocality` | `lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp` |
| `passes.ttgpuir.add_accelerate_matmul` | `createTritonGPUAccelerateMatmul` | `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` |
| `passes.ttgpuir.add_optimize_dot_operands` | `createTritonGPUOptimizeDotOperands` | `lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp` |
| `passes.ttgpuir.add_prefetch` | `createTritonGPUPrefetch` | `lib/Dialect/TritonGPU/Transforms/Prefetch.cpp` |
| `passes.ttgpuir.add_reduce_data_duplication` | `createTritonGPUReduceDataDuplication` | `lib/Dialect/TritonGPU/Transforms/ReduceDataDuplication.cpp` |
| `passes.ttgpuir.add_fuse_nested_loops` | `createTritonGPUFuseNestedLoops` | `lib/Dialect/TritonGPU/Transforms/FuseNestedLoops.cpp` |
| `passes.ttgpuir.add_combine_tensor_select_and_if` | `createTritonGPUCombineTensorSelectAndIf` | `lib/Dialect/TritonGPU/Transforms/CombineTensorSelectAndIf.cpp` |
| `passes.ttgpuir.add_reorder_instructions` | `createTritonGPUReorderInstructions` | `lib/Dialect/TritonGPU/Transforms/ReorderInstructions.cpp` |
| `passes.ttgpuir.add_coalesce_async_copy` | `createTritonGPUCoalesceAsyncCopy` | `lib/Dialect/TritonGPU/Transforms/CoalesceAsyncCopy.cpp` |
| `passes.ttgpuir.add_pipeline` | `createTritonGPUPipeline` | `lib/Dialect/TritonGPU/Transforms/Pipeline.cpp` |
| `passes.ttgpuir.add_schedule_loops` | `createTritonGPUScheduleLoops` | `lib/Dialect/TritonGPU/Transforms/ScheduleLoops.cpp` |
| `passes.ttgpuir.add_assign_latencies` | `createTritonGPUAssignLatencies` | `lib/Dialect/TritonGPU/Transforms/AssignLatencies.cpp` |

### 3.3 NVIDIA 专属 Pass

| Python 调用 | C++ 工厂 | 实现文件 |
|-------------|----------|---------|
| `nvidia.passes.ttnvgpuir.add_plan_cta` | `createTritonNvidiaGPUPlanCTAPass` | `lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp` |
| `nvidia.passes.ttnvgpuir.add_fence_insertion` | `createTritonGPUFenceInsertion` | `lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp` |
| `nvidia.passes.ttnvgpuir.add_lower_mma` | `createTritonNvidiaGPUMMALoweringPass` | `lib/Dialect/TritonNvidiaGPU/Transforms/MMALowering.cpp` |
| `nvidia.passes.ttnvgpuir.add_tma_lowering` | `createTritonNvidiaGPUTMALoweringPass` | `lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp` |
| `nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem` | `createTritonNvidiaGPUPromoteLHSToTMemPass` | `lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp` |
| `nvidia.passes.hopper.add_hopper_warpspec` | `createNVGPUWarpSpecialization` | `third_party/nvidia/hopper/lib/` |

### 3.4 AMD 专属 Pass

| Python 调用 | 实现文件 |
|-------------|---------|
| `amd.passes.ttgpuir.add_accelerate_matmul` | `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateMatmul.cpp` |
| `amd.passes.ttgpuir.add_pipeline` | `third_party/amd/lib/TritonAMDGPUTransforms/Pipeline.cpp` |
| `amd.passes.ttgpuir.add_schedule_loops` | `third_party/amd/lib/TritonAMDGPUTransforms/ScheduleLoops.cpp` |
| `amd.passes.ttgpuir.add_block_pingpong` | `third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp` |
| `amd.passes.ttgpuir.add_convert_to_buffer_ops` | `third_party/amd/lib/TritonAMDGPUTransforms/ConvertToBufferOps.cpp` |

---

## 四、涉及的 MLIR 代码

| 目录 | 内容 |
|------|------|
| `include/triton/Dialect/TritonGPU/Transforms/Passes.td` | TTGIR pass 声明 |
| `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td` | GPU layout 编码定义 |
| `include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td` | NVIDIA pass 声明 |
| `third_party/amd/include/TritonAMDGPUTransforms/Passes.td` | AMD pass 声明 |

---

## 五、关键转换说明

### `convert_to_ttgpuir` — 核心转换

```
输入:  tt dialect (tt.load, tt.dot, tt.addptr, ...)
      + arith, scf, math
输出: ttg dialect (ttg.load → 含 layout 编码的张量)
关键: 为每个张量类型附加 GPU 数据排布编码:
      - BlockedEncodingAttr (标准分块排布)
      - DotOperandEncodingAttr (dot 操作数排布)
      - MmaEncodingTrait (Tensor Core 排布)
    参数: numWarps, threadsPerWarp, numCTAs
```

---

## 六、进入条件、参数和输出

### 进入条件
`compile()` 中 `stages["ttgir"]` stage 被执行。

### 参数 (NVIDIA)

| 参数 | 类型 | 说明 |
|------|------|------|
| `mod` | `ir.module` | TTIR 优化后的 `mlir::ModuleOp` |
| `metadata` | dict | 编译元数据 |
| `opt` | CUDAOptions | 编译选项（含 num_warps, num_stages, num_ctas） |
| `capability` | int | GPU 计算能力 |

### 输出

| 输出 | 说明 |
|------|------|
| `mod` | TTGIR 优化后的 `mlir::ModuleOp`（ttg dialect） |
| `metadata["tensordesc_meta"]` | Tensor descriptor 元数据 |