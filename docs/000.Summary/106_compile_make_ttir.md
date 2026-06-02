# 06 - `make_ttir` 阶段分析

> 分析 TTIR 优化阶段：从 Python AST 转换后的 TTIR 到优化后的 TTIR

---

## 一、整体流程

```
输入:  tt dialect ModuleOp (含 arith, scf, math)
    │
    ├─→ inliner                  ← passes.common
    ├─→ rewrite_tensor_pointer   ← passes.ttir
    ├─→ [rewrite_tensor_descriptor_to_pointer]  ← SM<90
    ├─→ canonicalizer            ← passes.common
    ├─→ combine                  ← passes.ttir
    ├─→ reorder_broadcast        ← passes.ttir
    ├─→ cse                      ← passes.common
    ├─→ symbol_dce               ← passes.common
    └─→ loop_unroll              ← passes.ttir
    │
输出: 优化后的 tt dialect ModuleOp (仍为 TTIR，未引入 GPU layout)
```

**关键：此阶段在与 GPU 架构无关的 TTIR 级别做优化。**

---

## 二、涉及的 Python 代码

### 2.1 NVIDIA `make_ttir()`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/nvidia/compiler.py` | 229-244 | `CUDABackend.make_ttir()` — NVIDIA TTIR 管道 |

### 2.2 AMD `make_ttir()`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/amd/compiler.py` | 188-203 | `HIPBackend.make_ttir()` — AMD TTIR 管道 |

**NVIDIA 与 AMD 的差异：**

| Pass | NVIDIA | AMD |
|------|--------|-----|
| `rewrite_tensor_descriptor_to_pointer` | 仅 SM<90 | 总是执行 |
| `triton_licm` | 不在 make_ttir 中 | 在 make_ttir 中（cse 之后） |

### 2.3 调用链

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/nvidia/compiler.py` | 537-541 | `add_stages()` — 注册 `stages["ttir"] = make_ttir` |
| `python/triton/compiler/compiler.py` | 323-347 | compile() 中 `for ext, compile_ir in stages.items()` 执行 |

---

## 三、涉及的 C++ 代码

### 3.1 通过 `passes.ttir.*` 调用的 C++ Pass

| Python 调用 | C++ 工厂函数 | C++ 实现文件 |
|-------------|-------------|-------------|
| `passes.common.add_inliner(pm)` | `createInlinerPass` | 上游 MLIR |
| `passes.ttir.add_rewrite_tensor_pointer(pm)` | `createTritonRewriteTensorPointer` | `lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp` |
| `passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)` | `createTritonRewriteTensorDescriptorToPointer` | `lib/Dialect/Triton/Transforms/RewriteTensorDescriptorToPointer.cpp` |
| `passes.common.add_canonicalizer(pm)` | `createCanonicalizerPass` | 上游 MLIR |
| `passes.ttir.add_combine(pm)` | `createTritonCombineOps` | `lib/Dialect/Triton/Transforms/Combine.cpp` + `Combine.td` |
| `passes.ttir.add_reorder_broadcast(pm)` | `createTritonReorderBroadcast` | `lib/Dialect/Triton/Transforms/ReorderBroadcast.cpp` |
| `passes.common.add_cse(pm)` | `createCSEPass` | 上游 MLIR |
| `passes.common.add_symbol_dce(pm)` | `createSymbolDCEPass` | 上游 MLIR |
| `passes.ttir.add_loop_unroll(pm)` | `createTritonLoopUnroll` | `lib/Dialect/Triton/Transforms/LoopUnroll.cpp` |
| `passes.ttir.add_triton_licm(pm)` (AMD 额外) | `createTritonLoopInvariantCodeMotion` | `lib/Dialect/Triton/Transforms/LoopInvariantCodeMotion.cpp` |

### 3.2 `ir.pass_manager` 绑定

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/src/passes.cc` | 28-53 | `init_triton_passes_common()` + `init_triton_passes_ttir()` — 绑定 pass |
| `python/src/passes.h` | 1-44 | `ADD_PASS_WRAPPER_N` 宏 — 简化绑定 |

---

## 四、涉及的 MLIR 代码

| 目录 | 相关文件 | 功能 |
|------|---------|------|
| `include/triton/Dialect/Triton/Transforms/Passes.td` | - | TTIR pass 的 TableGen 声明 |
| `lib/Dialect/Triton/Transforms/Combine.cpp` | 299 行 | combine pass 实现（7 种模式优化） |
| `lib/Dialect/Triton/Transforms/Combine.td` | 24 行 | addptr 链合并的 TableGen 模式 |
| `lib/Dialect/Triton/Transforms/ReorderBroadcast.cpp` | - | broadcast 重排优化 |
| `lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp` | - | tensor pointer 退化 |
| `lib/Dialect/Triton/Transforms/RewriteTensorDescriptorToPointer.cpp` | - | tensor descriptor 退化 |
| `lib/Dialect/Triton/Transforms/LoopUnroll.cpp` | - | 循环展开 |
| `lib/Dialect/Triton/Transforms/LoopInvariantCodeMotion.cpp` | - | 循环不变代码外提 |

---

## 五、Pass 功能说明

| Pass | 输入 | 输出 | 功能 |
|------|------|------|------|
| `inliner` | tt + arith + scf | 同左 | 内联所有函数调用 |
| `rewrite_tensor_pointer` | tt | tt | `tt.make_tensor_ptr` → 传统指针语义 |
| `rewrite_tensor_descriptor_to_pointer` | tt | tt | `tt.make_tensor_descriptor` → 传统指针语义 |
| `canonicalizer` | 任意 | 同左 | 常量折叠、操作简化 |
| `combine` | tt | tt | 7 种模式优化：dot+add 融合、addptr 链合并、select+load 融合、broadcast→dot 等 |
| `reorder_broadcast` | tt | tt | delay broadcast after elementwise |
| `cse` | 任意 | 同左 | 公共子表达式消除 |
| `symbol_dce` | 任意 | 同左 | 删除未使用的函数/符号 |
| `loop_unroll` | tt | tt | 展开带 `tt.loop_unroll_factor` 的循环 |
| `triton_licm` (AMD) | tt | tt | 循环不变量外提（含 masked load） |

---

## 六、进入条件、参数和输出

### 进入条件
- `compile()` 中 `stages["ttir"]` stage 被执行
- 输入必须是 `ASTSource`（Python AST）而非 `IRSource`

### 参数 (NVIDIA)

| 参数 | 类型 | 说明 |
|------|------|------|
| `mod` | `ir.module` | `mlir::ModuleOp` — Python AST 转换后的 MLIR module |
| `metadata` | dict | 编译元数据字典 |
| `opt` | CUDAOptions | NVIDIA 编译选项 |
| `capability` | int | GPU 计算能力 (如 89, 90) |

### 输出
| 输出 | 说明 |
|------|------|
| `mod` | 优化后的 `mlir::ModuleOp`，仍是 tt dialect，未引入 GPU layout |