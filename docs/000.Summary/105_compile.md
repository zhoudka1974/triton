# 05 - `compile()` 函数分析

> 分析 `triton.compiler.compile()` — 编译管道的核心编排函数

---

## 一、整体流程

```
compile(src, target, options)  (compiler.py:226)
    │
    ├─→ 0. 确定 target 和 backend
    │   ├─ target 未指定 → 自动检测当前 GPU
    │   └─ backend = make_backend(target)  ← 选择 CUDABackend / HIPBackend
    │
    ├─→ 1. 缓存检查 (文件级缓存)
    │   ├─ hash = SHA256(配置key)
    │   ├─ metadata_path 存在 → 直接返回 CompiledKernel (缓存命中)
    │   └─ 否则 → 继续编译
    │
    ├─→ 2. 注册编译 Stage
    │   stages = dict()
    │   backend.add_stages(stages, options, src.language)
    │   ├─ NVIDIA: ttir → ttgir → llir → ptx → cubin
    │   └─ AMD:    ttir → ttgir → llir → amdgcn → hsaco
    │
    ├─→ 3. 初始化 MLIR Context
    │   context = ir.context()
    │   ir.load_dialects(context)         ← 加载 Triton + MLIR dialect
    │   backend.load_dialects(context)    ← 加载后端专属 dialect
    │
    ├─→ 4. Python AST → TTIR
    │   module = src.make_ir(target, ...)
    │   └─→ ASTSource.make_ir()
    │       └─→ code_generator.py: ast_to_ttir()
    │
    └─→ 5. 依次执行各 Stage
        for ext, compile_ir in stages.items():
            module = compile_ir(module, metadata)
            └─→ make_ttir → make_ttgir → make_llir → make_ptx → make_cubin
        ↓
        返回 CompiledKernel(src, metadata_group, hash)
```

---

## 二、涉及的 Python 代码

### 2.1 `compile()` 函数

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/compiler/compiler.py` | 226-360 | `compile()` — 编译入口，编排整个管道 |
| `python/triton/compiler/compiler.py` | 231-232 | 自动检测 target（如未指定） |
| `python/triton/compiler/compiler.py` | 234 | `backend = make_backend(target)` — 选择后端 |
| `python/triton/compiler/compiler.py` | 286-288 | `backend.add_stages(stages, ...)` — 注册编译阶段 |
| `python/triton/compiler/compiler.py` | 296-299 | 初始化 MLIR Context + 加载 dialect |
| `python/triton/compiler/compiler.py` | 304 | `src.make_ir()` — Python AST → MLIR |
| `python/triton/compiler/compiler.py` | 323-347 | for 循环依次执行各 stage |
| `python/triton/compiler/compiler.py` | 360 | 返回 `CompiledKernel` |

### 2.2 `make_backend()` 后端选择

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/compiler/compiler.py` | 363-368 | `make_backend()` — 遍历 backends 匹配 target |

### 2.3 缓存管理（文件级）

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/compiler/compiler.py` | 245-276 | 文件级缓存检查（与 `jit.py` 的内存缓存不同层） |
| `python/triton/compiler/compiler.py` | 246-247 | 计算配置 hash |
| `python/triton/compiler/compiler.py` | 265-276 | 缓存命中 → 直接返回 `CompiledKernel` |

### 2.4 `ASTSource` 和 `IRSource`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/compiler/compiler.py` | 52-84 | `ASTSource` — Python AST 源 |
| `python/triton/compiler/compiler.py` | 87-.. | `IRSource` — 从文件加载 IR |

### 2.5 `CompiledKernel`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/compiler/compiler.py` | 404-.. | `CompiledKernel` — 编译结果封装 |

### 2.6 `ast_to_ttir()`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/compiler/code_generator.py` | 1600-1639 | `ast_to_ttir()` — Python AST → MLIR ModuleOp |

---

## 三、涉及的 C++ 代码

本阶段是第一个真正调用 C++ 代码的阶段：

### 3.1 MLIR Context 和 Dialect 注册

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/src/ir.cc` | 363-378 | `ir.load_dialects()` — 注册所有 dialect 到 MLIRContext |

### 3.2 IR Builder

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/src/ir.cc` | - | `ir.builder` — 包装 `mlir::OpBuilder`，用于生成 MLIR 操作 |
| `python/src/ir.h` | 1-101 | `TritonOpBuilder` — Triton 增强的 OpBuilder |

### 3.3 Stage 函数的 C++ 调用

每个 stage（make_ttir, make_ttgir 等）内部创建 `ir.pass_manager` 并调用 `passes.*.add_*(pm)`，这些 pass 是 C++ 实现通过 pybind11 暴露给 Python 的：

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/src/passes.cc` | 28-37 | 通用 pass 绑定 |
| `python/src/passes.cc` | 39-53 | TTIR pass 绑定 |
| `python/src/passes.cc` | 55-97 | TTGIR pass 绑定 |
| `python/src/passes.cc` | 99-106 | 转换 pass 绑定 |
| `python/src/passes.cc` | 125-133 | 组装所有 pass 子模块 |

---

## 四、涉及的 MLIR 代码

MLIR 代码被各 stage 函数调用。简要列表：

| 目录 | 内容 |
|------|------|
| `lib/Dialect/Triton/Transforms/*.cpp` | TTIR pass 实现（combine, reorder_broadcast, loop_unroll 等） |
| `lib/Dialect/TritonGPU/Transforms/*.cpp` | TTGIR pass 实现（coalesce, accelerate_matmul, pipeline 等） |
| `lib/Dialect/TritonNvidiaGPU/Transforms/*.cpp` | NVIDIA 专属 pass 实现 |
| `lib/Conversion/TritonGPUToLLVM/*.cpp` | TritonGPU → LLVM lowering |
| `lib/Analysis/*.cpp` | 内存分配等分析 |

---

## 五、`make_ttir` 阶段管道示例（NVIDIA）

```python
def make_ttir(mod, metadata, opt, capability):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)
    if capability // 10 < 9:
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod, 'make_ttir')
    return mod
```

完整管道见后续文件 06-10。

---

## 六、进入条件、参数和输出

### 进入条件
- `_do_compile()` 调用 `self.compile(src, target=target, options=...)`

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | ASTSource / IRSource | 编译源（Python AST 或 IR 文件路径） |
| `target` | GPUTarget | 目标 GPU（如 `("cuda", 89, 32)`），可选，未指定则自动检测 |
| `options` | dict | 编译选项字典，被 backend.parse_options() 解析 |

### 输出

| 返回值 | 说明 |
|--------|------|
| `CompiledKernel` | 包含所有 stage 的编译结果（asm, metadata, function 句柄） |

`CompiledKernel` 关键属性：

```python
class CompiledKernel:
    asm = {
        "ttir": "...",      # TTIR 文本
        "ttgir": "...",     # TTGIR 文本
        "llir": "...",      # LLVM IR 文本
        "ptx": "...",       # PTX 汇编 (NVIDIA)
        "cubin": b"..."     # CUDA 二进制 (NVIDIA)
    }
    metadata = { ... }      # num_warps, shared, name 等
    function                # CUDA/HIP 函数句柄
    packed_metadata         # 启动参数
```