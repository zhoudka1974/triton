# 12 - `make_amdgcn` 阶段分析

> 分析 AMD GCN 汇编生成阶段：LLVM IR → AMD GCN 汇编

---

## 一、整体流程

```
输入: LLVM IR 文本字符串
    │
    ├─→ 提取内核函数名 (正则匹配 amdgpu_kernel)
    ├─→ llvm.translate_to_mir(...)      ← 生成 MachineIR
    ├─→ llvm.dump_sched_dag(...)        ← 调度 DAG 转储 (调试)
    ├─→ llvm.translate_to_asm(...)      ← LLVM IR → AMD GCN 汇编
    │   triple = 'amdgcn-amd-amdhsa'
    │   proc = options.arch → "gfx942"
    │   features 取决于架构
    │
输出: AMD GCN 汇编文本
```

---

## 二、涉及的 Python 代码

### 2.1 `make_amdgcn()`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/amd/compiler.py` | 442-464 | `HIPBackend.make_amdgcn()` — LLVM IR → AMD GCN 汇编 |

**代码：**

```python
def make_amdgcn(self, src, metadata, options):
    names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
    assert len(names) == 1
    metadata["name"] = names[0]
    flags = []
    features = '-real-true16' if 'gfx11' in options.arch else ''
    amdgcn = llvm.translate_to_asm(
        src, amd.TARGET_TRIPLE, options.arch, features,
        flags, options.enable_fp_fusion, False)
    return amdgcn
```

### 2.2 辅助功能

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/amd/compiler.py` | 455-456 | `llvm.translate_to_mir()` — 生成 MachineIR（调试用途） |
| `python/triton/backends/amd/compiler.py` | 457-458 | `llvm.dump_sched_dag()` — 调度 DAG 转储（调试用途） |

---

## 三、涉及的 C++/LLVM 代码

| Python 调用 | 功能 |
|-------------|------|
| `llvm.translate_to_asm(src, triple, proc, features, ...)` | 调用 LLVM AMDGPU Target 将 LLVM IR 翻译为 AMD GCN 汇编 |
| `llvm.translate_to_mir(src, ...)` | 调用 LLVM AMDGPU Target 生成 MachineIR |

---

## 四、涉及的 MLIR 代码

本阶段不涉及 MLIR。

---

## 五、进入条件、参数和输出

### 进入条件
- AMD 后端 `compile()` 中 `stages["amdgcn"]` stage 被执行
- 仅 AMD 后端有此阶段（NVIDIA 对应 make_ptx）

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | str | LLVM IR 文本 |
| `metadata` | dict | 编译元数据（会被写入内核名） |
| `options` | HIPOptions | AMD 编译选项（含 arch） |

### 输出

| 输出 | 说明 |
|------|------|
| `amdgcn` (str) | AMD GCN 汇编文本 |
| `metadata["name"]` | 内核函数名称 |