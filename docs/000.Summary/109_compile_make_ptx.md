# 09 - `make_ptx` 阶段分析

> 分析 PTX 生成阶段：LLVM IR → PTX 汇编

---

## 一、整体流程

```
输入: LLVM IR 文本字符串
    │
    ├─→ llvm.translate_to_asm(src, triple, proc, features, ...)
    │   triple = 'nvptx64-nvidia-cuda'
    │   proc = sm_arch_from_capability(capability) → "sm_89"
    │   features = get_features(...) → "+ptx86"
    │
    ├─→ 后处理: 修改 .version 和 .target 指令
    ├─→ 可选: 移除 debug flag (优化)
    │
输出: PTX 汇编文本
```

---

## 二、涉及的 Python 代码

### 2.1 `make_ptx()`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/nvidia/compiler.py` | 435-459 | `CUDABackend.make_ptx()` — LLVM IR → PTX 汇编 |

**代码：**

```python
def make_ptx(self, src, metadata, opt, capability):
    ptx_version = get_ptx_version_from_options(opt, self.target.arch)

    triple = 'nvptx64-nvidia-cuda'
    proc = sm_arch_from_capability(capability)
    features = get_features(opt, self.target.arch)
    flags = ["nvptx-mad-wide-opt"]
    ret = llvm.translate_to_asm(src, triple, proc, features, flags,
                                 opt.enable_fp_fusion, False)

    # 提取内核名称
    names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
    metadata["name"] = names[0]

    # 修改 PTX 版本和架构版本
    ptx_version = f'{ptx_version//10}.{ptx_version%10}'
    ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret)
    ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret)
    return ret
```

### 2.2 辅助函数

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/nvidia/compiler.py` | 99-102 | `sm_arch_from_capability()` — 计算能力→SM 名（如 89→"sm_89"） |
| `python/triton/backends/nvidia/compiler.py` | 79-90 | `get_features()` — 计算 PTX 特性字符串 |
| `python/triton/backends/nvidia/compiler.py` | 47-68 | `ptx_get_version()` — 获取最高支持的 PTX 版本 |
| `python/triton/backends/nvidia/compiler.py` | 71-76 | `get_ptx_version_from_options()` — 获取 PTX 版本号 |

---

## 三、涉及的 C++/LLVM 代码

| Python 调用 | 功能 |
|-------------|------|
| `llvm.translate_to_asm(...)` | 调用 LLVM 的 NVPTX Target 将 LLVM IR 翻译为 PTX 汇编 |
| `llvm.init_targets()` | 初始化 LLVM 目标（在 make_llir 中已调用） |

---

## 四、涉及的 MLIR 代码

本阶段不涉及 MLIR。输入是 LLVM IR（已从 MLIR LLVM dialect 转换到 LLVM IR）。

---

## 五、进入条件、参数和输出

### 进入条件
- NVIDIA 后端 `compile()` 中 `stages["ptx"]` stage 被执行
- 仅 NVIDIA 后端有此阶段（AMD 对应 make_amdgcn）

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | str | LLVM IR 文本 |
| `metadata` | dict | 编译元数据（会被写入内核名） |
| `opt` | CUDAOptions | 编译选项 |
| `capability` | int | GPU 计算能力 |

### 输出

| 输出 | 说明 |
|------|------|
| `ret` (str) | PTX 汇编文本 |
| `metadata["name"]` | 内核函数名称 |