# 10 - `make_cubin` 阶段分析

> 分析 CUBIN 生成阶段：PTX 汇编 → CUDA 二进制

---

## 一、整体流程

```
输入: PTX 汇编文本
    │
    ├─→ 获取 ptxas 编译器路径
    ├─→ 写入临时 .ptx 文件
    ├─→ 调用 ptxas 编译:
    │   ptxas [options] --gpu-name=sm_{capability} input.ptx -o output.o
    │
    ├─→ 读取编译结果 .o 文件
    ├─→ 清理临时文件
    │
输出: .cubin 二进制数据 (bytes)
```

---

## 二、涉及的 Python 代码

### 2.1 `make_cubin()`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/nvidia/compiler.py` | 461-535 | `CUDABackend.make_cubin()` — PTX → CUDA 二进制 |

**代码：**

```python
def make_cubin(self, src, metadata, opt, capability):
    ptxas = get_ptxas(self.target.arch).path
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
        tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
        fsrc.write(src)
        fsrc.flush()
        fbin = fsrc.name + '.o'

        debug_info = [...]      # -lineinfo / -g 等
        fmad = [] if opt.enable_fp_fusion else ["--fmad=false"]
        arch = sm_arch_from_capability(capability)
        disable_opt = ['--opt-level', '0'] if knobs.nvidia.disable_ptxas_opt else []
        ptx_extra_options = opt.ptx_options.split(" ") if opt.ptx_options else []

        ptxas_cmd = [
            ptxas, *debug_info, *fmad, '-v', *disable_opt,
            *ptx_extra_options,
            f'--gpu-name={arch}', fsrc.name, '-o', fbin
        ]
        subprocess.run(ptxas_cmd, check=True, ...)
        with open(fbin, 'rb') as f:
            cubin = f.read()
    return cubin
```

### 2.2 辅助函数

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/nvidia/compiler.py` | 34-35 | `get_ptxas()` — 根据架构选择 ptxas 或 ptxas-blackwell |

---

## 三、涉及的 C++/LLVM 代码

`make_cubin` 不直接调用 C++ 代码。它通过 `subprocess` 调用外部的 **`ptxas`** 编译器（NVIDIA CUDA 工具链的一部分）。

---

## 四、涉及的 MLIR 代码

本阶段不涉及 MLIR。

---

## 五、进入条件、参数和输出

### 进入条件
- NVIDIA 后端 `compile()` 中 `stages["cubin"]` stage 被执行
- 仅 NVIDIA 后端有此阶段（AMD 对应 make_hsaco）

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | str | PTX 汇编文本 |
| `metadata` | dict | 编译元数据 |
| `opt` | CUDAOptions | 编译选项 |
| `capability` | int | GPU 计算能力 |

### 输出

| 输出 | 说明 |
|------|------|
| `cubin` (bytes) | CUDA 二进制文件内容，可被 `cuModuleLoadData` 加载 |