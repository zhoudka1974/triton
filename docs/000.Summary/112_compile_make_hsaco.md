# 13 - `make_hsaco` 阶段分析

> 分析 HSACO 生成阶段：AMD GCN 汇编 → AMD 二进制

---

## 一、整体流程

```
输入: AMD GCN 汇编文本
    │
    ├─→ amd.assemble_amdgcn(src, arch, target_features)
    │   → 调用 LLVM AMDGPU Target 汇编为二进制
    │
    ├─→ amd.link_hsaco(tmp_in, tmp_out)
    │   → 链接生成最终 .hsaco 文件
    │
    ├─→ 读取链接后的 hsaco 二进制
    │
输出: .hsaco 二进制数据 (bytes)
```

---

## 二、涉及的 Python 代码

### 2.1 `make_hsaco()`

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/backends/amd/compiler.py` | 466-479 | `HIPBackend.make_hsaco()` — AMD GCN 汇编 → HSACO 二进制 |

**代码：**

```python
def make_hsaco(self, src, metadata, options):
    target_features = ''
    if knobs.compilation.enable_asan:
        target_features = '+xnack'
    hsaco = amd.assemble_amdgcn(src, options.arch, target_features)
    with tempfile.NamedTemporaryFile() as tmp_out:
        with tempfile.NamedTemporaryFile() as tmp_in:
            with open(tmp_in.name, "wb") as fd_in:
                fd_in.write(hsaco)
            amd.link_hsaco(tmp_in.name, tmp_out.name)
        with open(tmp_out.name, "rb") as fd_out:
            ret = fd_out.read()
    return ret
```

### 2.2 AMD C++ 扩展

| Python 调用 | C++ 功能 |
|-------------|---------|
| `amd.assemble_amdgcn(...)` | 调用 LLVM AMDGPU Target 汇编 |
| `amd.link_hsaco(...)` | 调用链接器生成 .hsaco 文件 |

---

## 三、涉及的 C++/LLVM 代码

| Python 调用 | 功能 |
|-------------|------|
| `amd.assemble_amdgcn(...)` | C++ 绑定中调用 LLVM AMDGPU 后端的汇编功能 |
| `amd.link_hsaco(...)` | C++ 绑定中调用链接功能 |
| `amd.TARGET_TRIPLE` | `"amdgcn-amd-amdhsa"` |

---

## 四、涉及的 MLIR 代码

本阶段不涉及 MLIR。

---

## 五、进入条件、参数和输出

### 进入条件
- AMD 后端 `compile()` 中 `stages["hsaco"]` stage 被执行
- 仅 AMD 后端有此阶段（NVIDIA 对应 make_cubin）

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | str | AMD GCN 汇编文本 |
| `metadata` | dict | 编译元数据 |
| `options` | HIPOptions | AMD 编译选项 |

### 输出

| 输出 | 说明 |
|------|------|
| `ret` (bytes) | HSACO 二进制文件内容，可被 `hipModuleLoadData` 加载 |