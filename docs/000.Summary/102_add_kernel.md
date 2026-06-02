# 02 - @triton.jit 装饰器定义阶段分析

> 分析用户编写 `@triton.jit` 装饰一个函数时，`JITFunction` 的初始化过程

---

## 一、整体流程

```
用户编写:
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...

执行:
    triton.jit(add_kernel)
    ↓
jit() 函数 (jit.py:886)
    ↓                  ┌─ 解释模式 → InterpretedFunction
    ↓ 判断 knobs.runtime.interpret
    ↓                  └─ 编译模式 → JITFunction(fn, ...)
    ↓
JITFunction.__init__(fn) (jit.py:751)
    ├─→ 调用 super().__init__(fn) → KernelInterface
    ├─→ self.params = []   ← 解析函数签名
    │   └─→ for param in self.signature.parameters.values():
    │       └─→ KernelParam(i, param, do_not_specialize, ...)
    ├─→ self.device_caches = defaultdict(self.create_binder)  ← 惰性
    └─→ self.kernel = None
```

**关键：此时不检测 GPU 也不选择 backend，这些延后到首次调用时才执行。**

---

## 二、涉及的 Python 代码

### 2.1 装饰器函数

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/runtime/jit.py` | 886-935 | `jit()` 装饰器函数，判断解释/编译模式，返回 `JITFunction` |
| `python/triton/runtime/jit.py` | 915-932 | 内部 `decorator()` 函数，创建 `JITFunction` 实例 |
| `python/triton/runtime/jit.py` | 355-371 | `KernelInterface` 基类，定义 `__getitem__`, `warmup`, `run` |
| `python/triton/runtime/jit.py` | 364-370 | `__getitem__` — `fn[grid]` 调用约定 |

### 2.2 JITFunction 初始化

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/runtime/jit.py` | 751-785 | `JITFunction.__init__()` — 解析签名，建立缓存结构 |
| `python/triton/runtime/jit.py` | 764-768 | 遍历参数创建 `KernelParam` 列表 |
| `python/triton/runtime/jit.py` | 771 | `self.device_caches = defaultdict(self.create_binder)` — 惰性 bind |
| `python/triton/runtime/jit.py` | 658-669 | `create_binder()` — 惰性初始化函数（首次调用时执行） |
| `python/triton/runtime/jit.py` | 662-664 | `create_binder()` 内部：获取 target、选择 backend |
| `python/triton/runtime/jit.py` | 668 | `create_function_from_signature()` — 创建参数绑定函数 |

### 2.3 KernelParam 类

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/runtime/jit.py` | 316-347 | `KernelParam` 类，描述内核参数信息 |

### 2.4 惰性初始化核心

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/runtime/jit.py` | 658-669 | `create_binder()` — GPU target 检测 + backend 选择 + 参数绑定器创建 |

**`create_binder()` 的代码：**

```python
def create_binder(self):
    from ..compiler import CompiledKernel, compile, ASTSource, make_backend
    target = driver.active.get_current_target()     # 检测 GPU (如 GPUTarget("cuda", 89, 32))
    backend = make_backend(target)                  # 选择 CUDABackend 或 HIPBackend
    self.CompiledKernel = CompiledKernel
    self.compile = compile
    self.ASTSource = ASTSource
    binder = create_function_from_signature(self.signature, self.params, backend)
    return {}, {}, target, backend, binder
           # ↑        ↑
           # kernel_  key_
           # cache    cache
```

---

## 三、涉及的 C++ 代码

在本阶段（装饰器定义），C++ 代码**不参与执行**。C++ 扩展 `libtriton.so` 已在 import 阶段加载就绪，但只有到编译阶段才会实际调用 C++ 的 MLIR builder 和 pass manager。

| 阶段 | C++ 使用 |
|------|---------|
| `@triton.jit` 装饰时 | 不使用 C++ 代码 |
| 首次调用时 (`create_binder`) | 不使用 C++ 代码（纯 Python），但通过 pybind11 绑定的 `ir.pass_manager` 等已可用 |

---

## 四、涉及的 MLIR 代码

本阶段**不涉及** MLIR 代码执行。MLIR 相关的操作（Dialect 注册、Pass 创建、IR 生成）全部延后到 `compile()` 阶段。

---

## 五、进入条件、参数和输出

### 进入条件
用户编写 Python 函数并用 `@triton.jit` 装饰。

### 参数
| 参数 | 类型 | 说明 |
|------|------|------|
| `fn` | Callable | 被装饰的 Python 函数 |
| `version` | int | JIT 版本号 |
| `do_not_specialize` | list[int\|str] | 不进行参数特化的参数索引/名称 |
| `do_not_specialize_on_alignment` | list[int\|str] | 不对齐进行特化的参数 |
| `debug` | bool | 调试模式 |
| `noinline` | bool | 禁止内联 |
| `repr` | Callable | 自定义 repr 函数 |
| `launch_metadata` | Callable | 启动元数据回调 |

### 输出
- `JITFunction` 实例
- 关键属性：
  - `self.params` — `KernelParam` 列表，描述所有参数
  - `self.device_caches` — 按 device 索引的缓存字典，值为 `(kernel_cache, key_cache, target, backend, binder)` 元组
  - `self.kernel` — 初始为 None，编译后设为 `CompiledKernel`