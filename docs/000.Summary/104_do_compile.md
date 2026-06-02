# 04 - `_do_compile` 函数分析

> 分析 `JITFunction._do_compile()` 的包装逻辑、进入条件、参数和输出

---

## 一、整体流程

```
_do_compile() 调用 (jit.py:826)
    │
    ├─→ hook 检查: 是否被 jit_cache_hook 拦截
    │
    ├─→ src = ASTSource(self, signature, constexprs, attrs)
    │   └─→ 包装内核信息: fn 对象、签名、常量、属性
    │
    ├─→ 判断异步编译模式:
    │   │
    │   ├─ async_mode 活跃:
    │   │   ├─→ cache_key = get_cache_key(src, backend, options, ...)
    │   │   └─→ async_mode.submit(cache_key, async_compile, finalize)
    │   │
    │   └─ 同步模式:
    │       ├─→ kernel = self.compile(src, target, options)  ← 调用 compile()
    │       └─→ kernel_cache[key] = kernel
    │
    └─→ return kernel
```

**`_do_compile` 本身不做编译，只是包装委托给 `compile()` 并处理缓存和异步。**

---

## 二、涉及的 Python 代码

### 2.1 `_do_compile()` 函数

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/runtime/jit.py` | 826-853 | `_do_compile()` — 编译入口包装函数 |
| `python/triton/runtime/jit.py` | 829 | 调用 `jit_cache_hook`，如果返回 True 则跳过编译 |
| `python/triton/runtime/jit.py` | 831 | 创建 `ASTSource` 对象 |
| `python/triton/runtime/jit.py` | 834-847 | 异步编译路径 |
| `python/triton/runtime/jit.py` | 849-852 | 同步编译路径 |

### 2.2 `ASTSource` 类

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/compiler/compiler.py` | 52-84 | `ASTSource` 类 — 包装内核 AST 及其元数据 |
| `python/triton/compiler/compiler.py` | 54-69 | `__init__()` — 设置 `self.fn`(函数), `self.signature`(签名), `self.constants`(常量), `self.attrs`(属性) |
| `python/triton/compiler/compiler.py` | 57 | `self.ext = "ttir"` — 源类型标记 |
| `python/triton/compiler/compiler.py` | 71-76 | `hash()` — 计算源哈希用于缓存 |
| `python/triton/compiler/compiler.py` | 78-81 | `make_ir()` — 调用 `ast_to_ttir()` 将 Python AST 转换为 MLIR |

### 2.3 异步编译

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/runtime/_async_compile.py` | - | `AsyncCompileMode` — 异步编译模式管理 |

---

## 三、涉及的 C++ 代码

本阶段不使用 C++ 代码。C++ 调用在 `_do_compile` 调用的 `compile()` 函数中触发。

---

## 四、涉及的 MLIR 代码

本阶段不使用 MLIR 代码。

---

## 五、进入条件、参数和输出

### 进入条件
- 缓存未命中（`kernel_cache.get(key) is None`）
- `jit_cache_hook` 未拦截（返回 `None`）

### 参数

| 参数 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `key` | str | `compute_cache_key()` | 缓存 key |
| `signature` | dict | `_pack_args()` | 参数类型签名 |
| `device` | int | `driver.active.get_current_device()` | 当前 GPU 设备索引 |
| `constexprs` | dict | `_pack_args()` | 编译时常量值 |
| `options` | CUDAOptions/HIPOptions | `_pack_args()` | 编译选项 |
| `attrs` | dict | 参数属性 | 指针对齐等信息 |
| `warmup` | bool | `run()` 参数 | 是否预热模式 |

### 输出

| 返回值 | 说明 |
|--------|------|
| `CompiledKernel` | 编译完成的内核对象 |
| `None` | 被 `jit_cache_hook` 拦截 |

### 副作用
- 编译结果存入 `kernel_cache[key]`