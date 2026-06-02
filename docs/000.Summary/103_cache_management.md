# 03 - 缓存管理分析

> 分析内核缓存机制：缓存结构、key 计算、缓存命中/未命中路径

---

## 一、整体流程

```
JITFunction.run() 被调用
    │
    ├─→ device = driver.active.get_current_device()   ← 当前设备索引
    │
    ├─→ self.device_caches[device]                     ← 按 device 索引的缓存
    │   └─→ 首次 → create_binder() 执行 target 检测 + backend 选择
    │   └─→ 后续 → 直接返回缓存的 (kernel_cache, key_cache, target, backend, binder)
    │
    ├─→ binder(*args, **kwargs)                        ← 参数绑定与特化
    │   └─→ 返回 (bound_args, specialization, options)
    │
    ├─→ key = compute_cache_key(kernel_key_cache, specialization, options)
    │
    ├─→ kernel = kernel_cache.get(key, None)
    │   │
    │   ├─ kernel is not None → 缓存命中 → 直接执行
    │   │
    │   └─ kernel is None → 缓存未命中
    │       ├─→ _do_compile() → compile() → 编译管道
    │       └─→ kernel_cache[key] = CompiledKernel   ← 存入缓存
    │
    └─→ kernel.run(grid, stream, ...)   ← 启动 GPU 内核
```

---

## 二、涉及的 Python 代码

### 2.1 缓存数据结构

| 文件 | 行号 | 代码 | 说明 |
|------|------|------|------|
| `python/triton/runtime/jit.py` | 771 | `self.device_caches = defaultdict(self.create_binder)` | 按 device 索引的惰性缓存，`defaultdict` 确保首次访问时自动调用 `create_binder()` |
| `python/triton/runtime/jit.py` | 707 | `kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]` | 解包缓存元组 |
| `python/triton/runtime/jit.py` | 669 | `return {}, {}, target, backend, binder` | `create_binder()` 返回值：`(kernel_cache: dict, key_cache: dict, target, backend, binder)` |

**缓存层次结构：**

```
JITFunction.device_caches
    └─ dict: device_id →
        (kernel_cache: dict, key_cache: dict, target, backend, binder)
            │               │
            │               └─ key_cache: {raw_key → str_cache_key}
            │
            └─ kernel_cache: {str_cache_key → CompiledKernel}
```

### 2.2 Key 计算

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/runtime/jit.py` | 563-584 | `compute_cache_key()` — 计算缓存 key |
| `python/triton/runtime/jit.py` | 564 | `key = (tuple(specialization), str(options))` — 原始 key |
| `python/triton/runtime/jit.py` | 568-582 | 将 `JITCallable` 对象替换为 hash，生成字符串 cache key |
| `python/triton/runtime/jit.py` | 583 | `kernel_key_cache[key] = cache_key` — 缓存 key 映射 |

**`compute_cache_key()` 逻辑：**

```python
def compute_cache_key(kernel_key_cache, specialization, options):
    key = (tuple(specialization), str(options))     # 原始 key
    cache_key = kernel_key_cache.get(key, None)
    if cache_key is not None:
        return cache_key                             # 已有映射，直接返回

    # 替换 JITCallable 为 hash，使源码改变时 key 自动失效
    def replace_callables(obj):
        if isinstance(obj, JITCallable):
            return obj.cache_key
        ...
    cache_key = str(replace_callables(specialization)) + str(options)
    kernel_key_cache[key] = cache_key                # 保存映射
    return cache_key
```

### 2.3 缓存命中/未命中

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/runtime/jit.py` | 712-713 | `key = compute_cache_key(...)` + `kernel_cache.get(key, None)` — 查缓存 |
| `python/triton/runtime/jit.py` | 716-722 | 未命中 → `_do_compile()` → 编译 → `kernel_cache[key] = kernel`（在 `_do_compile` 中） |
| `python/triton/runtime/jit.py` | 849 | 编译完成后 `kernel_cache[key] = kernel` — 存入缓存 |

### 2.4 全局变量检查

| 文件 | 行号 | 功能 |
|------|------|------|
| `python/triton/runtime/jit.py` | 724-729 | 检查 `used_global_vals` 是否有变化，确保缓存有效 |

---

## 三、涉及的 C++ 代码

本阶段不涉及 C++ 代码。缓存管理是纯 Python 逻辑。

---

## 四、涉及的 MLIR 代码

本阶段不涉及 MLIR 代码。

---

## 五、进入条件、参数和输出

### 进入条件
`JITFunction.run()` 被调用（用户通过 `kernel[grid](*args)` 调用内核）。

### 参数

| 变量 | 来源 | 说明 |
|------|------|------|
| `device` | `driver.active.get_current_device()` | 当前 GPU 设备索引 |
| `kernel_cache` | `self.device_caches[device]` | 存储 `key → CompiledKernel` 的字典 |
| `kernel_key_cache` | `self.device_caches[device]` | 存储 `specialization+options → cache_key` 的字典 |
| `specialization` | `binder(*args)` 返回 | 参数类型特化信息 |
| `options` | `binder(*args)` 返回 | 编译选项字典 |
| `key` | `compute_cache_key(...)` | 最终的缓存 key（字符串） |

### 输出（两种路径）

**缓存命中：**
- 跳过编译，直接执行
- 从 `kernel_cache` 获取已编译的 `CompiledKernel`

**缓存未命中：**
- 调用 `_do_compile()` → 触发完整编译管道
- 编译结果存入 `kernel_cache[key]`
- 返回 `CompiledKernel` 用于执行