import sys
sys.path.insert(0, '/home/zhoudka/work/code/triton/python')

import os
os.environ['TRITON_AST_LOG_NAME'] = 'ast_to_ttir_trace'

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir
from triton.compiler.compiler import ASTSource, make_backend

# 统一的编译环境
target = GPUTarget('cuda', 89, 32)
backend = make_backend(target)
options = backend.parse_options({'arch': 'sm89'})
context = ir.context()
ir.load_dialects(context)
backend.load_dialects(context)
codegen_fns = backend.get_codegen_implementation(options)
module_map = backend.get_module_map()

kernel_results = []


def compile_kernel(name, fn, sig, constexprs=None, attrs=None):
    """编译单个内核，记录结果"""
    constexprs = constexprs or {}
    attrs = attrs or {}
    print(f"\n{'='*60}")
    print(f"Kernel: {name}")
    print(f"{'='*60}")
    try:
        src = ASTSource(fn, sig, constexprs, attrs)
        module = src.make_ir(target, options, codegen_fns, module_map, context)
        assert module.verify(), "Module verification failed"
        print(f"  ✓ OK")
        kernel_results.append((name, "OK", None))
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        kernel_results.append((name, "FAIL", str(e)[:100]))


# ========== 1. 向量加法 (element-wise) ==========
@triton.jit
def kernel_vec_add(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    m = off < N
    x = tl.load(x_ptr + off, mask=m)
    y = tl.load(y_ptr + off, mask=m)
    z = x + y
    tl.store(out_ptr + off, z, mask=m)

compile_kernel("vec_add", kernel_vec_add,
               {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32', 'N': 'i32'})


# ========== 2. 矩阵乘法 (matmul) ==========
@triton.jit
def kernel_matmul(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, 32)
    num_pid_n = tl.cdiv(N, 32)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_am = pid_m * 32 + tl.arange(0, 32)
    offs_bn = pid_n * 32 + tl.arange(0, 32)
    offs_k = tl.arange(0, 16)
    a_ptrs = a_ptr + offs_am[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_bn[None, :]
    acc = tl.zeros([32, 32], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, 16)):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M)
        b = tl.load(b_ptrs, mask=offs_bn[None, :] < N)
        acc = tl.dot(a, b, acc)
        a_ptrs += 16
        b_ptrs += 16 * N
    c = acc.to(tl.float16)
    offs_cm = pid_m * 32 + tl.arange(0, 32)
    offs_cn = pid_n * 32 + tl.arange(0, 32)
    c_ptrs = c_ptr + offs_cm[:, None] * N + offs_cn[None, :]
    tl.store(c_ptrs, c, mask=offs_cm[:, None] < M)

compile_kernel("matmul", kernel_matmul,
               {'a_ptr': '*fp16', 'b_ptr': '*fp16', 'c_ptr': '*fp16',
                'M': 'i32', 'N': 'i32', 'K': 'i32'})


# ========== 3. 归约 (reduction) ==========
@triton.jit
def kernel_reduce(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    out = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, out)

compile_kernel("reduce", kernel_reduce,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 4. 分组归约 (grouped reduction with axis) ==========
@triton.jit
def kernel_grouped_reduce(x_ptr, out_ptr, ROW: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * ROW * 64 + tl.arange(0, 64)
    x = tl.load(x_ptr + off)
    out = tl.sum(x, axis=0)
    tl.store(out_ptr + pid * 64 // ROW, out)

compile_kernel("grouped_reduce", kernel_grouped_reduce,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32', 'ROW': 'i32'})


# ========== 5. 合并加 (fused multiply-add) ==========
@triton.jit
def kernel_fma(x_ptr, y_ptr, z_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off, mask=off < N)
    y = tl.load(y_ptr + off, mask=off < N)
    z = tl.load(z_ptr + off, mask=off < N)
    out = x * y + z
    tl.store(out_ptr + off, out, mask=off < N)

compile_kernel("fma", kernel_fma,
               {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'z_ptr': '*fp32',
                'out_ptr': '*fp32', 'N': 'i32'})


# ========== 6. 条件运算 (select/if) ==========
@triton.jit
def kernel_cond(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off, mask=off < N)
    y = tl.load(y_ptr + off, mask=off < N)
    # 条件运算
    result = tl.where(x > y, x, y)
    tl.store(out_ptr + off, result, mask=off < N)

compile_kernel("cond_select", kernel_cond,
               {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32', 'N': 'i32'})


# ========== 7. 类型转换 ==========
@triton.jit
def kernel_cast(x_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off, mask=off < N)
    # 多种类型转换
    x_f16 = x.to(tl.float16)
    x_i32 = x.to(tl.int32)
    result = x_f16 + x_i32
    tl.store(out_ptr + off, result, mask=off < N)

compile_kernel("cast", kernel_cast,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32', 'N': 'i32'})


# ========== 8. 广播操作 ==========
@triton.jit
def kernel_broadcast(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off, mask=off < N)
    # 标量广播
    scalar = tl.load(y_ptr + pid)
    result = x * scalar[:, None]  # 广播乘法
    tl.store(out_ptr + off, result, mask=off < N)

compile_kernel("broadcast", kernel_broadcast,
               {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32', 'N': 'i32'})


# ========== 9. 原子操作 ==========
@triton.jit
def kernel_atomic(x_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off, mask=off < N)
    # 跨线程求和 → 原子加
    total = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, total)

compile_kernel("atomic_add", kernel_atomic,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32', 'N': 'i32'})


# ========== 10. Scan (前缀和) ==========
@triton.jit
def kernel_scan(x_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off, mask=off < N)
    # 前缀和
    result = tl.cumsum(x, axis=0)
    tl.store(out_ptr + off, result, mask=off < N)

compile_kernel("scan", kernel_scan,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32', 'N': 'i32'})


# ========== 11. 空操作内核 (minimal) ==========
@triton.jit
def kernel_noop(x_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off, mask=off < N)
    tl.store(out_ptr + off, x, mask=off < N)

compile_kernel("noop", kernel_noop,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32', 'N': 'i32'})


# ========== 12. 多层嵌套循环 ==========
@triton.jit
def kernel_nested_loop(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    acc = tl.zeros([128], dtype=tl.float32)
    for i in range(16):
        x = tl.load(x_ptr + off + i * 128)
        acc = acc + x
    tl.store(out_ptr + off, acc)

compile_kernel("nested_loop", kernel_nested_loop,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 13. While 循环 ==========
@triton.jit
def kernel_while(x_ptr, out_ptr, MAX: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    i = 0
    while i < MAX:
        x = x + 1.0
        i = i + 1
    tl.store(out_ptr + off, x)

compile_kernel("while_loop", kernel_while,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32', 'MAX': 'i32'})


# ========== 14. 减法 ==========
@triton.jit
def kernel_sub(x_ptr, y_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    y = tl.load(y_ptr + off)
    z = x - y
    tl.store(out_ptr + off, z)

compile_kernel("sub", kernel_sub,
               {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 15. 真除法 + 等于/不等于 ==========
@triton.jit
def kernel_div_cmp(x_ptr, y_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    y = tl.load(y_ptr + off)
    div = x / y
    eq = x == y
    ne = x != y
    tl.store(out_ptr + off, div)

compile_kernel("div_cmp", kernel_div_cmp,
               {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 16. 按位或/异或/取反/移位 ==========
@triton.jit
def kernel_bitwise(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    a = x.to(tl.int32)
    o = a | 1
    xo = a ^ 1
    l = a << 2
    r = a >> 1
    n = ~a
    tl.store(out_ptr + off, n)

compile_kernel("bitwise", kernel_bitwise,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 17. tl.min / tl.max 归约 ==========
@triton.jit
def kernel_minmax(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    minv = tl.min(x, axis=0)
    maxv = tl.max(x, axis=0)
    tl.store(out_ptr + pid * 2 + 0, minv)
    tl.store(out_ptr + pid * 2 + 1, maxv)

compile_kernel("minmax", kernel_minmax,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 18. tl.sort ==========
@triton.jit
def kernel_sort(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    s = tl.sort(x, dim=0)
    tl.store(out_ptr + off, s)

compile_kernel("sort", kernel_sort,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 19. tl.trans (转置) ==========
@triton.jit
def kernel_trans(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = tl.arange(0, 64)
    x = tl.load(x_ptr + off)
    r = x.reshape([8, 8])
    t = tl.trans(r)
    s = tl.sum(t, axis=1)
    tl.store(out_ptr + pid * 8 + tl.arange(0, 8), s)

compile_kernel("trans", kernel_trans,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 20. tl.reshape ==========
@triton.jit
def kernel_reshape(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = tl.arange(0, 64)
    x = tl.load(x_ptr + off)
    r = x.reshape([8, 8])
    s = tl.sum(r, axis=1)
    tl.store(out_ptr + pid * 8 + tl.arange(0, 8), s)

compile_kernel("reshape", kernel_reshape,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 21. tl.exp / tl.sqrt / tl.abs ==========
@triton.jit
def kernel_math(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    e = tl.exp(x)
    s = tl.sqrt(x)
    a = tl.abs(x)
    tl.store(out_ptr + off, e + s + a)

compile_kernel("math", kernel_math,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 22. tl.minimum / tl.maximum ==========
@triton.jit
def kernel_minmax_elem(x_ptr, y_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    y = tl.load(y_ptr + off)
    a = tl.minimum(x, y)
    b = tl.maximum(x, y)
    tl.store(out_ptr + off, a + b)

compile_kernel("minmax_elem", kernel_minmax_elem,
               {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 23. atomic_max / atomic_xchg ==========
@triton.jit
def kernel_atomic_more(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    v = tl.sum(x, axis=0)
    tl.atomic_max(out_ptr, v)
    tl.atomic_xchg(out_ptr + 1, v)

compile_kernel("atomic_more", kernel_atomic_more,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 24. cat / join / split ==========
@triton.jit
def kernel_cat_join(x_ptr, y_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 64 + tl.arange(0, 64)
    x = tl.load(x_ptr + off)
    y = tl.load(y_ptr + off)
    c = tl.cat(x, y, can_reorder=True)
    tl.store(out_ptr + pid * 128 + tl.arange(0, 128), c)

compile_kernel("cat_join", kernel_cat_join,
               {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 25. flip / ravel ==========
@triton.jit
def kernel_flip_ravel(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    r = x.reshape([8, 16])
    f = tl.flip(r, dim=1)
    v = tl.ravel(f)
    tl.store(out_ptr + pid * 128 + tl.arange(0, 128), v)

compile_kernel("flip_ravel", kernel_flip_ravel,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 26. argmin / argmax ==========
@triton.jit
def kernel_argminmax(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    ai = tl.argmin(x, axis=0)
    aj = tl.argmax(x, axis=0)
    tl.store(out_ptr + pid * 2 + 0, ai)
    tl.store(out_ptr + pid * 2 + 1, aj)

compile_kernel("argminmax", kernel_argminmax,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 27. topk ==========
@triton.jit
def kernel_topk(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    v = tl.topk(x, k=8)
    tl.store(out_ptr + pid * 8 + tl.arange(0, 8), v)

compile_kernel("topk", kernel_topk,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 28. cumprod ==========
@triton.jit
def kernel_cumprod(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    r = tl.cumprod(x, axis=0)
    tl.store(out_ptr + off, r)

compile_kernel("cumprod", kernel_cumprod,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 29. atomic_and / atomic_or / atomic_cas ==========
@triton.jit
def kernel_atomic_bitwise(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    v = tl.sum(x, axis=0).to(tl.int32)
    iv = v.to(tl.int32)
    tl.atomic_and(out_ptr, iv)
    tl.atomic_or(out_ptr + 1, iv)

compile_kernel("atomic_bitwise", kernel_atomic_bitwise,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 30. math: log / sin / cos ==========
@triton.jit
def kernel_math_more(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    l = tl.log(x)
    s = tl.sin(x)
    c = tl.cos(x)
    tl.store(out_ptr + off, l + s + c)

compile_kernel("math_more", kernel_math_more,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 31. rsqrt ==========
@triton.jit
def kernel_rsqrt(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    r = tl.rsqrt(x)
    tl.store(out_ptr + off, r)

compile_kernel("rsqrt", kernel_rsqrt,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 32. 随机数 tl.rand ==========
@triton.jit
def kernel_rand(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    r = tl.rand(pid, off)
    tl.store(out_ptr + off, r)

compile_kernel("rand", kernel_rand,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 33. associative_scan ==========
@triton.jit
def _combine_add(a, b):
    return a + b

@triton.jit
def kernel_assoc_scan(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    r = tl.associative_scan(x, axis=0, combine_fn=_combine_add)
    tl.store(out_ptr + off, r)

compile_kernel("assoc_scan", kernel_assoc_scan,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 34. histogram ==========
@triton.jit
def kernel_histogram(x_ptr, out_ptr):
    pid = tl.program_id(0)
    off = pid * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + off)
    x_int = x.to(tl.int32)
    h = tl.histogram(x_int, num_bins=16)
    tl.store(out_ptr + pid * 16 + tl.arange(0, 16), h.to(tl.float32))

compile_kernel("histogram", kernel_histogram,
               {'x_ptr': '*fp32', 'out_ptr': '*fp32'})


# ========== 汇总 ==========
print(f"\n{'='*60}")
print(f"Summary: {sum(1 for _, s, _ in kernel_results if s == 'OK')}/{len(kernel_results)} OK")
for name, status, err in kernel_results:
    if status == "OK":
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}: {err}")