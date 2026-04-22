import os
import sys

# 清理缓存
import shutil
cache_dir = os.path.expanduser('~/.triton/cache')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"已清理缓存目录: {cache_dir}")

# ====================== 关键修复：环境变量 ======================
# 强制使用 CPU 后端，禁用 GPU 驱动
os.environ.update({
    'TRITON_KERNEL_DUMP': '1',
    'TRITON_ALWAYS_COMPILE': '1',
    'MLIR_ENABLE_DUMP': '1',
    'TRITON_DEBUG': '1',
    'TRITON_PRINT_IR': '1',
    'MLIR_DUMP_PATH': '/tmp/triton_dump',
    'LLVM_IR_ENABLE_DUMP': '1',
    # 核心：强制使用 CPU 编译模式
    'TRITON_BACKEND': 'cpu',
    # 关闭自动探测驱动
    'TRITON_NO_AUTOMATIC_DRIVER': '1',
})

# 确保 dump 目录存在
os.makedirs('/tmp/triton_dump', exist_ok=True)

import torch
import triton
import triton.language as tl

print(f"Triton 版本: {triton.__version__}")

# ====================== 简单加法 Kernel ======================
@triton.jit
def add_kernel(x, y, out, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x_val = tl.load(x + offs, mask=mask)
    y_val = tl.load(y + offs, mask=mask)
    z_val = x_val + y_val
    tl.store(out + offs, z_val, mask=mask)

def test():
    n = 1024
    # CPU 张量
    x = torch.randn(n, device='cpu')
    y = torch.randn(n, device='cpu')
    out = torch.empty_like(x)

    print("开始编译 Triton Kernel (CPU模式，无GPU依赖)...")

    # ====================== 关键修复：只编译，不启动驱动 ======================
    try:
        # 方法1：使用 warmup 纯编译
        add_kernel.warmup(
            x, y, out, n,
            BLOCK_SIZE=128,
            grid=(n//128 + 1,),
            device_type='cpu'  # 强制指定CPU
        )
        print("✅ Kernel 编译完成！")

    except Exception as e:
        print(f"⚠️ 编译过程输出 (用于调试):\n{e}")

    # 查看生成的 IR 文件
    dump_dir = '/tmp/triton_dump'
    if os.path.exists(dump_dir):
        files = os.listdir(dump_dir)
        if files:
            print(f"\n📂 生成的 Dump 文件 ({len(files)} 个):")
            for f in files[:10]:
                print(f"  - {f}")
        else:
            print("\n❌ Dump 目录为空，编译未生成 IR")

if __name__ == "__main__":
    test()
