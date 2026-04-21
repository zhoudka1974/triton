import os
import sys

# 清理缓存
import shutil
cache_dir = os.path.expanduser('~/.triton/cache')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"已清理缓存目录: {cache_dir}")

# 设置环境变量 - 关键：不要设置 TRITON_INTERPRET
os.environ.update({
    'TRITON_KERNEL_DUMP': '1',
    'TRITON_ALWAYS_COMPILE': '1',
    'MLIR_ENABLE_DUMP': '1',
    'MLIR_DEBUG_PASS': '1',
    'TRITON_DEBUG': '1',
    'TRITON_PRINT_IR': '1',
    'MLIR_DUMP_PATH': '/tmp/triton_dump',  # 指定 dump 路径
    'LLVM_IR_ENABLE_DUMP': '1',  # 也启用 LLVM IR dump
    'TRITON_USE_CPU': '1', # 使用 CPU 而不是 GPU
})

# 确保 dump 目录存在
os.makedirs('/tmp/triton_dump', exist_ok=True)

import torch
import triton
import triton.language as tl

print(f"Triton 版本: {triton.__version__}")

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
    x = torch.randn(n, device='cpu')
    y = torch.randn(n, device='cpu')
    out = torch.empty_like(x)
    
    print("运行 kernel...")
    
    # 使用 launch=False 来只编译不运行
    # 这样可以避免驱动问题
    compiled_kernel = add_kernel[(n,)](x, y, out, n, BLOCK_SIZE=128, launch=False)
    
    # 或者使用 warmup=True 来触发编译
    add_kernel.warmup(torch.float32, torch.float32, torch.float32, n, BLOCK_SIZE=128, grid=(n,))
    
    print("编译完成，检查 dump 目录...")
    
    # 检查 dump 目录
    dump_dir = '/tmp/triton_dump'
    if os.path.exists(dump_dir):
        files = os.listdir(dump_dir)
        if files:
            print(f"Dump 目录中的文件: {files[:10]}")  # 显示前10个文件
            # 查看第一个文件的内容
            if files:
                with open(os.path.join(dump_dir, files[0]), 'r') as f:
                    print(f"\n第一个文件的前1000字符:\n{f.read(1000)}")
        else:
            print("Dump 目录为空")
    
    print(out)

if __name__ == "__main__":
    test()