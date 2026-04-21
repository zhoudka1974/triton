import os
import sys

# 清理缓存
import shutil
cache_dir = os.path.expanduser('~/.triton/cache')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"已清理缓存目录: {cache_dir}")

# 设置环境变量
os.environ.update({
    'TRITON_KERNEL_DUMP': '1',
    'TRITON_ALWAYS_COMPILE': '1',
    'MLIR_ENABLE_DUMP': '1',
    'MLIR_DEBUG_PASS': '1',
    'TRITON_DEBUG': '1',
    'TRITON_PRINT_IR': '1',
    'TRITON_DUMP_IR': '1',
    'MLIR_DUMP_PATH': '/tmp/triton_dump',
    'LLVM_IR_ENABLE_DUMP': '1',
    'TRITON_DUMP_PTX': '1',
})

# 确保 dump 目录存在
os.makedirs('/tmp/triton_dump', exist_ok=True)

import torch
import triton
import triton.language as tl
from triton.compiler import compile

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
    
    print("编译 kernel...")
    
    # 方法1：使用 compile 函数直接编译
    try:
        # 获取 kernel 的源代码
        kernel_fn = add_kernel.fn
        
        # 使用 compile 编译
        compiled = compile(
            add_kernel,
            signature="*fp32,*fp32,*fp32,i32,i32",
            device=0,  # 即使没有 GPU，这里传入 0
            num_warps=4,
            num_stages=1,
        )
        print("编译成功")
    except Exception as e:
        print(f"编译失败: {e}")
        print("\n尝试其他方法...")
    
    # 方法2：尝试获取 IR
    print("\n尝试获取 Triton IR...")
    try:
        # 通过 ast 获取 IR
        from triton.compiler.compiler import ASTSource
        import inspect
        
        # 获取函数源码
        source = inspect.getsource(add_kernel.fn)
        print("Kernel 源代码:")
        print(source)
        
    except Exception as e:
        print(f"获取 IR 失败: {e}")
    
    # 检查 dump 目录
    dump_dir = '/tmp/triton_dump'
    if os.path.exists(dump_dir):
        files = os.listdir(dump_dir)
        if files:
            print(f"\nDump 目录中的文件: {files}")
            for file in files[:5]:  # 显示前5个文件
                filepath = os.path.join(dump_dir, file)
                print(f"\n文件: {file}")
                try:
                    with open(filepath, 'r') as f:
                        content = f.read(2000)  # 读取前2000字符
                        print(f"前2000字符:\n{content[:2000]}")
                except Exception as e:
                    print(f"读取文件失败: {e}")
        else:
            print("Dump 目录为空")
    
    # 检查 ~/.triton/dump/ 目录
    home_dump = os.path.expanduser('~/.triton/dump/')
    if os.path.exists(home_dump):
        files = os.listdir(home_dump)
        if files:
            print(f"\n~/.triton/dump/ 目录中的文件: {files[:5]}")

if __name__ == "__main__":
    test()