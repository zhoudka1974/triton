import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, out, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x_val = tl.load(x + offs, mask=mask)
    y_val = tl.load(y + offs, mask=mask)
    z_val = x_val + y_val
    tl.store(out + offs, z_val, mask=mask)

n = 1024
x = torch.randn(n)
y = torch.randn(n)
out = torch.empty_like(x)
# 强制编译
print("before add" )
print(out);
add_kernel[(n,)](x, y, out, n, BLOCK_SIZE=128)
print("after add")
print(out)
