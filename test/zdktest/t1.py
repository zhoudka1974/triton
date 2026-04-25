import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, n: tl.constexpr):
    pid=tl.program_id(0)
    if pid < n:
        x=tl.load(x_ptr + pid)
        tl.store(y_ptr + pid, x + 1)

import numpy as np
x=np.array([1, 2, 3], dtype=np.float32)
y=np.zeros_like(x)
add_kernel[(1,)](x, y, 3)
print(y)
