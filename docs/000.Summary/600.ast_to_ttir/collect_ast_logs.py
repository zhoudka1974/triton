#!/usr/bin/env python3
"""
扫描 python/test/ 下所有 @triton.jit 内核，调用 ast_to_ttir() 收集 AST→TTIR 对应关系日志。

策略: 不导入测试文件（依赖环境不完整），直接用 AST 解析 + exec 动态编译内核函数。

用法:
  cd /home/zhoudka/work/code/triton
  .venv/bin/python3 docs/000.Summary/600.ast_to_ttir/collect_ast_logs.py

输出:
  docs/000.Summary/600.ast_to_ttir/{path_safe_name}.log  每个文件的日志
  docs/000.Summary/600.ast_to_ttir/collect_summary.log  汇总报告
"""

import ast
import os
import sys
import subprocess
import json
import textwrap
from pathlib import Path

# 路径
TRITON_ROOT = Path(__file__).resolve().parents[3]  # → project root
TEST_DIR = TRITON_ROOT / "python" / "test"
OUTPUT_DIR = TRITON_ROOT / "docs" / "000.Summary" / "600.ast_to_ttir"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_PATH = TRITON_ROOT / "python"


def _has_triton_jit(node):
    """检查 AST 节点是否被 @triton.jit 装饰"""
    if not isinstance(node, ast.FunctionDef):
        return False
    for deco in node.decorator_list:
        # @triton.jit
        if (isinstance(deco, ast.Attribute) and deco.attr == "jit" and
            isinstance(deco.value, ast.Name) and deco.value.id == "triton"):
            return True
        # @jit (from triton import jit)
        if isinstance(deco, ast.Name) and deco.id == "jit":
            return True
    return False


def find_triton_jit_functions(filepath):
    """扫描文件，返回 (模块级 @triton.jit 函数名列表, 函数源码字典)"""
    source = open(filepath).read()
    tree = ast.parse(source)
    funcs = []
    func_sources = {}
    for item in tree.body:
        if isinstance(item, ast.FunctionDef) and _has_triton_jit(item):
            funcs.append(item.name)
            # 提取函数源码（包括 decorator）
            start_line = item.lineno - 1  # 0-based
            # 获取 decorator 所在行
            if item.decorator_list:
                start_line = item.decorator_list[0].lineno - 1
            end_line = item.end_lineno
            func_sources[item.name] = textwrap.dedent("\n".join(source.split("\n")[start_line:end_line]))
    return funcs, func_sources


SCRIPT_TEMPLATE = r"""# auto-generated: collect AST→TTIR mapping for {rel_path}
import sys
sys.path.insert(0, {python_path!r})
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['TRITON_AST_LOG_NAME'] = {log_name!r}

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction
from triton.compiler.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir

results = {{"file": {rel_path!r}, "total_kernels": {num_kernels}, "kernels": []}}

{kernel_code}

for kname, kernel_fn in KERNELS:
    try:
        jit_fn = triton.jit(kernel_fn)
        sig = {{}}
        for p in jit_fn.params:
            if p.is_constexpr:
                sig[p.name] = 'constexpr'
            else:
                sig[p.name] = '*fp32'
        constexprs = {{}}
        attrs = {{}}

        target = GPUTarget('cuda', 89, 32)
        backend = make_backend(target)
        options = backend.parse_options({{'arch': 'sm89'}})
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        codegen_fns = backend.get_codegen_implementation(options)
        module_map = backend.get_module_map()

        src = ASTSource(jit_fn, sig, constexprs, attrs)
        module = src.make_ir(target, options, codegen_fns, module_map, context)
        assert module.verify()
        results["kernels"].append({{"name": kname, "status": "OK"}})
    except Exception as e:
        import traceback
        results["kernels"].append({{"name": kname, "status": "FAIL", "error": str(e)[:200], "traceback": traceback.format_exc()[:300]}})

print("__RESULT__" + json.dumps(results))
"""


def process_file(py_file):
    """处理单个文件：AST 提取内核函数 → subprocess 动态编译→ ast_to_ttir()"""
    abspath = py_file.resolve()
    try:
        rel_path = str(abspath.relative_to(TRITON_ROOT))
    except ValueError:
        rel_path = str(abspath)
    log_name = rel_path.replace("/", "_").replace(".py", "")

    # 用 AST 提取 @triton.jit 函数的源码
    func_names, func_sources = find_triton_jit_functions(str(abspath))
    if not func_names:
        return {"file": rel_path, "total_kernels": 0, "kernels": [], "error": "No module-level @triton.jit found"}

    # 构建内核代码块：去除 @triton.jit decorator（之后手动调用 triton.jit()）
    kernel_lines = []
    kernel_lines.append("KERNELS = []")
    for idx, name in enumerate(func_names):
        src = func_sources[name]
        # 跳过 decorator 行只保留函数定义
        clean_src = "\n".join(l for l in src.split("\n")
                              if not l.strip().startswith("@triton.jit") and
                                 not l.strip().startswith("@jit"))
        kernel_lines.append(f"\n# Kernel {idx}: {name}")
        kernel_lines.append(clean_src)
        kernel_lines.append(f"KERNELS.append(('{name}', {name}))")

    kernel_code = "\n".join(kernel_lines)

    script = SCRIPT_TEMPLATE.format(
        rel_path=rel_path,
        python_path=str(PYTHON_PATH.resolve()),
        log_name=log_name,
        num_kernels=len(func_names),
        kernel_code=kernel_code,
    )

    # 写入临时文件运行（@triton.jit 要求函数定义在文件中）
    tmp = OUTPUT_DIR / f"._{log_name}.py"
    tmp.write_text(script)

    result = subprocess.run(
        [sys.executable, str(tmp)],
        capture_output=True, text=True, timeout=120,
        cwd=str(TRITON_ROOT),
    )

    # 解析结果
    for line in result.stdout.split("\n"):
        if line.startswith("__RESULT__"):
            parsed = json.loads(line[len("__RESULT__"):])
            # 清理临时文件
            try: tmp.unlink()
            except: pass
            return parsed

    # 解析失败
    try: tmp.unlink()
    except: pass
    stderr_line = (result.stderr or "")[:500]
    return {
        "file": rel_path,
        "total_kernels": len(func_names),
        "kernels": [],
        "error": stderr_line or "No result marker",
        "stderr": stderr_line,
        "stdout": result.stdout[:300],
    }


def main():
    print(f"Scanning {TEST_DIR} for @triton.jit files...")
    # 找到所有含模块级 @triton.jit 的文件
    jit_files = []
    for py_file in sorted(TEST_DIR.rglob("*.py")):
        if py_file.name in ("__init__.py", "conftest.py"):
            continue
        funcs, _ = find_triton_jit_functions(str(py_file))
        if funcs:
            jit_files.append((py_file, funcs))
    print(f"Found {len(jit_files)} files with module-level @triton.jit")
    total_kernels = sum(len(f) for _, f in jit_files)
    print(f"Total kernels: {total_kernels}")

    all_results = []
    ok_count = 0
    fail_count = 0

    for py_file, func_names in jit_files:
        try:
            rel = str(py_file.resolve().relative_to(TRITON_ROOT))
        except ValueError:
            rel = str(py_file)
        print(f"  {rel} ({len(func_names)} kernels)...", end=" ", flush=True)
        try:
            result = process_file(py_file)
            all_results.append(result)
            ok = sum(1 for k in result.get("kernels", []) if k["status"] == "OK")
            fl = sum(1 for k in result.get("kernels", []) if k["status"] == "FAIL")
            ok_count += ok
            fail_count += fl
            if ok + fl > 0:
                print(f"{ok} OK, {fl} FAIL")
            else:
                e = result.get("error", "")
                print(f"0 kernels ({e[:60]})")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            all_results.append({"file": rel, "kernels": [], "error": "TIMEOUT"})
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({"file": rel, "kernels": [], "error": str(e)[:100]})
        try:
            result = process_file(py_file)
            all_results.append(result)
            ok = sum(1 for k in result.get("kernels", []) if k["status"] == "OK")
            fl = sum(1 for k in result.get("kernels", []) if k["status"] == "FAIL")
            ok_count += ok
            fail_count += fl
            has_any_kernel = len(result.get("kernels", [])) > 0
            if has_any_kernel:
                print(f"{ok} OK, {fl} FAIL")
            else:
                err = result.get("error", "")
                print(f"0 kernels ({err[:60]})")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            all_results.append({"file": rel, "total_kernels": 0, "error": "TIMEOUT"})
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({"file": rel, "total_kernels": 0, "error": str(e)[:100]})

    # 写入汇总
    summary_path = OUTPUT_DIR / "collect_summary.log"
    with open(summary_path, "w") as f:
        f.write(f"# AST→TTIR Collection Summary\n")
        f.write(f"# Total files with @triton.jit: {len(jit_files)}\n")
        f.write(f"# Total kernels OK: {ok_count}\n")
        f.write(f"# Total kernels FAIL: {fail_count}\n\n")
        f.write("| File | Kernels | OK | FAIL | Detail |\n")
        f.write("|------|---------|----|------|--------|\n")
        for r in all_results:
            ks = r.get("kernels", [])
            ok = sum(1 for k in ks if k["status"] == "OK")
            fl = sum(1 for k in ks if k["status"] == "FAIL")
            names = ", ".join(k["name"] for k in ks if k["status"] == "OK")
            f.write(f"| {r['file']} | {len(ks)} | {ok} | {fl} | {names[:80]} |\n")

    print(f"\nDone. Summary: {ok_count} kernels OK, {fail_count} FAIL")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()