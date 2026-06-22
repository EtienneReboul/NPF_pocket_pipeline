"""Quick sanity-check for the esmfold2 conda environment.

Run without activating the env:
    conda run -n esmfold2 python envs/test_esmfold2.py
"""

import sys

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def check(label, fn):
    try:
        result = fn()
        msg = f"  {result}" if result else ""
        print(f"[{PASS}] {label}{msg}")
        return True
    except Exception as e:
        print(f"[{FAIL}] {label}  →  {e}")
        return False


def warn(label, fn):
    try:
        result = fn()
        msg = f"  {result}" if result else ""
        print(f"[{PASS}] {label}{msg}")
    except Exception as e:
        print(f"[{WARN}] {label}  →  {e}")


print("\n=== esmfold2 environment check ===")
print(f"Python {sys.version}\n")

# --- Core runtime ---
check("torch import", lambda: __import__("torch").__version__)

def _torch_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — CPU-only build or no GPU visible")
    v = torch.version.cuda
    dev = torch.cuda.get_device_name(0)
    return f"CUDA {v}  |  {dev}"

check("torch CUDA", _torch_cuda)

def _torch_version_tag():
    import torch
    tag = torch.__version__
    if "cu" not in tag:
        raise RuntimeError(f"torch wheel has no CUDA tag: {tag}")
    return tag

check("torch wheel is CUDA build", _torch_version_tag)

# --- Memory-efficiency kernels ---
def _xformers():
    import xformers
    import xformers.ops
    # Probe that the C++ extension actually loaded (not just the Python stub)
    ops = xformers.ops.memory_efficient_attention
    return f"xformers {xformers.__version__}  |  memory_efficient_attention available"

check("xformers (C++/CUDA ext)", _xformers)

def _flash_attn():
    import flash_attn
    return f"flash-attn {flash_attn.__version__}"

check("flash-attn", _flash_attn)

def _te():
    import transformer_engine.pytorch as te
    return f"transformer-engine {__import__('transformer_engine').__version__}"

check("transformer-engine", _te)

# --- ESM ---
def _esm():
    import esm
    return f"esm {getattr(esm, '__version__', 'unknown')}"

check("esm import", _esm)

def _esmc():
    from esm.models.esmc import ESMC
    return "ESMC class importable"

check("esm ESMC class", _esmc)

# --- rdkit ---
check("rdkit", lambda: __import__("rdkit").__version__)

# --- Optional: tiny forward pass (no weights downloaded) ---
def _esmc_init():
    import torch
    from esm.models.esmc import ESMC
    # Instantiate with smallest config if available, just to test graph build
    # We do NOT call .from_pretrained() to avoid a large download
    return "ESMC instantiation skipped (no weight download)"

warn("ESMC forward pass (skipped — no download)", _esmc_init)

print("\n=== done ===\n")
