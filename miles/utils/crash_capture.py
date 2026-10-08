"""Opt-in, bounded GPU failure capture for Megatron DSv4 training.

Enable with --custom-megatron-init-path miles.utils.crash_capture.install.
Records allocation history without synchronizing training. On the first OOM or
training exception per rank, persists allocator state and Python traceback local
metadata, including sparse-attention compile keys and launch arguments when those
frames are present. Does not copy CUDA tensor contents or retain tensor references.
Writes to <save parent>/diagnostics/job-<Slurm ID or Ray job ID>/rank*.
Requires the DSv4 Megatron CSA/DSA and Transformer Engine grouped-GEMM APIs.
Native process death or an inaccessible filesystem can still prevent a dump.
"""

import functools
import inspect
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_MAX_HISTORY = 20000
_MAX_FRAMES = 48


def _describe(value, depth=0):
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "storage_offset": value.storage_offset(),
            "data_ptr": value.data_ptr(),
            "alignment_mod_16": value.data_ptr() % 16,
            "requires_grad": value.requires_grad,
            "contiguous": value.is_contiguous(),
            "numel": value.numel(),
        }
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:2048]
    if depth < 3 and isinstance(value, (tuple, list, deque)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [_describe(x, depth + 1) for x in list(value)[:32]],
        }
    if depth < 3 and isinstance(value, dict):
        return {str(k)[:128]: _describe(v, depth + 1) for k, v in list(value.items())[:48]}
    # Never repr arbitrary modules/models: their formatting may read tensor values.
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if kind.startswith(("cuda.", "cutlass.", "tvm_ffi.")):
        try:
            return {"type": kind, "repr": repr(value)[:2048]}
        except Exception:
            pass
    return {"type": kind}


def _safe_describe(value):
    try:
        return _describe(value)
    except Exception as exc:
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}


def _frames(tb):
    frames = []
    while tb is not None:
        frame = tb.tb_frame
        frames.append(
            {
                "file": frame.f_code.co_filename,
                "function": frame.f_code.co_name,
                "line": tb.tb_lineno,
                "locals": {
                    k: _safe_describe(v) for k, v in list(frame.f_locals.items())[:96] if not k.startswith("__")
                },
            }
        )
        tb = tb.tb_next
    return frames[-_MAX_FRAMES:]


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _memory_status():
    result = {}
    for name, getter in {
        "stats": torch.cuda.memory_stats,
        "free_total_bytes": torch.cuda.mem_get_info,
        "stream": lambda: torch.cuda.current_stream().cuda_stream,
    }.items():
        try:
            result[name] = getter()
        except Exception as exc:
            result[name + "_error"] = str(exc)
    return result


def _node_report(directory, phase):
    path = directory.parent / f"{socket.gethostname()}-{phase}.json"
    try:
        # One reporter per node/phase, even when many local ranks fail together.
        with path.with_suffix(".lock").open("x"):
            pass
    except FileExistsError:
        return
    report = {}
    for name, command in {
        "nvidia_smi": ["nvidia-smi", "-q", "-x"],
        "kernel_log": ["dmesg", "--since", "10 minutes ago"],
    }.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=8)
            report[name] = {
                "returncode": result.returncode,
                "stdout": result.stdout[-262144:],
                "stderr": result.stderr[-4096:],
            }
        except Exception as exc:
            report[name] = {"error": str(exc)}
    _write_json(path, report)


class _Capture:
    """Owns this worker's bounded history and one-shot crash records."""

    def __init__(self, root, rank):
        self.directory = Path(root) / f"rank{rank:05d}-{socket.gethostname()}-pid{os.getpid()}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.step = {}
        self.seen = set()
        self.recent_calls = deque(maxlen=8)

    def start(self):
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=_MAX_HISTORY,
        )
        torch._C._cuda_attach_out_of_memory_observer(self.on_oom)
        # Fail initialization if snapshot writing is unavailable; don't silently run blind.
        torch.cuda.memory._dump_snapshot(str(self.directory / "startup.pickle"))
        _write_json(
            self.directory / "ready.json",
            {
                "rank": self.rank,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "module": __file__,
                "max_history_entries": _MAX_HISTORY,
                "capture": ["allocator_snapshot", "exception_frame_locals", "recent_kernel_calls"],
            },
        )
        _node_report(self.directory, "startup")
        logger.info("Crash capture READY rank=%s path=%s", self.rank, self.directory)

    def on_oom(self, device, alloc, device_alloc, device_free):
        self.capture(
            "oom",
            extra={
                "device": device,
                "requested_bytes": alloc,
                "device_allocated": device_alloc,
                "device_free": device_free,
            },
        )

    def capture(self, kind, exc=None, extra=None):
        if kind in self.seen:
            return
        self.seen.add(kind)
        # Persist CPU metadata first; no tensor .cpu(), .item(), reductions or sync.
        try:
            payload = {
                "unix_time": time.time(),
                "rank": self.rank,
                "step": self.step,
                "kind": kind,
                "extra": extra,
                "recent_calls": list(self.recent_calls),
                "exception": None if exc is None else f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc() if exc is not None else traceback.format_stack(),
                "frames": _frames(exc.__traceback__) if exc is not None else [],
            }
            _write_json(self.directory / f"{kind}.json", payload)
        except Exception:
            logger.exception("Crash metadata writing failed")
        try:
            torch.cuda.memory._dump_snapshot(str(self.directory / f"{kind}.pickle"))
        except Exception:
            logger.exception("Crash allocator snapshot failed")
        # OOM callbacks may run while the allocator lock is held: never query memory
        # statistics or invoke allocator APIs beyond the supported snapshot operation.
        if kind != "oom":
            try:
                _write_json(self.directory / f"{kind}-memory.json", _memory_status())
                _node_report(self.directory, "failure")
            except Exception:
                logger.exception("Crash device status writing failed")
        logger.error("Crash capture persisted kind=%s rank=%s path=%s", kind, self.rank, self.directory)

    def wrap_step(self, function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs).arguments
            self.step = {key: bound.get(key) for key in ("rollout_id", "step_id", "attempt", "num_microbatches")}
            self.recent_calls.clear()
            try:
                _write_json(self.directory / "latest-step.json", {**self.step, "memory": _memory_status()})
            except Exception:
                logger.exception("Step diagnostic write failed")
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                self.capture("exception", exc=exc)
                raise

        return wrapped

    def wrap_kernel(self, function, label):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            self.recent_calls.append(
                {"kernel": label, "time": time.time(), "args": _safe_describe(args), "kwargs": _safe_describe(kwargs)}
            )
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                # The innermost boundary captures first, before autograd unwinds.
                self.capture("exception", exc=exc, extra={"kernel": label})
                raise

        return wrapped


def install(args):
    """Megatron initialization hook: instrument this worker before model loading."""
    if not args.save:
        raise ValueError("Crash capture requires --save to locate diagnostic artifacts")

    # Optional heavy imports only when the explicit diagnostic hook is selected.
    from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention
    from transformer_engine.pytorch.cpp_extensions import gemm
    from transformer_engine.pytorch.module import grouped_linear

    from miles.backends.megatron_utils import model

    if getattr(model.train_one_step, "_crash_capture_installed", False):
        return
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        # Ray retains the job-specific temp directory in its worker argv.
        match = re.search(r"ray-(\d+)/", " ".join(sys.argv))
        job = match.group(1) if match else f"unknown-{int(time.time())}"
    capture = _Capture(Path(args.save).parent / "diagnostics" / f"job-{job}", args.rank)
    capture.start()
    fused_sparse_attention._ensure_dsa_namespace()
    namespace = fused_sparse_attention._DSA
    namespace.sparse_attention_backward_wrapper = capture.wrap_kernel(
        namespace.sparse_attention_backward_wrapper, "cudnn.DSA.sparse_attention_backward_wrapper"
    )
    wrapped_gemm = capture.wrap_kernel(gemm.general_grouped_gemm, "transformer_engine.general_grouped_gemm")
    gemm.general_grouped_gemm = wrapped_gemm
    grouped_linear.general_grouped_gemm = wrapped_gemm
    model.train_one_step = capture.wrap_step(model.train_one_step)
    model.train_one_step._crash_capture_installed = True
    return capture
