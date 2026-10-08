"""Opt-in DSv4 SFT batch validation and first-nonfinite capture.

Enable with --custom-megatron-init-path miles.utils.finite_capture.install.
Optionally add --custom-rollout-log-function-path
miles.utils.finite_capture.log_rollout_data to persist sample/pack identity
without replacing the selected rollout function or the default metrics logger.
Requires --qkv-format thd --allgather-cp (the contiguous global-stream layout).
Supports both single-conversation and packed SFT samples. Adds synchronous value
checks; do not use for throughput measurement. Operator tensor payloads are
bounded at 4 GiB for the first reporting rank in a job, plus its CPU batch.
Other ranks preserve metadata. Uses the crash capture module's output directory
and the same DSv4 Megatron/Transformer Engine API requirements.
"""

import functools
import inspect
import os
import re
import sys
import time
from pathlib import Path

import torch

from miles.utils import crash_capture

_CHECK_ELEMENTS = 16 * 1024**2
_PAYLOAD_BYTES = 4 * 1024**3
_BATCH_KEYS = (
    "tokens",
    "unconcat_tokens",
    "loss_masks",
    "input_loss_masks",
    "total_lengths",
    "response_lengths",
    "subseq_lens",
    "cu_seqlens",
    "sample_indices",
)


class DiagnosticFailure(RuntimeError):
    """A validation failure with evidence already persisted."""


def _job_id():
    match = re.search(r"ray-(\d+)/", " ".join(sys.argv))
    return os.environ.get("SLURM_JOB_ID") or (match.group(1) if match else f"unknown-{os.getpid()}")


def _root(args):
    return Path(args.save).parent / "diagnostics" / f"job-{_job_id()}"


def _tensors(value, prefix="value"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from _tensors(child, f"{prefix}[{i}]")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _tensors(child, f"{prefix}.{key}")


def _chunks(tensor):
    if tensor.numel() <= _CHECK_ELEMENTS:
        yield tensor
        return
    dimension = max(range(tensor.ndim), key=lambda dim: tensor.shape[dim])
    step = max(1, _CHECK_ELEMENTS // (tensor.numel() // tensor.shape[dimension]))
    for part in tensor.split(step, dim=dimension):
        yield from _chunks(part)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format)
    if isinstance(value, (list, tuple)):
        return [_cpu(x) for x in value]
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    return value


def _batch_errors(batch, *, vocab_size, cp_size, cp_rank):
    errors = []
    tokens = batch["unconcat_tokens"]
    full_masks = []
    boundaries = [0]
    for i, (ids, mask, total, response) in enumerate(
        zip(tokens, batch["loss_masks"], batch["total_lengths"], batch["response_lengths"], strict=True)
    ):
        if ids.ndim != 1 or ids.numel() != total or total < 2:
            errors.append(f"sample {i}: token length/shape mismatch")
        if ids.dtype not in (torch.int32, torch.int64) or bool(((ids < 0) | (ids >= vocab_size)).any()):
            errors.append(f"sample {i}: invalid token ID/dtype")
        if not (0 <= response < total) or mask.numel() != response:
            errors.append(f"sample {i}: invalid response or mask length")
            continue
        if not bool(((mask == 0) | (mask == 1)).all()):
            errors.append(f"sample {i}: mask is not binary/finite")
        lens = (batch.get("subseq_lens") or [None] * len(tokens))[i] or [total]
        if sum(lens) != total or any(length <= 0 for length in lens):
            errors.append(f"sample {i}: invalid conversation lengths")
            continue
        expanded = torch.zeros(total, dtype=mask.dtype)
        if response:
            expanded[-response:] = mask
        local = 0
        for length in lens:
            if expanded[local] != 0:
                errors.append(f"sample {i}: supervised first token at conversation offset {local}")
            local += length
            boundaries.append(boundaries[-1] + length)
        full_masks.append(expanded)
    if errors:
        return errors
    original = torch.cat(tokens)
    full_mask = torch.cat(full_masks)
    local_tokens = batch["tokens"].flatten()
    total_padded = local_tokens.numel() * cp_size
    pad = total_padded - original.numel()
    if pad < 0:
        return ["CP token allocation shorter than original stream"]
    if pad:
        original = torch.cat((original, torch.zeros(pad, dtype=original.dtype)))
        full_mask = torch.cat((full_mask, torch.zeros(pad, dtype=full_mask.dtype)))
        boundaries.append(total_padded)
    expected = original.chunk(cp_size)[cp_rank]
    expected_mask = full_mask.chunk(cp_size)[cp_rank]
    if not torch.equal(local_tokens, expected):
        errors.append("CP slice/padding tokens differ from the global stream")
    if not torch.equal(batch["input_loss_masks"].flatten(), expected_mask):
        errors.append("CP input mask differs from the global response mask")
    if batch["cu_seqlens"].tolist() != boundaries:
        errors.append("cu_seqlens differs from conversation boundaries plus padding")
    return errors


class _FiniteCapture:
    def __init__(self, capture, *, vocab_size, cp_size, cp_rank):
        self.capture = capture
        self.vocab_size = vocab_size
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.context = {}
        self.microbatch = 0
        self.handles = []
        self.installed = False
        self.layer_stack = []

    def fail(self, label, reason, payload, *, context=None):
        context = self.context if context is None else context
        directory = self.capture.directory
        evidence = {
            "label": label,
            "reason": reason,
            "step": self.capture.step,
            "context": {k: v for k, v in context.items() if k != "batch"},
            "layer_stack": list(self.layer_stack),
            "unix_time": time.time(),
            "payload_metadata": crash_capture._safe_describe(payload),
        }
        crash_capture._write_json(directory / "first-nonfinite.json", evidence)
        claim = directory.parent / "first-value-capture"
        try:
            claim.mkdir()
            winner = True
        except FileExistsError:
            winner = False
        if winner:
            try:
                self._save_payload(claim, evidence, payload, context)
            except Exception as exc:
                crash_capture._write_json(claim / "save-error.json", {"error": str(exc)})
            finally:
                (claim / "complete").write_text(str(directory))
        else:
            # Give the winning rank time to copy evidence before Ray kills peers.
            deadline = time.monotonic() + 90
            while not (claim / "complete").exists() and time.monotonic() < deadline:
                time.sleep(0.25)
        raise DiagnosticFailure(f"{label}: {reason}; evidence={directory}")

    def _save_payload(self, claim, evidence, payload, context):
        crash_capture._write_json(claim / "failure.json", evidence)
        batch = context.get("batch")
        if batch is not None:
            torch.save(batch, claim / "batch.pt")
        budget = _PAYLOAD_BYTES
        manifest = []
        saved = {}
        for index, (name, tensor) in enumerate(_tensors(payload)):
            size = tensor.numel() * tensor.element_size()
            entry = {"name": name, "bytes": size, "metadata": crash_capture._safe_describe(tensor)}
            identity = (
                str(tensor.device),
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
            )
            if identity in saved:
                entry["file"] = saved[identity]
                entry["alias"] = True
            elif size <= budget:
                cpu = _cpu(tensor)
                filename = f"tensor-{index:04d}.pt"
                torch.save(cpu, claim / filename)
                del cpu
                entry["file"] = filename
                saved[identity] = filename
                budget -= size
            else:
                entry["omitted"] = "4 GiB payload budget"
            manifest.append(entry)
            crash_capture._write_json(claim / "tensors.json", manifest)

    def check(self, label, value, *, payload=None, context=None, allow_negative_inf=False):
        with torch.no_grad():
            for name, tensor in _tensors(value):
                if not (tensor.is_floating_point() or tensor.is_complex()):
                    continue
                for part in _chunks(tensor.detach()):
                    valid = torch.isfinite(part)
                    if allow_negative_inf:
                        valid |= torch.isneginf(part)
                    if not bool(valid.all()):
                        self.fail(
                            label, f"nonfinite {name}", {"bad_tensor": tensor, "operator": payload}, context=context
                        )

    def sparse_indices(self, args, kwargs):
        q, kv, _, _, _, _, indices = args[:7]
        lengths = kwargs.get("topk_length")
        if indices.ndim != 2 or indices.shape[0] != q.shape[0] or indices.dtype != torch.int32:
            self.fail("sparse_indices", "invalid index shape/dtype", {"args": args, "kwargs": kwargs})
        width = indices.shape[1]
        if lengths is not None:
            if lengths.shape != (q.shape[0],) or lengths.dtype != torch.int32:
                self.fail("sparse_indices", "invalid length shape/dtype", {"args": args, "kwargs": kwargs})
            if bool(((lengths < 0) | (lengths > width)).any()):
                self.fail("sparse_indices", "invalid valid-prefix length", {"args": args, "kwargs": kwargs})
        columns = torch.arange(width, device=indices.device)
        for start in range(0, indices.shape[0], 256):
            block = indices[start : start + 256]
            active = columns[None, :] < lengths[start : start + 256, None] if lengths is not None else block >= 0
            if bool((active & ((block < 0) | (block >= kv.shape[0]))).any()):
                self.fail("sparse_indices", "active index out of KV range", {"args": args, "kwargs": kwargs})

    def wrap_sparse(self, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            payload = {"args": args, "kwargs": kwargs}
            self.sparse_indices(args, kwargs)
            self.check("sparse_backward.input", args[:4] + args[5:7], payload=payload)
            # -inf LSE is permitted for fully masked rows; NaN/+inf are not.
            self.check("sparse_backward.lse", args[4], payload=payload, allow_negative_inf=True)
            result = function(*args, **kwargs)
            self.check("sparse_backward.output", result, payload={**payload, "result": result})
            return result

        return wrapped

    def wrap_attention_forward(self, function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs).arguments
            payload = {"args": args, "kwargs": kwargs}
            self.check("sparse_forward.input", (bound["q"], bound["kv"], bound.get("attn_sink")), payload=payload)
            self.sparse_indices(
                (bound["q"], bound["kv"], None, None, None, bound.get("attn_sink"), bound["topk_idxs"]),
                {"topk_length": bound.get("topk_length")},
            )
            result = function(*args, **kwargs)
            # LSE may use infinity sentinels on fully masked rows; inspect the
            # actual attention output here, then validate the backward inputs.
            self.check("sparse_forward.output", result[0], payload={**payload, "result": result})
            return result

        return wrapped

    def wrap_gemm(self, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            # args[2] is a destination buffer and may be uninitialized before GEMM.
            inputs = args[:2] if args else (kwargs.get("A"), kwargs.get("B"))
            self.check("grouped_gemm.input", inputs, payload={"args": args, "kwargs": kwargs})
            result = function(*args, **kwargs)
            output = args[2] if len(args) > 2 else kwargs.get("out")
            self.check(
                "grouped_gemm.output", (output, result), payload={"args": args, "kwargs": kwargs, "result": result}
            )
            return result

        return wrapped

    def wrap_batch(self, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            batch = function(*args, **kwargs)
            cpu = _cpu({key: batch[key] for key in _BATCH_KEYS if key in batch})
            self.context = {
                "step": dict(self.capture.step),
                "microbatch": self.microbatch,
                "sample_indices": crash_capture._safe_describe(cpu.get("sample_indices")),
                "batch": cpu,
            }
            self.microbatch += 1
            errors = _batch_errors(cpu, vocab_size=self.vocab_size, cp_size=self.cp_size, cp_rank=self.cp_rank)
            if errors:
                self.fail("batch_validation", "; ".join(errors), cpu)
            crash_capture._write_json(
                self.capture.directory / "latest-batch.json",
                {
                    **{k: v for k, v in self.context.items() if k != "batch"},
                    "sample_indices": cpu.get("sample_indices"),
                    "total_lengths": cpu["total_lengths"],
                    "response_lengths": cpu["response_lengths"],
                    "supervised_tokens": [int(mask.sum()) for mask in cpu["loss_masks"]],
                    "cp_rank": self.cp_rank,
                    "cp_size": self.cp_size,
                },
            )
            return batch

        return wrapped

    def _gradient(self, label, context, gradient):
        if context is not None:
            self.context = context
        self.check(label, gradient, payload={"gradient": gradient}, context=context)
        return gradient

    def _parameter_gradient(self, name, parameter, gradient):
        # TE fused accumulation writes main_grad, then returns a dummy tensor to
        # trigger Megatron's DDP hook. Its storage can be uninitialized; DDP
        # discards it unless zero_out_wgrad requests an additional accumulation.
        fused = getattr(parameter, "grad_added_to_main_grad", False)
        if fused:
            self.check(f"parameter.{name}.main_grad", parameter.main_grad, payload={"main_grad": parameter.main_grad})
        if not fused or getattr(parameter, "zero_out_wgrad", False):
            self.check(f"parameter.{name}.gradient", gradient, payload={"gradient": gradient})
        return gradient

    def _pre_layer(self, name, module, inputs, kwargs):
        self.layer_stack.append(name)
        self.check(f"{name}.forward_input", (inputs, kwargs), payload={"inputs": inputs, "kwargs": kwargs})

    def _post_layer(self, name, module, inputs, kwargs, output):
        self.check(f"{name}.forward_output", output, payload={"inputs": inputs, "kwargs": kwargs, "output": output})
        context = self.context
        for label, tensor in _tensors(output):
            if tensor.requires_grad:
                tensor.register_hook(functools.partial(self._gradient, f"{name}.{label}.backward_input", context))
        for label, tensor in _tensors((inputs, kwargs)):
            if tensor.requires_grad:
                tensor.register_hook(functools.partial(self._gradient, f"{name}.{label}.backward_output", context))
        if self.layer_stack:
            self.layer_stack.pop()

    def attach_layers(self, models):
        if self.installed:
            return
        names = []
        for chunk, model in enumerate(models):
            for name, module in model.named_modules():
                kind = type(module).__name__
                if kind not in ("GPTModel", "TransformerLayer") and not kind.endswith("TransformerLayer"):
                    continue
                label = f"chunk{chunk}.{name}:{kind}:layer{getattr(module, 'layer_number', None)}"
                names.append(label)
                self.handles.append(
                    module.register_forward_pre_hook(functools.partial(self._pre_layer, label), with_kwargs=True)
                )
                self.handles.append(
                    module.register_forward_hook(functools.partial(self._post_layer, label), with_kwargs=True)
                )
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.handles.append(
                        parameter.register_hook(functools.partial(self._parameter_gradient, name, parameter))
                    )
        if not names:
            raise RuntimeError("No model/layer boundaries instrumented")
        crash_capture._write_json(
            self.capture.directory / "finite-ready.json", {"layers": names, "payload_budget": _PAYLOAD_BYTES}
        )
        self.installed = True

    def wrap_step(self, function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs).arguments
            self.microbatch = 0
            self.context = {}
            self.layer_stack.clear()
            self.attach_layers(bound["model"])
            return function(*args, **kwargs)

        return wrapped

    def wrap_loss(self, function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs).arguments
            self.check("loss.logits", bound["logits"], payload={"logits": bound["logits"]})
            result = function(*args, **kwargs)
            self.check("loss.output", result, payload={"logits": bound["logits"], "result": result})
            return result

        return wrapped


def install(args):
    """Install diagnostic checks before constructing the model."""
    if args.qkv_format != "thd" or not args.allgather_cp:
        raise ValueError("Finite capture requires --qkv-format thd --allgather-cp")

    # Training-only dependencies are loaded only by this opt-in initialization hook.
    from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention
    from transformer_engine.pytorch.cpp_extensions import gemm
    from transformer_engine.pytorch.module import grouped_linear
    from miles.backends.megatron_utils import model
    from miles.backends.training_utils.parallel import get_parallel_state

    capture = crash_capture.install(args)
    if capture is None:
        raise RuntimeError("Finite checks must be installed once through their own init hook")
    parallel = get_parallel_state()
    probe = _FiniteCapture(
        capture, vocab_size=args.padded_vocab_size, cp_size=parallel.cp.size, cp_rank=parallel.cp.rank
    )
    fused_sparse_attention._csa_fwd_flash_mla = probe.wrap_attention_forward(fused_sparse_attention._csa_fwd_flash_mla)
    namespace = fused_sparse_attention._DSA
    namespace.sparse_attention_backward_wrapper = probe.wrap_sparse(namespace.sparse_attention_backward_wrapper)
    wrapped_gemm = probe.wrap_gemm(gemm.general_grouped_gemm)
    gemm.general_grouped_gemm = wrapped_gemm
    grouped_linear.general_grouped_gemm = wrapped_gemm
    model.get_batch = probe.wrap_batch(model.get_batch)
    model.loss_function = probe.wrap_loss(model.loss_function)
    model.train_one_step = probe.wrap_step(model.train_one_step)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Persist sample identity through the existing custom rollout logging hook."""
    path = _root(args) / "batches"
    path.mkdir(parents=True, exist_ok=True)
    records = []
    for item in samples:
        sample = item[0] if isinstance(item, (list, tuple)) else item
        records.append(
            {
                "sample_index": sample.index,
                "pack": sample.prompt,
                "total_tokens": len(sample.tokens),
                "response_length": sample.response_length,
                "subseq_lens": (sample.metadata or {}).get("subseq_lens"),
            }
        )
    crash_capture._write_json(path / f"rollout-{rollout_id:06d}.json", records)
    return False
