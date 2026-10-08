import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from miles.utils import crash_capture, finite_capture


def _probe(tmp_path):
    capture = crash_capture._Capture(tmp_path / "job", rank=0)
    return finite_capture._FiniteCapture(capture, vocab_size=128, cp_size=1, cp_rank=0)


def _batch(cp_rank=0):
    # Two conversations in one pack, one conversation in another, then padding.
    tokens = [torch.tensor([10, 11, 12, 13]), torch.tensor([20, 21, 22])]
    stream = torch.tensor([10, 11, 12, 13, 20, 21, 22, 0])
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0])
    return {
        "unconcat_tokens": tokens,
        "tokens": stream.chunk(2)[cp_rank].unsqueeze(0),
        "loss_masks": [torch.tensor([1, 0, 1]), torch.tensor([1, 1])],
        "input_loss_masks": mask.chunk(2)[cp_rank].unsqueeze(0),
        "total_lengths": [4, 3],
        "response_lengths": [3, 2],
        "subseq_lens": [[2, 2], [3]],
        "cu_seqlens": torch.tensor([0, 2, 4, 7, 8]),
    }


@pytest.mark.parametrize("rank", [0, 1])
def test_valid_packed_batch_with_padding_and_contiguous_cp(rank):
    assert finite_capture._batch_errors(_batch(rank), vocab_size=128, cp_size=2, cp_rank=rank) == []


@pytest.mark.parametrize(
    "defect,expected",
    [
        ("token", "invalid token"),
        ("mask", "mask is not binary"),
        ("boundary", "supervised first token"),
        ("lengths", "conversation lengths"),
        ("cp", "CP slice"),
        ("cu_seqlens", "cu_seqlens"),
    ],
)
def test_batch_diagnostics_identify_data_and_partition_errors(defect, expected):
    batch = _batch()
    if defect == "token":
        batch["unconcat_tokens"][0][0] = 128
    elif defect == "mask":
        batch["loss_masks"][0][0] = 2
    elif defect == "boundary":
        batch["loss_masks"][0][1] = 1
    elif defect == "lengths":
        batch["subseq_lens"][0] = [2, 3]
    elif defect == "cp":
        batch["tokens"][0][0] = 99
    else:
        batch["cu_seqlens"][1] = 3
    errors = finite_capture._batch_errors(batch, vocab_size=128, cp_size=2, cp_rank=0)
    assert any(expected in error for error in errors)


def test_fused_te_dummy_gradient_is_ignored_but_main_grad_is_checked(tmp_path):
    probe = _probe(tmp_path)
    probe.fail = Mock(side_effect=finite_capture.DiagnosticFailure)
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad_added_to_main_grad = True
    parameter.main_grad = torch.zeros(2)
    dummy = torch.full((2,), float("nan"))
    assert probe._parameter_gradient("weight", parameter, dummy) is dummy
    probe.fail.assert_not_called()
    parameter.main_grad[0] = float("nan")
    with pytest.raises(finite_capture.DiagnosticFailure):
        probe._parameter_gradient("weight", parameter, dummy)
    assert "main_grad" in probe.fail.call_args.args[0]


@pytest.mark.parametrize("fused,zero_out", [(False, False), (True, True)])
def test_real_parameter_gradients_are_checked(tmp_path, fused, zero_out):
    probe = _probe(tmp_path)
    probe.fail = Mock(side_effect=finite_capture.DiagnosticFailure)
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad_added_to_main_grad = fused
    parameter.zero_out_wgrad = zero_out
    parameter.main_grad = torch.zeros(2)
    with pytest.raises(finite_capture.DiagnosticFailure):
        probe._parameter_gradient("weight", parameter, torch.full((2,), float("nan")))
    assert "gradient" in probe.fail.call_args.args[0]


def test_first_nonfinite_saves_evidence_with_bounded_payload(tmp_path, monkeypatch):
    probe = _probe(tmp_path)
    monkeypatch.setattr(finite_capture, "_PAYLOAD_BYTES", 8)
    probe.context = {"microbatch": 2, "batch": {"sample_indices": [17]}}
    with pytest.raises(finite_capture.DiagnosticFailure, match="nonfinite"):
        probe.check("layer.output", torch.tensor([float("nan"), 1.0]), payload=torch.ones(10))
    claim = probe.capture.directory.parent / "first-value-capture"
    assert (claim / "complete").exists()
    assert torch.load(claim / "batch.pt", weights_only=True)["sample_indices"] == [17]
    manifest = json.loads((claim / "tensors.json").read_text())
    assert sum(entry["bytes"] for entry in manifest if "file" in entry) <= 8
    assert any("omitted" in entry for entry in manifest)
    assert json.loads((claim / "failure.json").read_text())["context"]["microbatch"] == 2


def test_lse_negative_infinity_sentinel_is_allowed(tmp_path):
    probe = _probe(tmp_path)
    probe.fail = Mock(side_effect=finite_capture.DiagnosticFailure)
    probe.check("lse", torch.tensor([0.0, -float("inf")]), allow_negative_inf=True)
    with pytest.raises(finite_capture.DiagnosticFailure):
        probe.check("lse", torch.tensor([float("inf")]), allow_negative_inf=True)


def test_noncontiguous_checks_are_chunked_and_cover_all_elements(monkeypatch):
    monkeypatch.setattr(finite_capture, "_CHECK_ELEMENTS", 4)
    value = torch.arange(30).reshape(5, 6).t()
    chunks = list(finite_capture._chunks(value))
    assert all(part.numel() <= 4 for part in chunks)
    assert sorted(torch.cat([part.flatten() for part in chunks]).tolist()) == list(range(30))


def test_kernel_failure_preserves_original_error_and_metadata_when_dump_fails(tmp_path, monkeypatch):
    capture = crash_capture._Capture(tmp_path, rank=0)
    error = RuntimeError("cuDNN kernel failed")
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", Mock(side_effect=RuntimeError("CUDA unavailable")))
    monkeypatch.setattr(crash_capture, "_memory_status", lambda: {})
    monkeypatch.setattr(crash_capture, "_node_report", lambda *args: None)
    wrapped = capture.wrap_kernel(Mock(side_effect=error), "sparse_attention_backward")
    with pytest.raises(RuntimeError) as raised:
        wrapped(torch.ones(2, 3))
    assert raised.value is error
    evidence = json.loads((capture.directory / "exception.json").read_text())
    assert evidence["extra"]["kernel"] == "sparse_attention_backward"
    assert evidence["recent_calls"][0]["args"]["items"][0]["shape"] == [2, 3]
    assert evidence["frames"]


def test_grouped_gemm_does_not_validate_uninitialized_destination(tmp_path):
    probe = _probe(tmp_path)
    destination = torch.full((2,), float("nan"))

    def gemm(a, b, out):
        out.copy_(a + b)
        return out

    assert torch.equal(probe.wrap_gemm(gemm)(torch.ones(2), torch.ones(2), destination), torch.full((2,), 2.0))


def test_rollout_log_preserves_default_logging_and_has_no_rollout_dependency(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    sample = SimpleNamespace(index=7, prompt={"pack_id": 9}, tokens=[1, 2], response_length=1, metadata=None)
    args = SimpleNamespace(save=str(tmp_path / "checkpoints"))
    assert finite_capture.log_rollout_data(3, args, [sample], {}, 1.0) is False
    records = json.loads((tmp_path / "diagnostics/job-42/batches/rollout-000003.json").read_text())
    assert records[0]["sample_index"] == 7
    assert records[0]["pack"] == {"pack_id": 9}


def test_unsupported_batch_layout_fails_before_optional_imports():
    with pytest.raises(ValueError, match="--allgather-cp"):
        finite_capture.install(SimpleNamespace(qkv_format="thd", allgather_cp=False))
