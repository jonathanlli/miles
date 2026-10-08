import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.utils.checkpoint_retention import guard_deletion, install


@pytest.mark.parametrize("iteration", [499, 999, 1499])
def test_completed_rollout_milestones_survive_pruning(tmp_path, iteration):
    checkpoint = tmp_path / f"iter_{iteration:07d}"
    checkpoint.mkdir()
    delete = Mock(side_effect=lambda *args, **kwargs: checkpoint.rmdir())
    guard_deletion(delete, 500)(tmp_path, iteration, True, lower_priority=True)
    assert checkpoint.is_dir()
    delete.assert_not_called()


@pytest.mark.parametrize("iteration", [49, 84, 498, 549, 998, 1049])
def test_intermediates_keep_megatron_deletion_contract(iteration):
    delete = Mock(return_value="deleted")
    guarded = guard_deletion(delete, 500)
    assert guarded("checkpoints", iteration, True, True, cpu_priority=10, io_priority=3) == "deleted"
    delete.assert_called_once_with("checkpoints", iteration, True, True, cpu_priority=10, io_priority=3)


def test_keyword_arguments_and_interval_one():
    delete = Mock()
    guard_deletion(delete, 1)(save_path="checkpoints", iteration_to_delete=0)
    delete.assert_not_called()


@pytest.mark.parametrize("interval", [0, -500])
def test_invalid_interval_is_rejected(interval):
    with pytest.raises(ValueError, match="positive"):
        guard_deletion(Mock(), interval)
    with pytest.raises(ValueError, match="positive"):
        install(SimpleNamespace(save_retain_interval=interval))


def test_unset_retention_does_not_import_megatron(monkeypatch):
    monkeypatch.setitem(sys.modules, "megatron", None)
    install(SimpleNamespace())
    install(SimpleNamespace(save_retain_interval=None))


def test_install_is_idempotent_and_rejects_changed_policy(monkeypatch):
    megatron = ModuleType("megatron")
    training = ModuleType("megatron.training")
    checkpointing = ModuleType("megatron.training.checkpointing")
    delete = Mock(_miles_retention_interval=None)
    checkpointing._async_delete_checkpoint_impl = delete
    training.checkpointing = checkpointing
    megatron.training = training
    for module in (megatron, training, checkpointing):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    install(SimpleNamespace(save_retain_interval=500))
    guarded = checkpointing._async_delete_checkpoint_impl
    install(SimpleNamespace(save_retain_interval=500))
    assert checkpointing._async_delete_checkpoint_impl is guarded
    guarded("checkpoints", 499)
    delete.assert_not_called()
    guarded("checkpoints", 549)
    delete.assert_called_once_with("checkpoints", 549)
    with pytest.raises(ValueError, match="cannot change"):
        install(SimpleNamespace(save_retain_interval=1000))
