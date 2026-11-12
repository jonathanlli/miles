"""Protect completed-rollout milestones in Megatron's checkpoint pruning.

Miles saves zero-based rollout IDs: with --save-retain-interval 500, checkpoint
499 represents 500 completed rollouts (500 updates for one-update SFT rollouts).
Megatron also retains its native iteration multiples; this conservative guard
adds protection without making any checkpoint newly eligible for deletion.
Deletion remains owned by Megatron's successful-save finalizer, so the previous
checkpoint is not removed until its successor has finished saving.
"""

import functools
import logging

logger = logging.getLogger(__name__)


def guard_deletion(delete, interval):
    """Keep completed-rollout milestones; delegate all other deletion unchanged."""
    if interval <= 0:
        raise ValueError("Checkpoint retention interval must be positive")

    @functools.wraps(delete)
    def wrapped(save_path, iteration_to_delete, *args, **kwargs):
        if iteration_to_delete >= 0 and (iteration_to_delete + 1) % interval == 0:
            logger.info(
                "Retaining Miles checkpoint %s (%s completed rollouts)", iteration_to_delete, iteration_to_delete + 1
            )
            return None
        return delete(save_path, iteration_to_delete, *args, **kwargs)

    return wrapped


def install(args):
    interval = getattr(args, "save_retain_interval", None)
    if interval is None:
        return
    if interval <= 0:
        raise ValueError("Checkpoint retention interval must be positive")
    # Megatron is optional outside the training initialization hook.
    from megatron.training import checkpointing

    function = checkpointing._async_delete_checkpoint_impl
    previous_interval = getattr(function, "_miles_retention_interval", None)
    if previous_interval is not None:
        if previous_interval != interval:
            raise ValueError("Checkpoint retention interval cannot change after initialization")
        return
    wrapped = guard_deletion(function, interval)
    wrapped._miles_retention_interval = interval
    checkpointing._async_delete_checkpoint_impl = wrapped
