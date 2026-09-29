"""SFT rollout over pre-tokenized, pre-packed conversations.

One dataset row is one *pack*: ``{"shard": tag, "pack": k}`` (see ``--input-key``).  The pack index of a
shard lists its documents; their token ids and loss masks are read from the shard's binary token store.
A pack becomes one :class:`Sample` whose ``metadata["subseq_lens"]`` carries the conversation boundaries so
the trainer can build per-conversation ``cu_seqlens`` (no attention across conversations, positions restart).

Shard layout (``<root>/<tag>/``): ``docs.tokens.u32``, ``docs.mask.u8``, ``docs.index.npy`` (offset, length,
trained per document) and ``packs.<target>.json`` (``{"packs": [{"docs": [...]}]}``).

Args used: ``--packed-sft-data-root``, ``--packed-sft-pack-target``, ``--packed-sft-max-tokens``.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

SAMPLE_PRINTED = False


@lru_cache(maxsize=None)
def _shard(root: str, tag: str, target: int):
    base = Path(root) / tag
    index = np.load(base / "docs.index.npy")
    tokens = np.memmap(base / "docs.tokens.u32", dtype="<u4", mode="r")
    mask = np.memmap(base / "docs.mask.u8", dtype=np.uint8, mode="r")
    packs = json.loads((base / f"packs.{target}.json").read_text())["packs"]
    return index, tokens, mask, packs


def load_pack(root: str, tag: str, pack_id: int, target: int, max_tokens: int) -> tuple[list[int], list[int], list[int]]:
    """Return (token ids, loss mask, per-conversation lengths) of one pack, truncated to ``max_tokens``."""
    index, tokens, mask, packs = _shard(root, tag, target)
    ids, loss, lens = [], [], []
    budget = max_tokens
    for doc in packs[pack_id]["docs"]:
        offset, length, _ = (int(v) for v in index[doc])
        take = min(length, budget)
        if take <= 0:
            break
        ids.extend(tokens[offset : offset + take].tolist())
        loss.extend(mask[offset : offset + take].tolist())
        lens.append(take)
        budget -= take
    return ids, loss, lens


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    assert args.rollout_global_dataset
    global SAMPLE_PRINTED

    samples = data_buffer.get_samples(args.rollout_batch_size)
    for i, sample in enumerate(samples):
        (sample,) = sample
        spec = sample.prompt
        if isinstance(spec, str):
            spec = json.loads(spec)
        ids, loss, lens = load_pack(
            args.packed_sft_data_root, spec["shard"], int(spec["pack"]), args.packed_sft_pack_target, args.packed_sft_max_tokens
        )
        first = next((k for k, v in enumerate(loss) if v), len(ids))
        if first >= len(ids) - 1:
            # nothing trainable survived truncation; keep one masked response token so shapes stay valid
            first = len(ids) - 1
            loss[-1] = 0
        response_length = len(ids) - first
        sample.tokens = ids
        sample.response_length = response_length
        sample.loss_mask = loss[first:]
        sample.reward = 0
        sample.metadata = {**(sample.metadata or {}), "subseq_lens": lens}
        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                f"packed_sft_rollout: {spec=} tokens={len(ids)} conversations={len(lens)} "
                f"trained={sum(loss)} response_length={response_length}"
            )
            SAMPLE_PRINTED = True
    return samples
