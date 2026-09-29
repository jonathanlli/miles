import argparse

from miles.rollout.packed_sft_rollout import PackedSftConfig, generate_rollout
from miles.utils.args.custom_function import CustomFunctionConfig


def test_packed_flags_survive_custom_hook_serialization():
    parser = argparse.ArgumentParser()
    generate_rollout.config_class.add_arguments(parser)
    args = parser.parse_args(
        ["--packed-sft-data-root", "/shards", "--packed-sft-pack-target", "32", "--packed-sft-max-tokens", "40"]
    )
    hook = CustomFunctionConfig.model_validate(
        {"path": "miles.rollout.packed_sft_rollout.generate_rollout", "config": vars(args)}
    )
    restored = CustomFunctionConfig.model_validate_json(hook.model_dump_json())
    assert isinstance(restored.config, PackedSftConfig)
    assert restored.config.packed_sft_data_root == "/shards"
    assert restored.config.packed_sft_pack_target == 32
    assert restored.config.packed_sft_max_tokens == 40


def test_packed_flag_defaults_remain_compatible():
    config = PackedSftConfig()
    assert config.packed_sft_data_root is None
    assert config.packed_sft_pack_target == 520000
    assert config.packed_sft_max_tokens == 524288
