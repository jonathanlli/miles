import argparse

import pytest

from miles.utils.args.component_orchestrator import OrchestratorOnlyConfig


@pytest.mark.parametrize("duration", [None, 225.0])
def test_exit_duration_survives_orchestrator_serialization(duration):
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str)
    parser.add_argument("--exit-duration-in-mins", type=float, default=None)
    OrchestratorOnlyConfig.add_arguments(parser)
    argv = [] if duration is None else ["--exit-duration-in-mins", str(duration)]
    args = parser.parse_args(argv)
    config = OrchestratorOnlyConfig(**vars(args), requested_load=None)
    restored = OrchestratorOnlyConfig.model_validate_json(config.model_dump_json())
    assert restored.exit_duration_in_mins == duration
