from miles.utils.args.schema import A, Arg, BaseConfig


class OrchestratorOnlyConfig(BaseConfig):
    load: A[str | None, Arg(reset=True)] = None
    requested_load: str | None
    exit_duration_in_mins: A[
        float | None,
        Arg(reset=True, help="Save and exit asynchronous training after this many minutes."),
    ] = None
