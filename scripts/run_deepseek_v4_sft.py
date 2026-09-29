"""DeepSeek-V4-Flash-Base SFT training script (v18 agentic recipe).

=====================

Pure SFT of the DeepSeek-V4-Flash base model on pre-tokenized, pre-packed conversations
(``miles.rollout.packed_sft_rollout``): one dataset row is one pack of whole conversations, the
trainer builds per-conversation ``cu_seqlens`` inside the pack, and the loss is token-level cross
entropy over the assistant tokens of the whole global batch (``--calculate-per-token-loss``).

The recipe: global batch 32 packs of <= 520K tokens, 3000 steps, AdamW (0.9, 0.95, wd 0.1),
peak lr 2e-5 -> 5e-6 cosine with 150 warmup steps over a 5850-step horizon (the run stops at
~51% of the cosine, like the reference run), grad clip 1.0, no router aux loss, expert bias frozen.

Model side: ``--dsv4-impl megatron`` (native ``dsv4_hybrid`` attention, cuDNN fused DSA
indexer) with THD packing, tensor parallel 1 (required by that implementation), pipeline and
expert parallelism, and contiguous context parallelism for the long packs (``--allgather-cp``).
The torch_dist checkpoint must already exist (``scripts/run_deepseek_v4.py prepare-spmd`` or
``tools/convert_hf_to_torch_dist.py`` with ``--dsv4-impl megatron``).

This is pure SFT: ``train_async.py`` runs with ``--debug-train-only``, no SGLang engine starts.

=====================

Args:
  --hf-checkpoint: BF16 HF checkpoint dir (tokenizer + config).
  --ref-load: Megatron torch_dist checkpoint of the base model.
  --data-root / --train-order: materialized shard root and the blended pack order (jsonl).
  --num-nodes / --num-gpus-per-node: training allocation (default: 8 x 8).
  --pipeline-model-parallel-size / --context-parallel-size / --expert-model-parallel-size:
    parallel layout; TP is always 1. With 64 GPUs the default is PP8 x CP8 x EP8 (DP 1).
  --max-tokens-per-gpu: token budget of one micro-batch per CP rank (x CP = pack window).
  --exit-duration-in-mins: save and exit before the scheduler kills the job (0 disables).

=====================

  MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=<head-ip> python scripts/run_deepseek_v4_sft.py \\
      --hf-checkpoint /path/DeepSeek-V4-Flash-Base-bf16 --ref-load /path/DeepSeek-V4-Flash-Base_torch_dist \\
      --data-root /path/shards --train-order /path/train_order_520000.jsonl --output-dir /path/run
"""

from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    hf_checkpoint: str = "/root/models/DeepSeek-V4-Flash-Base-bf16"
    ref_load: str = "/root/models/DeepSeek-V4-Flash-Base_torch_dist"
    data_root: str = "/root/datasets/dsv4_v18/shards"
    train_order: str = "/root/datasets/dsv4_v18/blend/train_order_520000.jsonl"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8

    # data
    pack_target: int = 520000
    max_tokens: int = 524288  # window: an oversized single-conversation pack is truncated here
    global_batch_size: int = 32
    num_steps: int = 3000

    # parallelism (TP fixed at 1 by --dsv4-impl megatron); None derives the 64-GPU layout PP8 x CP8 x EP8
    # from the allocation (PP 8 when the world is a multiple of 64 GPUs, else 1; CP and EP fill up to 8)
    pipeline_model_parallel_size: int | None = None
    context_parallel_size: int | None = None
    expert_model_parallel_size: int | None = None
    decoder_first_pipeline_num_layers: int = 5
    decoder_last_pipeline_num_layers: int = 2
    # token budget of one micro-batch per CP rank; None = max_tokens / CP (one pack per micro-batch)
    max_tokens_per_gpu: int | None = None
    dsa_kernel_backend: str = "cudnn"
    # the fused CSA attention evaluates an indexer "teacher" even with a zero loss coefficient; the dense
    # variant (Megatron's default) costs seconds per layer at 512K, the sparse one is cheap
    dsa_indexer_use_sparse_loss: bool = True

    # optimizer / schedule
    lr: float = 2e-5
    min_lr: float = 5e-6
    lr_warmup_iters: int = 150
    lr_decay_iters: int = 5850
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    clip_grad: float = 1.0
    optimizer_cpu_offload: bool = True
    grad_reduce_in_fp32: bool = True

    # router: keep the base model's aux-loss-free balancing state, train the gate
    freeze_router_bias: bool = True
    freeze_router_gate: bool = False
    # late-bound OOM mitigation (see recipe section 6): e.g. 2.0 caps expert capacity, dropping by probs
    moe_expert_capacity_factor: float | None = None

    # checkpointing / wall clock
    save_interval: int = 50
    save_retain_interval: int = 500
    exit_duration_in_mins: int = 225
    train_memory_margin_bytes: int = 3 * 1024**3
    log_probs_chunk_size: int = 8192
    # a torch_dist save of this model is ~3.7 TB and takes >10 min; the default NCCL timeout kills it
    distributed_timeout_minutes: int = 60

    extra_args: str = ""

    def __post_init__(self):
        assert self.decoder_first_pipeline_num_layers >= 3, "the 3 hash-MoE layers must live on pipeline stage 0"
        world = self.num_nodes * self.num_gpus_per_node
        if self.pipeline_model_parallel_size is None:
            self.pipeline_model_parallel_size = 8 if world % 64 == 0 else 1
        if self.context_parallel_size is None:
            self.context_parallel_size = min(8, world // self.pipeline_model_parallel_size)
        if self.expert_model_parallel_size is None:
            self.expert_model_parallel_size = min(8, world // self.pipeline_model_parallel_size)
        if self.max_tokens_per_gpu is None:
            self.max_tokens_per_gpu = -(-self.max_tokens // self.context_parallel_size)
        model_parallel = self.pipeline_model_parallel_size * self.context_parallel_size
        assert world % model_parallel == 0, f"{world} GPUs not divisible by PP x CP = {model_parallel}"
        assert (world // self.pipeline_model_parallel_size) % self.expert_model_parallel_size == 0, (
            "EP must divide DP x CP"
        )
        assert self.max_tokens_per_gpu * self.context_parallel_size >= self.max_tokens, (
            "a whole pack must fit one micro-batch: max_tokens_per_gpu x CP >= max_tokens"
        )


def execute(args: ScriptArgs):
    U = args.create_backend()
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--load {args.output_dir}/checkpoints "
        f"--save {args.output_dir}/checkpoints "
        f"--save-interval {args.save_interval} "
        f"--save-retain-interval {args.save_retain_interval} "
        "--ckpt-fully-parallel-save-process-group ep_dp "
    )

    sft_args = (
        "--rollout-function-path miles.rollout.packed_sft_rollout.generate_rollout "
        f"--prompt-data {args.train_order} "
        "--input-key sample "
        f"--packed-sft-data-root {args.data_root} "
        f"--packed-sft-pack-target {args.pack_target} "
        f"--packed-sft-max-tokens {args.max_tokens} "
        # the order file is already blended and shuffled: consume it once, in order
        f"--num-rollout {args.num_steps} "
        f"--rollout-batch-size {args.global_batch_size} "
        f"--global-batch-size {args.global_batch_size} "
        "--loss-type sft_loss "
        "--calculate-per-token-loss "
        "--disable-compute-advantages-and-returns "
        # no rollout generation at all, hence no sglang engine
        "--debug-train-only "
    )

    cp_args = ""
    if args.context_parallel_size > 1:
        # DSv4 hybrid attention supports CP only in the contiguous layout: miles slices the packed
        # stream contiguously (--allgather-cp) and Megatron is told so (--cp-partition-mode); the
        # sequence-packing-scheduler flag only satisfies Megatron's config check here (miles packs).
        cp_args = "--allgather-cp --cp-partition-mode contiguous --sequence-packing-scheduler dp_balanced "

    layer_split_args = ""
    if args.pipeline_model_parallel_size > 1:
        layer_split_args = (
            f"--decoder-first-pipeline-num-layers {args.decoder_first_pipeline_num_layers} "
            f"--decoder-last-pipeline-num-layers {args.decoder_last_pipeline_num_layers} "
        )
    perf_args = (
        "--tensor-model-parallel-size 1 "
        f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} "
        f"{layer_split_args}"
        f"--context-parallel-size {args.context_parallel_size} "
        f"--expert-model-parallel-size {args.expert_model_parallel_size} "
        "--expert-tensor-parallel-size 1 "
        f"{cp_args}"
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"--seq-length {args.max_tokens} "
        f"--log-probs-chunk-size {args.log_probs_chunk_size} "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style cosine "
        f"--min-lr {args.min_lr} "
        f"--lr-warmup-iters {args.lr_warmup_iters} "
        f"--lr-decay-iters {args.lr_decay_iters} "
        f"--weight-decay {args.weight_decay} "
        f"--adam-beta1 {args.adam_beta1} "
        f"--adam-beta2 {args.adam_beta2} "
        f"--clip-grad {args.clip_grad} "
    )
    if args.optimizer_cpu_offload:
        optimizer_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "

    router_args = "--moe-router-dtype fp32 "
    if args.freeze_router_bias:
        router_args += "--freeze-e-score-correction-bias "
    if args.freeze_router_gate:
        router_args += "--moe-router-freeze-gate "
    if args.moe_expert_capacity_factor is not None:
        router_args += f"--moe-expert-capacity-factor {args.moe_expert_capacity_factor} --moe-token-drop-policy probs "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        f"{'--accumulate-allreduce-grads-in-fp32 ' if args.grad_reduce_in_fp32 else ''}"
        "--dsv4-impl megatron "
        f"--dsa-kernel-backend {args.dsa_kernel_backend} "
        f"{'--dsa-indexer-use-sparse-loss ' if args.dsa_indexer_use_sparse_loss else ''}"
        "--model-name deepseekv4 "  # for mbridge load
        "--qkv-format thd "
        f"--train-memory-margin-bytes {args.train_memory_margin_bytes} "
        f"--distributed-timeout-minutes {args.distributed_timeout_minutes} "
        f"{f'--exit-duration-in-mins {args.exit_duration_in_mins} ' if args.exit_duration_in_mins > 0 else ''}"
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{sft_args} "
        f"{optimizer_args} "
        f"{router_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="deepseek-v4-flash",
        megatron_path=args.megatron_path,
        train_script="train_async.py",
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
