import os
import argparse

from datasets import load_dataset

from transformers.trainer import get_scheduler
from openrlhf.datasets import SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import SFTTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

from code_dagger.utils import get_diff, apply_patch
from code_dagger.train import train_policy
from code_dagger.sample import sample_trajectories

def main(args):

    policy_api_base = args.api_base if args.api_base is not None else args.policy_api_base
    policy_api_key = args.api_key if args.api_key is not None else args.policy_api_key

    expert_api_base = args.api_base if args.api_base is not None else args.expert_api_base
    expert_api_key = args.api_key if args.api_key is not None else args.expert_api_key

    assert policy_api_base is not None, "Policy API base is required"
    assert expert_api_base is not None, "Expert API base is required"

    for iteration in range(0, max_iterations):

        # Check if files exist so the iteration can be skipped
        # check if sampling is needed
        do_sampling = True

        if iteration == 0:
            policy_model = args.base_policy_model

        if do_sampling and sample_trajectories(
            policy_model=policy_model,
            expert_model=args.base_expert_model,
            task_path=args.task_path,
            policy_api_base=policy_api_base,
            policy_api_key=policy_api_key,
            expert_api_base=expert_api_base,
            expert_api_key=expert_api_key,
            output_path=args.output_trajectory_path,
            base_task=args.base_task,
            backend=args.backend,
            iteration=iteration,
            max_new_tokens=args.max_new_tokens,
            n_samples=None,
        ):

            policy_trajectory = load_dataset("json", data_files=os.path.join(output_path, f"{iteration}:{base_task}:policy", "output.jsonl"), split="train")
            expert_trajectory = load_dataset("json", data_files=os.path.join(output_path, f"{iteration}:{base_task}:expert", "output.jsonl"), split="train")
            # Process the trajectories
            # idx = 0
            # base_code = get_code_snippet(d[idx]['step'][0]['full_input'])
            # predicted = d[idx]['step'][0]['aux']['predicted']
            # x1 = apply_patch(predicted, base_code)
            # x2 = d[idx]['answer'][0].split("\n")
            # get_diff(x1, x2)
            pass

        command = [
            "deepspeed", "--master_port", "8291", "--module", "openrlhf.cli.train_sft",
            "--save_path", f"{args.output_train_path}/model/",
            "--ckpt_path", f"{args.output_train_path}/ckpt/",
            "--load_checkpoint",
            "--save_steps", "100",
            "--logging_steps", "1",
            "--eval_steps", "-1",
            "--train_batch_size", "512",
            "--micro_train_batch_size", "32",
            "--pretrain", policy_model,
            "--save_hf_ckpt",
            "--bf16",
            "--max_epochs", f"{args.max_epochs}",
            "--max_samples", "128000",
            "--max_len", "1024",
            "--zero_stage", "3",
            "--learning_rate", "1e-5",
            "--lr_warmup_ratio", "0.001",
            "--dataset", f"json@{args.train_dataset_path}/",
            "--input_key", "input",
            "--output_key", "output",
            "--flash_attn",
            "--input_template", "{}",
            "--gradient_checkpointing",
            "--adam_offload",
            "--packing_samples",
            # "--use_ds_universal_ckpt",
            # "--use_liger_kernel",
        ]

        # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process = subprocess.Popen(command)
        exit_code = process.wait()

        policy_model = f"{args.output_train_path}/model/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Sampling
    parser.add_argument("--base_expert_model", type=str, required=True, help="Path to the base expert model")
    parser.add_argument("--base_policy_model", type=str, required=True, help="Path to the base policy model")
    parser.add_argument("--base_task", type=str, required=True, help="Base task identifier")
    parser.add_argument("--output_trajectory_path", type=str, required=True, help="Path to save the output")
    parser.add_argument("--max_iterations", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta parameter for blending")
    parser.add_argument("--backend", type=str, default="vllm", help="Backend to use")
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for the API")
    parser.add_argument("--api_key", type=str, default=None, help="API key for authentication")
    parser.add_argument("--task_path", type=str, default=None, help="Path to the task file")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to generate")
    parser.add_argument("--policy_api_base", type=str, default=None, help="Base URL for the policy API")
    parser.add_argument("--policy_api_key", type=str, default=None, help="API key for the policy API")
    parser.add_argument("--expert_api_base", type=str, default=None, help="Base URL for the expert API")
    parser.add_argument("--expert_api_key", type=str, default=None, help="API key for the expert API")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens to generate")

    # Training
    parser.add_argument("--train_dataset_path", type=str, default="sft training dataset path")
    parser.add_argument("--max_epochs", type=int, default=2)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed Tensor parallel size")

    # custom dataset
    parser.add_argument("--dataset", type=str, default=None, help="Path to the training dataset")
    parser.add_argument("--dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=1000000, help="Maximum number of samples to use")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--multiturn", action="store_true", default=False, help="Use compacted multiturn dataset")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_sft")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()
    # Evaluate the model. 
    # Train to recover from current state
    # Ablations: SFT, SFT+DPO
    # How would a model track error recovery?
    # Hypothesis: Learn to recognize error based on expert's trajectory.
    # Patches from fully generated code.
    main(args)