import os
import sys
import time
import argparse
import jsonlines
import subprocess
from datasets import load_dataset

from .utils import get_diff, apply_patch
from .sample import sample_trajectories
from .aggregate import aggregate_dataset

from yeval.model import Server, get_host_and_port

def main(args):

    for iteration in range(0, args.max_iterations):

        # Check if files exist so the iteration can be skipped
        # check if sampling is needed
        if iteration == 0:
            policy_model = args.base_policy_model

        if not args.no_sampling:

            base_task = args.base_task

            policy_api_base = args.policy_api_base or args.api_base
            policy_api_key = args.policy_api_key or args.api_key

            expert_api_base = args.expert_api_base or args.api_base
            expert_api_key = args.expert_api_key or args.api_key

            assert policy_api_base is not None, "Policy API base is required"
            assert expert_api_base is not None, "Expert API base is required"

            PI = {
                "policy": {
                    "model": policy_model,
                    "api_base": policy_api_base,
                    "api_key": policy_api_key,
                    "chat_completion": False,
                    "tasks": {
                        f"{base_task}:policy": {}
                    }
                },
                "expert": {
                    "model": args.base_expert_model,
                    "api_base": expert_api_base,
                    "api_key": expert_api_key,
                    "chat_completion": True,
                    "tasks": {
                        f"{base_task}:expert": {},
                        "expert_evaluation": {
                            "data_kwargs": {
                                "data_files":  os.path.join(
                                                    args.output_trajectory_path,
                                                    f"{iteration}:{base_task}:policy",
                                                    "output.jsonl"
                                                    )
                            }
                        }
                    }
                },
            }

            # inference with policy
            # inference with expert
            # a. inference on task
            # b. inference on visited states by policy
            # Sample based on 
            # pi_i = beta_i * pi_star + (1 - beta_i) * pi_hat_i

            for pi_i in ["policy", "expert"]:
                if pi_i == "policy" and args.no_policy:
                    continue
                elif pi_i == "expert" and args.no_expert:
                    continue

                if args.serve:
                    host, port = get_host_and_port(PI[pi_i]['api_base'])
                    policy_api = Server(
                        model_name=PI[pi_i]["model"],
                        host=host, port=port, backend=args.backend,
                        pp_size=args.pp_size, tp_size=args.tp_size,
                        max_model_len=args.max_model_len
                        )
                    process = policy_api.start()

                sample_trajectories(
                    model=PI[pi_i]["model"],
                    api_base=PI[pi_i]['api_base'],
                    api_key=PI[pi_i]['api_key'],
                    task_dict=PI[pi_i]["tasks"],
                    chat_completion=PI[pi_i]["chat_completion"],
                    output_path=args.output_trajectory_path,
                    task_path=args.task_path,
                    iteration=iteration,
                    n_samples=args.n_samples,
                )

                if args.serve:
                    policy_api.stop(process)

            agg_trajectory = aggregate_dataset(
                iteration,
                args.base_task,
                args.output_trajectory_path,
            )

            sft_data_path = os.path.join(args.output_trajectory_path, f"{iteration}:sft_dataset/")
            os.makedirs(sft_data_path, exist_ok=True)
            with jsonlines.open(os.path.join(sft_data_path, "output.jsonl"), mode="w") as writer:
                writer.write_all(agg_trajectory)

            if args.only_do_sampling or args.no_expert or args.no_policy:
                sys.exit()

        command = [
            "deepspeed", "--master_port", "8291", "--module", "openrlhf.cli.train_sft",
            "--save_path", f"{args.output_train_path}/{iteration}:model/",
            "--ckpt_path", f"{args.output_train_path}/ckpt/",
            "--load_checkpoint",
            "--save_steps", "100",
            "--logging_steps", "1",
            "--eval_steps", "-1",
            "--train_batch_size", "512",
            "--micro_train_batch_size", f"{args.micro_train_batch_size}",
            "--pretrain", f"{policy_model}",
            "--save_hf_ckpt",
            "--bf16",
            "--max_epochs", f"{args.max_epochs}",
            "--max_samples", "128000",
            "--max_len", "1024",
            "--zero_stage", "3",
            "--learning_rate", "1e-4",
            "--lr_warmup_ratio", "0.001",
            "--dataset", f"json@{args.train_dataset_path or sft_data_path}/",
            "--input_key", "input",
            "--output_key", "output",
            "--flash_attn",
            "--input_template", "{}",
            "--gradient_checkpointing",
            "--adam_offload",
            "--packing_samples",
            "--seed", f"{args.seed}",
            # "--use_ds_universal_ckpt",
            # "--use_liger_kernel",
        ]

        try:
            process = subprocess.Popen(command)
            exit_code = process.wait()
        except:
            if "process" in locals():
                process.terminate()
                process.wait()

        policy_model = f"{args.output_train_path}/model/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Control Parameters
    parser.add_argument("--only_do_sampling", action="store_true", default=False, help="Skip training and only do sampling")
    parser.add_argument("--output_train_path", type=str, default="data/", help="Path to save the training output")
    parser.add_argument("--no_policy", action="store_true", default=False, help="don't sample from policy")
    parser.add_argument("--no_expert", action="store_true", default=False, help="don't sample from expert")
    parser.add_argument("--no_sampling", action="store_true", default=False, help="don't sample at all")
    parser.add_argument("--serve", action="store_true", default=False, help="Serve models")
    parser.add_argument("--max_rps", type=int, default=10, help="Max requests per second")

    # Sampling
    parser.add_argument("--base_expert_model", type=str, required=True, help="Path to the base expert model")
    parser.add_argument("--base_policy_model", type=str, required=True, help="Path to the base policy model")
    parser.add_argument("--base_task", type=str, required=True, help="Base task identifier")
    parser.add_argument("--output_trajectory_path", type=str, required=True, help="Path to save the output")
    parser.add_argument("--max_iterations", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta parameter for blending")
    parser.add_argument("--backend", type=str, default="vllm", help="Backend to use")
    parser.add_argument("--api_base", type=str, default="http://127.0.0.1:9000/v1/", help="Base URL for the API")
    parser.add_argument("--api_key", type=str, default="token-dummy", help="API key for authentication")
    parser.add_argument("--task_path", type=str, default=None, help="Path to the task file")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to generate")
    parser.add_argument("--policy_api_base", type=str, default=None, help="Base URL for the policy API")
    parser.add_argument("--policy_api_key", type=str, default=None, help="API key for the policy API")
    parser.add_argument("--expert_api_base", type=str, default=None, help="Base URL for the expert API")
    parser.add_argument("--expert_api_key", type=str, default=None, help="API key for the expert API")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Max new tokens to generate")
    parser.add_argument("--pp_size", type=int, default=1, help="Number of samples to generate in parallel")
    parser.add_argument("--tp_size", type=int, default=1, help="Number of samples to generate in parallel")

    # Training
    parser.add_argument("--train_dataset_path", type=str, default=None, help="sft training dataset path")
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=1000000, help="Maximum number of samples to use")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")

    # # wandb parameters
    # parser.add_argument("--use_wandb", type=str, default=None)
    # parser.add_argument("--wandb_org", type=str, default=None)
    # parser.add_argument("--wandb_group", type=str, default=None)
    # parser.add_argument("--wandb_project", type=str, default="openrlhf_train_sft")
    # parser.add_argument(
    #     "--wandb_run_name",
    #     type=str,
    #     default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    # )

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
