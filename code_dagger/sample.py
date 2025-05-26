import os
import time
import asyncio
import subprocess

from yeval.utils import import_modules
from yeval.task import TASK_LIST, YevalTask
from yeval.evaluation import EvaluateSystem

# TODO: Mechanism to adjust compute budget for expert model
# Budget 6ND
# Trajectory
# 1. State 1 -> Full Code to Edits
# 2. State 2 -> Full Code to Edits
# 3. State End
def sample_trajectories(
    model,
    api_base,
    api_key,
    task_dict,
    output_path,
    task_path=None,
    iteration=0,
    n_samples=None,
    max_rps=2048,
    chat_completion=True,
    ):

    if task_path is not None:
        import_modules(task_path)

    evaluator = EvaluateSystem(
        model=model,
        api_base=api_base,
        api_key=api_key,
        chat_completion=chat_completion,
        max_new_tokens=512,
        output_path=output_path,
        max_rps=max_rps,
        )

    for task_name, task_kwargs in task_dict.items():
        print(f"Running task: {task_name}")
        task_run_name = f"{iteration}:{task_name}"
        policy_output_path = os.path.join(output_path, task_run_name)
        task_object = TASK_LIST[task_name](
            **task_kwargs,
            )

        asyncio.run(
            evaluator.run(
                task_object,
                run_name=task_run_name,
                n_samples=n_samples
            )
        )

    return True

if __name__ == "__main__":
    # policy_api_base = expert_api_base = "http://127.0.0.1:11434/v1/"
    # policy_api_key = expert_api_key = "ollama"
    # policy_model = "qwen3-4b-edit"
    # expert_model = "qwen3-4b"
    # backend = "ollama"

    policy_api_base = expert_api_base = "http://127.0.0.1:9000/v1/"
    policy_api_key = expert_api_key = "vllm"
    policy_model = "/data/user_data/lsutawik/edit-code/Qwen-Qwen3-4B/model/"
    expert_model = "Qwen/Qwen3-4B"
    backend = "vllm"

    sample_trajectories(
        policy_model,
        policy_api_base,
        policy_api_key,
        expert_model,
        expert_api_base,
        expert_api_key,
        "output/",
        "mbpp",
        task_path="tasks/",
        backend=backend,
        iteration=0,
        max_new_tokens=1024,
        n_samples=4,
    )