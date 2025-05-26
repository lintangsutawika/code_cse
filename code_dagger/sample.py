import os
import time
import asyncio
import subprocess

from yeval.utils import import_modules
from yeval.task import TASK_LIST, YevalTask
from yeval.evaluation import EvaluateSystem

from yeval.model import Server

# TODO: Mechanism to adjust compute budget for expert model
# Budget 6ND
# Trajectory
# 1. State 1 -> Full Code to Edits
# 2. State 2 -> Full Code to Edits
# 3. State End
def sample_trajectories(
    policy_model,
    policy_api_base,
    policy_api_key,
    expert_model,
    expert_api_base,
    expert_api_key,
    output_path,
    base_task,
    task_path=None,
    backend="vllm",
    iteration=0,
    max_model_len=2048,
    n_samples=None,
    policy_only=False,
    expert_only=False,
    no_policy=False,
    no_expert=False,
    pp_size=1,
    tp_size=1,
    max_rps=10,
    ):

    if task_path is not None:
        import_modules(task_path)

    # inference with policy
    # inference with expert
    # a. inference on task
    # b. inference on visited states by policy
    # Sample based on 
    # pi_i = beta_i * pi_star + (1 - beta_i) * pi_hat_i

    try:
        for pi_i in pi_list:
            print(f"Running {pi_i} model, {POLICY[pi_i]['model']}")
            model_api = Server(
                model_name=POLICY[pi_i]['model'],
                host=POLICY[pi_i]['api_base'],
                backend=backend,
                max_model_len=max_model_len,
                pp_size=pp_size,
                tp_size=tp_size,
            )
            process = model_api.start()

            evaluator = EvaluateSystem(
                model=POLICY[pi_i]['model'],
                api_base=POLICY[pi_i]['api_base'],
                api_key=POLICY[pi_i]['api_key'],
                chat_completion=False,
                max_new_tokens=max_model_len-512,
                output_path=output_path,
                max_rps=max_rps,
                )

            if pi_i == "policy":
                task_list = [f"{base_task}:policy"]
            if pi_i == "expert":
                task_list = [f"{base_task}:expert", "expert_evaluation"]

            for task_name in task_list:
                if task_name == "expert_evaluation":
                    task_kwargs = {
                        "data_kwargs": {
                            "data_files": f'{os.path.join(output_path, f"{iteration}:{base_task}:policy")}/output.jsonl'
                        }
                    }
                else:
                    task_kwargs = {}
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

            if (pi_i == "policy" and not no_policy) or (pi_i == "expert" and not no_expert):
                model_api.stop(process)
                time.sleep(10)

        return True
    except Exception as e:
        print(f"Error: {e}")
        if 'process' in locals():
            process.terminate()
            process.wait()
        return False

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