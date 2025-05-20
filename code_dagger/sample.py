import os
import time
import asyncio
import subprocess

from yeval.utils import import_modules
from yeval.task import TASK_LIST, YevalTask
from yeval.evaluation import EvaluateSystem


def get_host_and_port(api_base):
    if "/v1/" in api_base:
        api_base = api_base.split("/v1/")[0]
    elif api_base.endswith("/"):
        api_base = api_base[:-1]
    *host, port = api_base.split(":")
    host = ":".join(host)
    return host, port

class Server:
    def __init__(
        self,
        model_name,
        host="http://127.0.0.1", port=8000, backend="vllm",
        max_model_len=4096,
        pp_size=1, tp_size=1
    ):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.backend = backend
        self.max_model_len = max_model_len
        self.pp_size = pp_size
        self.tp_size = tp_size

    def start(self):

        if self.backend == "ollama":
            command = [
                "ollama",
                "run",
                self.model_name,
            ]
        elif self.backend == "vllm":
            command = [
                "vllm", "serve", self.model_name,
                # "--host", str(self.host),
                "--port", str(self.port),
                "--max_model_len", str(self.max_model_len),
                "--pipeline_parallel_size", str(self.pp_size),
                "--tensor_parallel_size", str(self.tp_size),
                "--distributed-executor-backend", "mp"
            ]

        self.process = subprocess.Popen(command, shell=False, stdout=subprocess.DEVNULL)
        print(f"{self.backend} server {self.model_name}, started with PID: {self.process.pid}")
        return self.process

    def stop(self):
        if self.backend == "ollama":
            command = ["ollama", "stop", self.model_name]
            self.stop_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.stop_process.wait()
        elif self.backend == "vllm":
            self.process.terminate()
            self.process.wait()
        print(f"{self.backend} server terminated.")

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
):

    if task_path is not None:
        import_modules(task_path)

    POLICY = {
        "policy": {
            "model": policy_model,
            "api_base": policy_api_base,
            "api_key": policy_api_key,
        },
        "expert": {
            "model": expert_model,
            "api_base": expert_api_base,
            "api_key": expert_api_key,
        },
    }
    # inference with policy
    # inference with expert
    # a. inference on task
    # b. inference on visited states by policy
    # Sample based on 
    # pi_i = beta_i * pi_star + (1 - beta_i) * pi_hat_i

    try:
        for pi_i in ["policy", "expert"]:
            print(f"Running {pi_i} model, {POLICY[pi_i]['model']}")
            host, port = get_host_and_port(POLICY[pi_i]['api_base'])
            model_api = Server(
                model_name=POLICY[pi_i]['model'],
                host=host, port=port, backend=backend,
                max_model_len=max_model_len
                )
            process = model_api.start()
            evaluator = EvaluateSystem(
                model=POLICY[pi_i]['model'],
                api_base=POLICY[pi_i]['api_base'],
                api_key=POLICY[pi_i]['api_key'],
                chat_completion=False,
                max_new_tokens=max_model_len-512,
                output_path=output_path,
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
                        # sampling_args=simple_parse_args_string(args.sample_args) if args.sample_args else None,
                        run_name=task_run_name,
                        n_samples=n_samples
                    )
                )

            model_api.stop()
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