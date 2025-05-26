import sys
import subprocess
import numpy as np

from yeval.task import register_task, YevalTask
from yeval.metrics.pass_at_k import classical_pass_at_k, openai_pass_at_k

from code_dagger.utils import (
    apply_patch,
    get_code_snippet,
    )

def postprocess_patch(x, state):
    x = "@@" + x
    original_input = state["full_input"].split("<|diff|>")[0].strip()
    current_code = [line for line in original_input.split("```") if line != ""][-1].strip()
    updated_code = "\n".join(apply_patch(x, current_code))

    return updated_code, state

def convert_to_patch(code_snippet):
    if isinstance(code_snippet, str):
        code_snippet = [code_snippet.strip()]
    code_length = len(code_snippet)
    code_patch = "\n".join([f"@@ -0,0 +1,{code_length} @@"]+[f"+{line}" for line in code_snippet])
    return code_patch

def preprocess_patch(x, state):

    current_step = state["current_step"]
    if current_step > 0:
        current_code = state["step"][current_step-1]["output"][0]
        # current_code = current_code.split("\n")
        # code_length = len(current_code)
        # updated_code = "\n".join([f"@@ -0,0 +1,{code_length} @@"]+[f"+{line}" for line in current_code])
        updated_code = f"```\n{current_code}\n```"
        # x = x.split("<|diff|>")[0]+f"<|diff|>{updated_code}\n<|diff|>@@"
        x = x.split("```")[0]+updated_code+"\n<|diff|>@@"
    
    return x, state

def exit_fn(x, state):
    if x.strip().endswith("<|diff|>\n<|diff|>"):
        return True
    else:
        return False

def pass_at_1(completion, test):
    try:
        test_program = completion + "\n" + "\n".join(test)
        subprocess_result = subprocess.run([sys.executable, "-c", test_program], timeout=10, text=True, capture_output=True)
        if subprocess_result.returncode == 0:
            return 1
        return 0
    except Exception as e:
        return 0

@register_task("mbpp:policy")
class MBPPStep(YevalTask):
    data_path="evalplus/mbppplus"
    input_text=lambda x: f"{x['prompt']}\n```\n{x["code"].split(":")[0].strip()+":"}\n```\n<|diff|>@@"
    loop_max=10
    # sampling_args={"n": 10}
    loop_exit=exit_fn
    output_text=lambda x: x["test_list"]
    test_split="test"
    evaluation={"pass@1": pass_at_1}
    # eval_at_k=True
    # evaluation={
    #     "pass@1": partial(openai_pass_at_k, k=1, metric_fn=pass_at_1),
    #     "pass@10": partial(openai_pass_at_k, k=10, metric_fn=pass_at_1)
    # }
    preprocessor=preprocess_patch
    postprocessor=postprocess_patch

@register_task("mbpp:expert")
class MBPPStep(YevalTask):
    data_path="evalplus/mbppplus"
    input_text=lambda x: f"{x['prompt']}\n```\n{x["code"].split(":")[0].strip()+":"}\n```"+"/no_think"
    sampling_args={
        "n": 10,
        "temperature": 0.7,
        }
    output_text=lambda x: x["test_list"]
    test_split="test"
    sample_agg_fn={
        "pass@1": np.mean,
        "idx_pass@1": lambda x: x,
        # "pass@10": lambda x: 1 if sum(x)>0 else 0,
        }
    evaluation={
        "pass@1": pass_at_1,
        "idx_pass@1": pass_at_1,
        # "pass@10": pass_at_1,
        }
    # eval_at_k=True
    # evaluation={
    #     "pass@1": partial(openai_pass_at_k, k=1, metric_fn=pass_at_1),
    #     "pass@10": partial(openai_pass_at_k, k=10, metric_fn=pass_at_1)
    # }
    postprocessor=get_code_snippet

def unroll_trajectories(dataset):

    def _unroll(examples):
        all_idx = []
        all_score = []
        all_sample_id = []
        all_sentence = []
        all_ground_truth = []
        all_original_completion = []
        for idx, (score, sample_id, trajectory) in enumerate(
            zip(
                examples["pass@1"],
                examples["sample_id"],
                examples["step"],
                )
        ):
            for step in trajectory:
                all_idx.append(idx)
                all_score.append(score)
                all_sample_id.append(sample_id)
                all_sentence.append(step["full_input"].split("<|diff|>")[0].strip())
                all_original_completion.append("@@"+step["completion"][0])
                all_ground_truth.append(step["ground_truth"])

        return {
            "idx": all_idx,
            "original_score": all_score,
            "sample_id": all_sample_id,
            "input": all_sentence,
            "output": all_ground_truth,
            "predicted": all_original_completion,
            }

    for key in dataset.num_columns.keys():
        dataset[key] = dataset[key].map(_unroll, batched=True, remove_columns=dataset[key].column_names)
    return dataset


@register_task("expert_evaluation")
class ExpertEvaluation(YevalTask):
    data_path="json"
    input_text=lambda x: "Complete/Fix the code snippet based on the following command\n"+x["input"].split("<|diff|>")[0].strip()+"/no_think"
    output_text=lambda x: x["output"]
    sampling_args={
        "n": 10,
        "temperature": 1.0,
        }
    sample_agg_fn={
        "pass@1": np.mean,
        "idx_pass@1": lambda x: x,
        }
    test_split="train"
    evaluation={
        "pass@1": pass_at_1,
        "idx_pass@1": pass_at_1,
        }
    preprocessing=unroll_trajectories
    postprocessor=get_code_snippet
    aux_keys=[
        "sample_id",
        "predicted",
        "original_score"
        ]
    # eval_at_k=True
    # evaluation={
    #     "pass@1": partial(openai_pass_at_k, k=1, metric_fn=pass_at_1),
    #     "pass@10": partial(openai_pass_at_k, k=10, metric_fn=pass_at_1)
    # }

@register_task("gsm8k_patch_by_patch")
class GSM8kStep(MBPPStep):
    data_path="openai/gsm8k"
    data_name="main"
    input_text=lambda x: f"{x['question']}\nWrite a function to solve the following problem.\n```\ndef solution():\n```\n<|diff|>@@"
    loop_max=10
    loop_exit=exit_fn
    output_text=lambda x: [f"assert solution() == {x["answer"].split("####")[-1].strip()}"]
    test_split="test"
    evaluation={"pass@1": pass_at_1}
    # evaluation={
    #     "pass@1": partial(openai_pass_at_k, k=1, metric_fn=pass_at_1),
    #     "pass@10": partial(openai_pass_at_k, k=10, metric_fn=pass_at_1)
    # }
    preprocessor=preprocess_patch
    postprocessor=postprocess_patch


if __name__ == "__main__":
    pass
