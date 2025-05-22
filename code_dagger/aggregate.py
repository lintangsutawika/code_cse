import os
import sys

from tqdm import tqdm
from datasets import load_dataset

from pylintseq.utils import (
    inflate_edit_path,
    lintseq_backward_sampling_pythonic,
    )

from code_dagger.utils import (
    get_diff,
    apply_patch,
    get_code_snippet,
    )

def aggregate_dataset(
    iteration,
    base_task,
    output_path,
):

    print("Loading trajectories...")
    all_trajectories = {
        "policy_trajectory": load_dataset("json", data_files=os.path.join(output_path, f"{iteration}:{base_task}:policy", "output.jsonl"), split="train"),
        "expert_trajectory": load_dataset("json", data_files=os.path.join(output_path, f"{iteration}:{base_task}:expert", "output.jsonl"), split="train"),
        "corrected_trajectory": load_dataset("json", data_files=os.path.join(output_path, f"{iteration}:expert_evaluation", "output.jsonl"), split="train"),
    }

    filtered_expert_trajectory = []
    for traj_name, trajectory in all_trajectories.items():
        print(f"Processing {traj_name}...")
        for sample in tqdm(trajectory):
            step = sample["step"][0]
            query = step["full_input"]
            starting_code = get_code_snippet(query).split("\n")
            

            if traj_name == "policy_trajectory":
                query = query.split("<|diff|>")[0].strip()
                metric = [step['eval']["pass@1"]]
                code_source = [["@@"+code for code in step["completion"]]]
            else:
                query = query.split("/no_think")[0]
                query = query.replace("Complete/Fix the code snippet based on the following command\n", "").strip()
                metric = step['eval']["idx_pass@1"]
                code_source = step['output']

            predicted = ""
            original_score = False
            if traj_name == "corrected_trajectory":
                predicted = step['aux']['predicted']
                original_score = step['aux']['original_score']

            if original_score:
                continue
            else:
                starting_code = apply_patch(predicted, starting_code)

            query = query.split("```")[0]

            previous_code = None
            for score, code_text in zip(metric, code_source):
                if score:

                    if code_text == previous_code:
                        continue
                    else:
                        previous_code = code_text

                    base_code = str(starting_code)
                    if traj_name in ["expert_trajectory", "corrected_trajectory"]:
                        edit_path = lintseq_backward_sampling_pythonic(
                            code_text,
                            children_per_round=1,
                            top_k=1,
                            max_population_size=1,
                            max_depth=512,
                            indent_bias_sampling_factor=1,
                            ignore_imports=False,
                            ignore_comments=True,
                            ignore_global_defs=True,
                            ignore_init_errors=False,
                        )

                        if edit_path is None:
                            continue

                        edit_sequence = edit_path[0][0]
                        _, diff_seq = inflate_edit_path(code_text, edit_sequence)

                        len_diff_seq = len(diff_seq)
                        for idx, seq in enumerate(diff_seq):
                            if (traj_name == "corrected_trajectory") and (idx == 0):
                                correct_patch = code_text.split("\n")
                                patch_seq = get_diff(base_code, correct_patch)
                            else:
                                patch_seq = seq

                            if idx == len_diff_seq - 1:
                                patch_seq += "\n<|diff|>\n<|diff|>"
                            else:
                                patch_seq += "\n<|diff|>"

                            state_code = "\n".join(base_code)

                            filtered_expert_trajectory.append({
                                    "input": query + f"\n```\n{state_code}\n```" if state_code != "" else query,
                                    "output": patch_seq,
                                    "source": traj_name,
                                })

                            base_code = apply_patch(seq, base_code)
                            # query = query + f"\n```\n{"\n".join(updated_code)}\n```"
                    else:
                        state_code = "\n".join(base_code)
                        query = query + f"\n```\n{state_code}\n```" if state_code != "" else query
                        filtered_expert_trajectory.append({
                                "input": query,
                                "output": code_text,
                                "source": traj_name,
                            })
            #         break
            # # if len(filtered_expert_trajectory) > 0:
            #     # break

    return filtered_expert_trajectory

if __name__ == "__main__":
    # Example usage
    iteration = 0
    base_task = "mbpp"
    output_path = "output/"
    aggregated_traj = aggregate_dataset(iteration, base_task, output_path)