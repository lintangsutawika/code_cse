import functools
import jsonlines
import sys
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from argparse import ArgumentParser
import multiprocessing
import gc
import warnings

from datasets import load_dataset

warnings.filterwarnings("ignore")

from pylintseq.generate import subprocess_task
from pylintseq.utils import set_seed_everywhere

from code_dagger.utils import (
    apply_patch,
    convert_to_patch,
)

def get_code_state(sample):

    code_states = []
    edit_path = [edit for edit in sample["edit_path"] if edit != ""]
    updated_code = ""
    for idx, edit in enumerate(edit_path):
        if idx == 0:
            input_instruction = sample['source_instruction']
        else:
            # input_instruction = sample['source_instruction']+f"\n<|diff|>{edit_patch}\n<|diff|>\n<|diff|>"
            input_instruction = sample['source_instruction']+f"\n```\n{"\n".join(updated_code)}\n```"

        input_instruction += "\n<|diff|>"

        output_edit = f"{edit}\n<|diff|>"
        if idx == len(edit_path) - 1:
            output_edit += "\n<|diff|>"

        updated_code = apply_patch(edit, updated_code)
        edit_patch = convert_to_patch(updated_code)

        data_dict = {
            "instruction": input_instruction,
            "edit": output_edit,
        }
        code_states.append(data_dict)

    return code_states

def main(args):

    set_seed_everywhere(args.seed)

    data_field = [args.code_data_field]
    if not args.prompt_data_field is None:
        data_field.insert(0, args.prompt_data_field)

    dataset = load_dataset(
        args.data_path,
        args.data_name,
        data_files=args.data_files, split=args.data_split
        )
    df = dataset.to_pandas()[data_field].astype("string")

    if args.num_samples == -1:
        samples = np.arange(len(df))
    else:
        samples = np.random.choice(
            np.arange(len(df)), size=(args.num_samples,), replace=False
        )

    # Convert sample indices to numpy array
    samples = np.array(samples)

    # Number of parallel tasks
    num_proc = min(
        args.num_workers, multiprocessing.cpu_count()
    )  # Optimize based on CPU count
    num_paths_per_proc = 8  # Reduced paths to limit memory usage

    # Efficient task distribution across processes
    task_args = [
        (i, min(num_paths_per_proc, len(samples) - i))
        for i in range(0, len(samples), num_paths_per_proc)
    ]

    total_tasks = len(samples) * args.num_edit_paths_per_sample
    batch_size = 64  # Slightly larger batch for faster processing but manageable memory

    args.name_or_path = args.data_path
    source_file = args.data_path.replace("/", "-")
    file_name = source_file[: source_file.find(".")] \
        + f"_{args.num_samples}_{args.num_edit_paths_per_sample}_{args.seed}.jsonl"

    if args.as_states:
        file_name = "states_" + file_name

    with tqdm(total=total_tasks, desc="Processing", ncols=100) as pbar:
        with jsonlines.open(os.path.join(args.output_path, file_name), mode="w") as writer:
            for batch_start in range(0, len(task_args), batch_size):

                # Parallel processing of subprocesses with minimal data passed
                results = Parallel(n_jobs=num_proc, backend="loky", timeout=1000)(
                    delayed(subprocess_task)(
                        start_i,
                        num_samples,
                        args,
                        df,
                        samples,
                    )
                    for start_i, num_samples in task_args[
                        batch_start : batch_start + batch_size
                    ]
                )

                # Collect results and write to file
                for result in results:
                    if result:
                        if args.as_states:
                            state_result = [state for sample in result for state in get_code_state(sample)]
                            writer.write_all(state_result)
                        else:
                            writer.write_all(result)
                        pbar.update(len(result))

                # Explicit garbage collection after processing batch
                gc.collect()


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare data for Code Trajectory")
    # Dataset
    parser.add_argument("--data_path", type=str, required=True, help="Dataset to prepare")
    parser.add_argument("--data_name", type=str, default=None, help="Dataset to prepare")
    parser.add_argument("--data_files", type=str, default=None, help="Dataset to prepare")
    parser.add_argument("--data_split", type=str, default="train", help="Dataset to prepare")
    parser.add_argument("--prompt_data_field", type=str, default=None, help="Field name for prompt data in the dataset")
    parser.add_argument("--code_data_field", type=str, default="code", help="Field name for code data in the dataset")
    
    parser.add_argument("--as_states", action="store_true", help="Prepare data as code states")
    parser.add_argument("--output_path", type=str, default="prepared_data", help="Output directory for prepared data")
    
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process (-1 for all)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--num_edit_paths_per_sample", type=int, default=1, help="Number of edit paths per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    main(args)