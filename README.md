# Learning to Code with Compute-Scaled Experts

## Installation

```
cd code_cse/
pip install -e .
```

## Data Preperation

Original training data
- ise-uiuc/Magicoder-OSS-Instruct-75K 
    - problem, solution
- bigcode/self-oss-instruct-sc2-exec-filter-50k
    - instruction, response

```
python -m code_cse.data \
    --data_path bigcode/self-oss-instruct-sc2-exec-filter-50k \
    --prompt_data_field instruction \
    --code_data_field response \
    --output_path train_data/ \
    --num_workers 100 \
    --num_samples 100 \
    --num_edit_paths_per_sample 5 \
    --as_states
```

## Initial Policy

The policy we want to improve is one finetuned to generate code edits.

## Training

Note: should change mbpp as base task since it's ideally what we want to evaluate performance on.

```
python -m code_cse.main \
    --base_expert_model ${EXPERT_MODEL} \
    --base_policy_model ${POLICY_MODEL} \
    --base_task mbpp \
    --output_trajectory_path output/ \
    --task_path code_cse/tasks/
```
