# Learning to Code with Compute-Scaled Experts

## Installation

```
git clone https://github.com/lintangsutawika/code_cse.git
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
    --output_path ${DATA_PATH}$ \
    --num_workers 100 \
    --num_samples 100 \
    --num_edit_paths_per_sample 5 \
    --as_states
```

## Initial Policy

The policy we want to improve is one finetuned to generate code edits.
We can do this by finetuning an initial base policy with OpenRLHF.

```
torchrun --nproc_per_node 8 --master_port 8291 \
    --module openrlhf.cli.train_sft \
        --save_path ${MODEL_SAVE_PATH} \
        --ckpt_path ${TEMP_SAVE_PATH} \
        --load_checkpoint \
        --save_steps 100 \
        --logging_steps 1 \
        --eval_steps -1 \
        --train_batch_size 512 \
        --micro_train_batch_size 32 \
        --pretrain ${BASE_MODEL} \
        --save_hf_ckpt \
        --bf16 \
        --max_epochs 10 \
        --max_samples 128000 \
        --max_len 2048 \
        --zero_stage 3 \
        --learning_rate 1e-5 \
        --lr_warmup_ratio 0.001 \
        --dataset json@${DATA_PATH}/ \
        --input_key instruction \
        --output_key edit \
        --input_template "{}" \
        --gradient_checkpointing \
        --overlap_comm \
        --adam_offload \
        --flash_attn \
        --packing_samples \
        --grad_accum_dtype bf16 \
        --use_liger_kernel
```

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
