import os
import fire

from transformers.trainer import get_scheduler
from openrlhf.datasets import SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import SFTTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

def train_dagger(
    base_expert_model: str,
    base_policy_model: str,
    max_iterations: int = 10,
    beta: float = 0.5,
):
    for iteration in range(0, max_iterations):

        if get_trajectories(
            policy_model=base_policy_model,
            expert_model=base_expert_model,
            iteration=iteration,
        ):
            # Process the trajectories
            # idx = 0
            # base_code = get_code_snippet(d[idx]['step'][0]['full_input'])
            # predicted = d[idx]['step'][0]['aux']['predicted']
            # x1 = apply_patch(predicted, base_code)
            # x2 = d[idx]['answer'][0].split("\n")
            # get_diff(x1, x2)
            pass
        else:
            break

        trainer = SFTTrainer(
            model=model,
            strategy=strategy,
            optim=optim,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            scheduler=scheduler,
            max_norm=args.max_norm,
            pretrain_mode=args.pretrain_mode,
            batch_size=args.train_batch_size,
            max_epochs=args.max_epochs,
            tokenizer=tokenizer,
            save_hf_ckpt=args.save_hf_ckpt,
            disable_ds_ckpt=args.disable_ds_ckpt,
        )

        trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

        # save model checkpoint after fitting on only rank0
        strategy.save_model(model, tokenizer, args.save_path)

        # Evaluate the model. 
        # Train to recover from current state
        # Ablations: SFT, SFT+DPO
        # How would a model track error recovery?
        # Hypothesis: Learn to recognize error based on expert's trajectory.
        # Patches from fully generated code.



if __name__ == "__main__":
    fire.Fire(train_dagger)