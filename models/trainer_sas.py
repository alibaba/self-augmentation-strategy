from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass, field
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
import os
import torch
import math
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import Trainer, TrainingArguments

@dataclass
class SasTrainingArguments(TrainingArguments):
    # Support to set up warmup steps as a ratio
    # original warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    warmup_steps: float = field(default=0.1, metadata={"help": "Linear warmup over warmup_steps."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "gradient accumalation steps."})
    do_predict_train_data: bool = field(default=False, metadata={"help": "Whether to run predictions on the train set."})    


class SasTrainerForFinetune(Trainer):
    # Added to support the usage of warm-up ratio (between 0 and 1), instead of warmup steps (integer values) as requried by the run_glue.py
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if 0 < self.args.warmup_steps < 1:
            self.args.warmup_steps = int(self.args.warmup_steps * num_training_steps)

        super().create_optimizer_and_scheduler(num_training_steps)

    def _load_optimizer_and_scheduler(self, model_path):
        """If optimizer and scheduler states exist, load them."""
        if model_path is None:
            return

        if os.path.isfile(os.path.join(model_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(model_path, "scheduler.pt")
        ):
            # SAS-specific: Don't resume from the optiizer status of pretraining
            pass

        if self.deepspeed:
            # Not sure how to check if there is a saved deepspeed checkpoint, but since it just return None if it fails to find a deepspeed checkpoint this is sort of a check-n-load function
            self.deepspeed.load_checkpoint(model_path, load_optimizer_states=True, load_lr_scheduler_states=True)

class SasTrainerForPretrain(Trainer):
    # In Huggingface Transformers 4.3, the default Trainer doesn't support to disable random sampler (random huffle) via training_arg
    # As a temporary solution, overwrite the get_train_dataloader in Trainer
    def get_train_dataloader(self) -> DataLoader:
        train_sampler = (
                SequentialSampler(self.train_dataset)
                if self.args.local_rank == -1
                else SequentialDistributedSampler(self.train_dataset)
        ) 

        # print(train_sampler)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        # Save model checkpoint
        super()._save_checkpoint(model, trial, metrics)

        if self.model.save_data_augmentation > 0 and self.is_world_process_zero():
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

            torch.save(self.model.dataset.buffer_da, os.path.join(output_dir, "data_augmentation.pt"))
            torch.save(self.model.dataset.buffer_da_flag, os.path.join(output_dir, "data_augmentation_flag.pt"))

class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
