from transformers import DefaultFlowCallback, TrainerState, TrainerControl
from transformers.integrations import WandbCallback, TensorBoardCallback
from transformers import TrainingArguments
import os
import torch
from transformers.utils import logging
logger = logging.get_logger(__name__)

class SasTrainerCallback(DefaultFlowCallback):
    def __init__(self):
        self.state = None

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.state = state

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control = super().on_step_end(args, state, control, **kwargs)
        if state.global_step <= 500 and (state.global_step - 1) % 10 == 0 and args.logging_first_step:
            control.should_log = True
        
        return control

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_train_begin(TrainingArguments, state, control, **kwargs)

        model_path = getattr(kwargs['model'], 'model_name_or_path', '')
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            if not args.ignore_data_skip:
                kwargs['model'].dataset.buffer_da = torch.load(os.path.join(model_path, "data_augmentation.pt"))
                kwargs['model'].dataset.buffer_da_flag = torch.load(os.path.join(model_path, "data_augmentation_flag.pt"))

                logger.info("  Continuing training from checkpoint and load data augmentation")

        return control


    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs['train_dataloader'].collate_fn.epoch_train = state.epoch
        if state.epoch >= 1:  # Disable the warmup period of data augmentation
            kwargs['train_dataloader'].collate_fn.unigram_warmup_step = 0

        control.should_log = True  # first step of each epoch
        return control


class SasWandbCallback(WandbCallback):
    def __init__(self):
        super().__init__()
        self.log = dict()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model, **kwargs)
        if self._wandb is not None:
            self._wandb.run.name = args.output_dir.split("/")[-1][0:60]
            self._wandb.run.notes = args.output_dir.split("/")[-1]


    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model, reinit=False)
        if state.is_world_process_zero:
            self._wandb.log({**logs, **self.log}, step=state.global_step)


class SasTensorBoardCallback(TensorBoardCallback):
    def __init__(self):
        super().__init__()
        self.log = dict()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        logs = {**logs, **self.log}
        super().on_log(args, state, control, logs, **kwargs)

