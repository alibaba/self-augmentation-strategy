# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import sys
sys.path.append(".")

from utils.data_collator_sas import DataCollatorForSasPretraining, SASTrainDataset, SASValidationDataset
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import argparse

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    # Trainer,  # Repalce it by SasTrainerForPretrain. See below.
    TrainingArguments,
    DefaultFlowCallback,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers.integrations import WandbCallback, TensorBoardCallback

from models.trainer_sas import SasTrainerForPretrain as Trainer
from models.modeling_sas import SasForPreTraining
from utils.configuration_sas import SasConfig

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    import torch.cuda.amp as amp
    _has_apex = True
    print("\nApex is available.\n")
except ImportError:
    _has_apex = False
    print("\nApex is not available.\n")

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def parse_config():

    parser = argparse.ArgumentParser()

    # training hyper-parameters
    parser.add_argument('-bert_setting', type=str, default="SAS_small")
    parser.add_argument('-num_epoch', type=int, default=5)
    parser.add_argument('-max_steps', type=int, default=250_000)
    parser.add_argument('-warmup_steps', type=float, default=10000)
    parser.add_argument('-batch_size', type=int, default=8, help='')  # 128 in case of Electra
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.00175, help='')    # 5e-4 in case of Electra
    parser.add_argument('-weight_decay', type=float, default=0.01)
    parser.add_argument('-correct_bias', type=int, default=1)
    parser.add_argument('-adam_epsilon', type=float, default=1e-6)
    parser.add_argument('-adam_beta1', type=float, default=0.9)
    parser.add_argument('-adam_beta2', type=float, default=0.999)

    # SAS related
    parser.add_argument('-model_type', type=str, default="sas")
    parser.add_argument('-model_name_or_path', type=str, default=None)
    parser.add_argument('-overwrite_pretrained_config', type=int, default=0)
    parser.add_argument('-cold_start_epochs', type=float,default=1.0, help='# of epochs for cold start')
    parser.add_argument('-cold_start_augumentation_method', type=str, default="unigram", help='mlm/unigram/random')
    parser.add_argument('-mlm_probability', type=float, default=0.15, help='0-1')
    parser.add_argument('-whole_word_masking', type=int, default=0, help='')

    parser.add_argument('-position_embedding_type', type=str, default='absolute', help='absolute;absolute_self_only;relative_key;relative_key_query')
    parser.add_argument('-max_position_embeddings', type=int, default=128, help='Used in absolute and relative position embeddings')

    parser.add_argument('-relative_position_embedding', type=int, default=0, help='0: absolute only; < 0: relative only; >0: both')
    parser.add_argument('-absolute_position_embedding', type=int, default=1, help='1: BERT default position embedding at input layer')

    parser.add_argument('-augmentation_temperature', type=float, default=1.0)
    parser.add_argument('-augmentation_copies', type=int, default=1)
    parser.add_argument('-gen_weight', type=float, default=1)
    parser.add_argument('-dis_weight', type=str, default='50-50',
                        help='It should be a string that includes either a single number such as "50" to indicate a flat value, ' +
                        'or two numbers such as "50-100" to indicate the startng end ending value')   # Stating and ending weight. Will be convert to an array 
    parser.add_argument('-dis_weight_scheduler', type=int, default=4, help='0: Flat; 1: Linear increasing; 2: Epoch-wise Step Increasing; 4: Two-stages')
    parser.add_argument('-dynamic_masking', type=int, default=0, help='0: Static masking; 1: Dynamic masking')

    # Logistic related
    parser.add_argument('-seed', type=int, default=50, help='')
    parser.add_argument('-tokenizer_name', type=str, default="google/electra-small-generator")
    parser.add_argument('-use_fast_tokenizer', type=bool, default=True)  # Our current data was generated with using fast tokenizer. This setting will be automatically applied in fine-tuning.
    parser.add_argument('-dataset', type=str, default="1M")
    parser.add_argument('-dataset_eval', type=str, default="eval")
    parser.add_argument('-data_size', type=int, default=100000000)
    parser.add_argument('-eval_data_size', type=int, default=51200)
    parser.add_argument('-eval_steps', type=int, default=4000, help='')
    parser.add_argument('-logging_steps', type=int, default=200, help='')
    parser.add_argument('-dataloader_num_workers', type=int, default=4, help='')
    parser.add_argument('-save_total_limit', type=int, default=2, help='')
    parser.add_argument('-save_data_augmentation', type=int, default=0, help='0 or 1: Save data augmentation together with checkpoints or not')
    parser.add_argument('-save_steps', type=int, default=50000, help='')
    parser.add_argument('--local_rank', type=int, default=-1, help='')

    parser.add_argument('-debug_config', type=dict, default=None, help='')
    parser.add_argument('-debugExtraMetrics', type=int, default=1, help='Show extra metrics or not (some of these metrics require extra notable calculation)')
    parser.add_argument('-debugMemStatsInterval', type=int, default=1000, help='Show memory stats or not')
    parser.add_argument('-debugGradOverflowInterval', type=int, default=100, help='Show gradient relatd data or not')
    parser.add_argument('-debugActivationInterval', type=int, default=100_000_000, help='Show activatiom relatd data or not')
    parser.add_argument('-debugMultiTasksConflictInterval', type=int, default=1000, help='Show gradient analysis between RTD vs. MLM ')

    parser.add_argument('-has_apex', type=str, default=_has_apex, help='has apex or not')
    parser.add_argument('-fp16', type=str, default="O2", help='O1/O2')
    parser.add_argument('-cuda', type=str, default="0", help='')
    parser.add_argument('-option', type=str, default="0", help='')

    parser.add_argument('-pretrain_path', type=str, default=None, help='')
    parser.add_argument('-do_train', type=bool, default=True, help='')
    parser.add_argument('-do_eval', type=bool, default=False, help='')
    parser.add_argument('-start_from_checkpoint', type=bool, default=False, help='')
    parser.add_argument('-data_path', type=str, default="../datasets/")
    parser.add_argument('-output_dir', type=str, default="default")

    return parser.parse_args()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    opt = parse_config()
    print(opt)
    opt.debug_config = {
        'logging_steps': opt.logging_steps,
        'debugExtraMetrics': opt.debugExtraMetrics,
        'debugMemStatsInterval': opt.debugMemStatsInterval,
        'debugGradOverflowInterval': opt.debugGradOverflowInterval,
        'debugActivationInterval': opt.debugActivationInterval,
        'debugMultiTasksConflictInterval': opt.debugMultiTasksConflictInterval,
    }
    if len(opt.cuda) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda) if opt.cuda != "-1" else ""

    opt.model_name_or_path = None if opt.model_name_or_path == 'None' else opt.model_name_or_path

    model_args = ModelArguments(tokenizer_name=opt.tokenizer_name, 
        use_fast_tokenizer=opt.use_fast_tokenizer, 
        model_type=opt.model_type,
        model_name_or_path=opt.model_name_or_path  # support restore checkpoint and continue training
    )

    training_args = TrainingArguments(
        output_dir="../output/%s" % opt.output_dir,
        logging_dir="../output/tb/%s" % opt.output_dir,
        overwrite_output_dir=True,
        do_train=opt.do_train,
        do_eval=opt.do_eval,
        num_train_epochs=-1,
        max_steps=opt.max_steps,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        logging_first_step=True,
        logging_steps=opt.logging_steps,  # 20,
        save_steps=opt.save_steps,  # 50000,
        save_total_limit=opt.save_total_limit,  # 2
        eval_steps=opt.eval_steps,  # 4000,
        warmup_steps=opt.warmup_steps,
        weight_decay=opt.weight_decay,
        adam_epsilon=opt.adam_epsilon,
        per_device_train_batch_size=opt.batch_size,
        per_device_eval_batch_size=opt.batch_size,
        learning_rate=opt.lr,
        fp16=_has_apex and (opt.fp16 != "None"),
        fp16_opt_level=opt.fp16,
        dataloader_num_workers=opt.dataloader_num_workers,
        seed=opt.seed,
        local_rank=opt.local_rank,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=model_args.use_fast_tokenizer)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    tokenized_datasets = dict()
    tokenized_datasets["train"] = SASTrainDataset(
        file_path=opt.data_path + "wiki_corpus_%s.npy" % opt.dataset,
        dataset_size=opt.data_size,
        tokenizer=tokenizer,   
        seed=opt.seed, 
    )
    tokenized_datasets["validation"] = SASValidationDataset(
        file_path=opt.data_path + "wiki_corpus_%s.npy" % opt.dataset_eval,
        dataset_size=opt.eval_data_size,
        tokenizer=tokenizer,
    )
    # the "data_collator" class implements how data is preprocessed for the language model and pre-training tasks.
    data_collator = DataCollatorForSasPretraining(
        tokenizer=tokenizer, 
        cold_start_augumentation_method=opt.cold_start_augumentation_method,
        mlm_probability=opt.mlm_probability, 
        whole_word_masking=opt.whole_word_masking,
        dynamic_masking=opt.dynamic_masking,
        empirical_distribution_file=opt.data_path + "unigram.npy",
        dataset_sizes={'train': len(tokenized_datasets["train"]), 'eval': len(tokenized_datasets["validation"])}
    )

    # Load pretrained model and tokenizer
    if model_args.model_name_or_path and not opt.overwrite_pretrained_config:
        config = SasConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = SasConfig()
        logger.warning("You are instantiating a new config instance from scratch.")

        bert_size = {
            # Electra-version small: [4, 12, 256, 1024],; BERT-24checkpoitns version small: [8, 4, 512, 2048]
            "SAS_small": [4, 12, 256, 1024],
            "SAS_base": [12, 12, 768, 3072],
            "SAS_large": [16, 24, 1024, 4096]
        }

        config = SasConfig(
            num_attention_heads=bert_size[opt.bert_setting][0],
            num_hidden_layers=bert_size[opt.bert_setting][1],
            hidden_size=bert_size[opt.bert_setting][2],
            embedding_size=bert_size[opt.bert_setting][2],
            intermediate_size=bert_size[opt.bert_setting][3],
            tie_word_embeddings=True,
            position_embedding_type=opt.position_embedding_type,
            max_position_embeddings=opt.max_position_embeddings,
 
            # Below is special parameter for SAS
            gen_weight=opt.gen_weight,
            dis_weight=opt.dis_weight,
            dis_weight_scheduler=opt.dis_weight_scheduler,
            dynamic_masking=opt.dynamic_masking,      

            absolute_position_embedding=opt.absolute_position_embedding,
            relative_position_embedding=opt.relative_position_embedding,

            cold_start_epochs=opt.cold_start_epochs,
            debug_config=opt.debug_config
        )

    if model_args.model_name_or_path:
        model = SasForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = SasForPreTraining(config)
    model.resize_token_embeddings(len(tokenizer))
    model.dataset = tokenized_datasets["train"]
    model.model_name_or_path = model_args.model_name_or_path
    model.save_data_augmentation = opt.save_data_augmentation
    print(model)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        callbacks=[model.trainer_callback, model.wandb_callback, model.tensorboard_callback],
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    for cb in [cb for cb in trainer.callback_handler.callbacks if type(cb) in [DefaultFlowCallback, WandbCallback, TensorBoardCallback]]:
        trainer.remove_callback(cb)

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
