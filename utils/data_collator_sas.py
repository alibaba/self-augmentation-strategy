from numpy.core.numeric import indices
from numpy.lib.twodim_base import mask_indices
import torch
import os 
import numpy as np
from typing import Dict, List, Tuple
from torch._C import dtype
from torch.utils.data.dataset import Dataset
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
from transformers.tokenization_utils_base import BatchEncoding
from .sas_utils import SequenceSideInfo




class SASValidationDataset(Dataset):
    """
    Inherit from xiuxin.data.SASValidationDataset
    Used in train_electra_*.py and model_electra.py only
    """
    def __init__(self, file_path, dataset_size=100000000, tokenizer=None):

        self.examples = np.load(file_path)
        np.random.shuffle(self.examples)
        if dataset_size <= self.examples.shape[0]:
            self.examples = self.examples[:dataset_size].copy()  # Scling creates a view. Use copy to release the memory of the original big nparray (See https://github.com/numpy/numpy/issues/15746)

        self.tokenizer = tokenizer

        self.cls_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.sep_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)

        # The current dataset doesn't include CLS at all. It is merely sequence of 128 tokens. Replace the frist one as [CLS]
        if self.examples[0, 0] != self.cls_index: 
            self.examples = np.roll(self.examples, shift=1, axis=1)
            self.examples[:, 0] = self.cls_index

        self.vocab_size = self.tokenizer.vocab_size
        self.vocab_dtype = (torch.short if self.vocab_size <= 32768 else torch.long)

        self.dataset_size = self.examples.shape[0]
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return {'idx': i, 'inputids': self.examples[i]}


class SASTrainDataset(Dataset):
    """
    Inherit from SASValidationDataset above. 
    Used in model_electra.ElectraAdvanced setting only.
    Support incremental schedule. This class handle the buffer.
    """

    def __init__(self, file_path, dataset_size=100000000, tokenizer=None, seed=None):

        np.random.seed(seed)
        self.examples = np.load(file_path)
        np.random.shuffle(self.examples)

        if dataset_size <= self.examples.shape[0]:
            self.examples = self.examples[:dataset_size].copy()  # Scling creates a view. Use copy to release the memory of the original big nparray (See https://github.com/numpy/numpy/issues/15746)

        self.dataset_size = self.examples.shape[0]

        self.tokenizer = tokenizer

        self.cls_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.sep_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        # The current dataset doesn't include CLS at all. It is merely sequence of 128 tokens. Replace the frist one as [CLS]
        if self.examples[0, 0] != self.cls_index:
            self.examples = np.roll(self.examples, shift=1, axis=1)
            self.examples[:, 0] = self.cls_index

        # Used vocab_dtype to reduce memory by using short int in case of small vocab dataset_size
        self.vocab_size = self.tokenizer.vocab_size
        self.vocab_dtype = (torch.short if self.vocab_size <= 32768 else torch.long)

        # Used to store data augmentations
        self.buffer_da = None
        self.buffer_da_flag = None

    def __len__(self):
        return len(self.examples)

    def update_buffer_da(self, idx, flag_freeze_generation=False):
        '''
        Pop the first element in the buffer. 
        Typically used in the beginning of the forward pass which need fake_tokens. 
        It cannot be used in data_collator! Data_collator have multi-threads and cannot modify variable in dataset.
        '''
        # shift dimension -2 (augmentations per instance, which is used when freezing generator) by 1, in order to use the different augmentations in different epochs.
        if self.buffer_da is not None and self.buffer_da.shape[-2] > 1:
            tmp = torch.roll(self.buffer_da[idx, ...], 1, -2)
            self.buffer_da[idx, ...] = tmp
            del tmp     # Jingqiao's note: Shanzhu tested that the roll function doesn't release memory right away in cpu, althoug it does in gpu

    def __getitem__(self, i):
        # Read-only. Used by data_collator.

        if self.buffer_da is None:
            return {'idx': i, 'input_ids': self.examples[i], 'data_augmentation': None}
        else:
            # -2 dimension: return one augmentation Per Instance: Only reture one (always return the last one after rolling augmentations each time)
            # -1 dimension: return all augmentations Per Mixing (to be mixed in data collator): Always return all valid ones
            da_flag = self.buffer_da_flag[i, :]
            if da_flag.sum() == 0:  # DA is not prepared yet
                return {'idx': i, 'input_ids': self.examples[i], 'data_augmentation': None}
            else:
                return {'idx': i, 'input_ids': self.examples[i], 'data_augmentation': self.buffer_da[i, :, -1, da_flag]}

    def save_buffer_da(self, idx, buf, aug_positions=None, augPerMixing=1):
        '''
        Save fake tokens into the buffer. 
        Typically used in the forward pass after calling `update_buffer_da` to clean the buffer. 
        It cannot be used in data_collator! Data_collator have multi-threads and cannot modify variable in dataset.
        '''

        dtype = self.vocab_dtype
        # Create empty buffer all in once, instead of incrementally increase its dataset_size after each mini-batch. 
        # This is to avoid fragmented memory allocation and test possible memory shortage at the earliest time.
        if self.buffer_da is None:
            # shape: dataset_size x seq_len x count_augPerInstance x count_augPerMixing
            count_augPerInstance = buf.shape[1]     # Number of augmentations per instance (used when freezing the generator, and using one augmentation per subsequent epoch)
            count_augPerMixing = augPerMixing       # Number of historical distributions to be mixed together. E.g., we might want to mix the distributions predicted in recent two epochs for each instance
            self.buffer_da = -100 * torch.ones([self.dataset_size, aug_positions.shape[1], count_augPerInstance, count_augPerMixing], dtype=dtype)     # Use sys.getsizeof(self.buffer_da.storage()) to measure its memory usage. In case of nparray, use self.buffer_da.__i
            self.buffer_da_flag = torch.zeros(self.dataset_size, count_augPerMixing, dtype=torch.bool)

        # Shape = batch_size x seq_len x count_augPerInstance
        token_to_save = -100 * torch.ones((idx.shape[0], aug_positions.shape[1], buf.shape[1]), dtype=dtype)
        token_to_save[aug_positions] = buf.type(dtype).cpu()                    # Convert to cpu after changing its type from long to short (save time to convert from gpu to cpu)
        if self.buffer_da.shape[-1] >= 2: 
            # In case of augPerMixing>=2 (i.e., abs(opt.augMixing)=1), it means to mix teacher's outputs in previous epochs. Shift to remove old ones before adding new one to the end
            self.buffer_da[idx, ..., 0:-1] = self.buffer_da[idx, ..., 1:]       # Shift old distributions to the left
            self.buffer_da_flag[idx, 0:-1] = self.buffer_da_flag[idx, 1:]
        self.buffer_da[idx, ..., -1] = token_to_save                            # Now, add the new one to the end of the dimension
        self.buffer_da_flag[idx, -1] = True


class DataCollatorForSasPretraining:
    """
    Data collator used for SAS.
    - Supported cold_start_augumentation_method : mlm / unigram / random / generator / mask
    - This collator support incremental schedule if accompany with `SASDataset`. 
    - It also act as in incremental schedule stage 1 of accompany with `SASValidationDataset`.

    - Output returns different combination according to current stage and cold_start_augumentation_method setting. 
    - The stage is determined by input tuple fed by `SASDataset`.
    - Every setting return `input_ids` for forward pass and `batch_idx` for cold_start_augumentation_method storage.
    - Return "mlm_labels" to do MLM task and/or "rtd_labels" to do RTD task. 

    - Stage 1 
        - How to determine the stage?
            - For `SASValidationDataset`, the input is a list of list. It always stay in stage 1.
            - For `SASDataset`, the input is a list of tuple with dataset_size 4. If sedond value
                                    in the tuple is None, it means stage 1 occured. 
        - Output list: 
            - If cold_start_augumentation_method == `mlm` or `generator`, return `mlm_labels` only. 
            - Else if the input is a list of tuple, return both mlm_labels.
            - Else if self.mlm is True, return both mlm_labels.
            - Else, return `rtd_labels` only. 
    - Stage 2 
        - How to determine the stage? 
            - Only `SASDataset` with a list of tuple input and 4 value in tuple is not None 
                                    will activate the stage 2.
        - Output list: 
            - If `need_new_fake_token` (4th value in tuple) is True, return both mlm_labels. 
            - Else, return `rtd_labels` only. It is because that SASDataset have more fake_tokens
                                    in buffer. We do not need to calculate new fake tokens. 
    """

    def __init__(self, tokenizer, empirical_distribution_file = None,
                 cold_start_augumentation_method = 'unigram', 
                 mlm_probability = 0.15, 
                 whole_word_masking=0,
                 dynamic_masking=0,
                 dataset_sizes=None):

        self.tokenizer = tokenizer
        self.cls_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.sep_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)

        self.cold_start_augumentation_method = cold_start_augumentation_method
        self.mlm_probability = mlm_probability
        self.whole_word_masking = whole_word_masking
        self.dynamic_masking = dynamic_masking
        self.dataset_sizes = dataset_sizes

        # We found that in first 1000 step, unigram distribution is too hard for discriminator. We modify its temperature to do warmup. 
        if self.cold_start_augumentation_method == "unigram":  
            empirical_distribution = np.load(empirical_distribution_file) if os.path.exists(empirical_distribution_file) else None
            self.empirical_distribution = torch.Tensor(empirical_distribution / empirical_distribution.sum())

            empirical_distribution_warmup = np.power(empirical_distribution, 1/2) # avoid divide zero warning
            self.empirical_distribution_warmup = torch.Tensor(empirical_distribution_warmup / empirical_distribution_warmup.sum())
            self.unigram_warmup_step = 1000 # Note: it is initialized to 1000 at the beginning of every single epoch. Fix it by setting it to 0 after 1st epoch. See related fix in trainer_advanced.py

        else:
            self.unigram_warmup_step = 0

        self.training = None
        self.epoch_train = -1
        self.epoch_eval = -1

        if self.whole_word_masking:
            self.sequence_side_info = SequenceSideInfo()

        if self.tokenizer.mask_token is None:
            raise ValueError("This tokenizer does not have a mask token which is necessary for masked language modeling.")

    # Whole word masking - Simplified code based on https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L342
    def create_masked_lm_predictions(self, tokens, mlm_probability, whole_word_masking=False, aug_per_step=1):
        """Creates the predictions for the masked LM objective."""
        tokens_length = tokens.shape[0]  # remember the length of the original sequence with end padding
        tokens = tokens[tokens != self.tokenizer.pad_token_id] # added to omit the ending padding tokens

        tokens = tokens.numpy()
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token in [self.cls_index, self.sep_index]:
                continue
            # Whenever we see the ## token, we append it to the previous set of word indexes.
            if (whole_word_masking and len(cand_indexes) >= 1 and token in self.sequence_side_info.ind_subtokens):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        np.random.shuffle(cand_indexes)

        num_to_mask = max(1, int(round(len(tokens) * mlm_probability)))
        masked_indices = torch.zeros((aug_per_step, tokens_length), dtype=torch.bool)

        set_checked = 0
        for aug_count in range(aug_per_step):
            num_masked = 0
            covered_indexes = list()
            for i in range(set_checked, len(cand_indexes)):
                index_set = cand_indexes[i]

                if num_masked >= num_to_mask:
                    break
                if num_masked + len(index_set) > num_to_mask:
                    continue

                covered_indexes += index_set
                num_masked += len(index_set)

            set_checked = i
            masked_indices[aug_count, covered_indexes] = True

        return masked_indices

    def generate_masked_indices(self, inputs, inputs_ori):
        # Sample RTD / Masked position. By default, mask and fake token are the same.

        if self.whole_word_masking == 1:
            masked_indices = torch.stack([self.create_masked_lm_predictions(
                inputs_ori[r, :], 
                mlm_probability=self.mlm_probability, 
                whole_word_masking=self.whole_word_masking, 
            ) for r in range(inputs_ori.shape[0])]).view(-1, inputs_ori.shape[1])

        else:
            probability_matrix = torch.full(inputs.shape, self.mlm_probability)
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()]
            if self.tokenizer._pad_token is not None:
                padding_mask = inputs.eq(self.tokenizer.pad_token_id)
                probability_matrix.masked_fill_(padding_mask, value=0.0)
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

            masked_indices = torch.bernoulli(probability_matrix).bool()

        return masked_indices
    
    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        assert type(examples[0]) is dict
        self.training = (True if 'data_augmentation' in examples[0] else False)

        vocab_dtype = torch.tensor(examples[0]['input_ids']).dtype
        inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(a['input_ids']) for a in examples], batch_first=True, padding_value=self.tokenizer.pad_token_id).type(vocab_dtype)
        inputs_ori = inputs.clone()
        idx = torch.LongTensor([a['idx'] for a in examples])    # We use it as index - pytorch required dtype=long when using an integer as index

        dynamic_masked_indices = None; dynamic_masked_labels = None
        if self.training:  # Include self_trained data augmentation

            # For "incremental" setting, the examples will return tuple-3 and the 3rd one is the data augmentation. 
            # For the first epoch, tuple[1] is None. 

            none_count = np.array([a['data_augmentation'] is None for a in examples]).sum()
            if none_count == 0:     # This means buffer is filled with data augmentation from previous epochs. Use it!

                # If second element in tuple is None, it means the first epoch will not do incremental RTD but original.
                buffer_da = torch.stack([a['data_augmentation'] for a in examples], dim=0)
                rtd_token = buffer_da[..., -1]

                # To enforce rtd_labels is correct, because fake token could happen to the original token (~1% in case of unigram distribution)
                rtd_labels = torch.logical_and(rtd_token >= 0, rtd_token != inputs_ori)                

                # mlm label: incldue the original tokens at the positions to do mlm prediction
                mlm_labels = inputs_ori.clone()
                mlm_labels[~rtd_labels] = -100

                # Training data inputs after applying data augmentation
                inputs = torch.where(rtd_labels, rtd_token, inputs_ori)

                return {"input_ids": inputs.long(), "mlm_labels": mlm_labels.long(), "rtd_labels": rtd_labels.long(), 
                        "side_info_sets": {"batch_idx": idx.long(), "dataset_sizes": self.dataset_sizes, "dynamic_masked_indices": dynamic_masked_indices, "dynamic_masked_labels": dynamic_masked_labels}}

            else:
                assert none_count == len(examples), ("! none_count error found. %d (%d~%d)" % (none_count, examples[0]['idx'], examples[-1]['idx']))

        masked_indices = self.generate_masked_indices(inputs, inputs_ori)

        n_sample = int(masked_indices.sum())
        mlm_labels = inputs_ori.clone()
        mlm_labels[~masked_indices] = -100

        if self.cold_start_augumentation_method == "mlm":
            # Original BERT's MLM input : inputs and mlm_labels

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(mlm_labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(mlm_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), mlm_labels.shape, dtype=vocab_dtype)
            inputs[indices_random] = random_words[indices_random]

            return {"input_ids": inputs.long(), "mlm_labels": mlm_labels.long(), 
                    "side_info_sets": {"batch_idx": idx.long(), "dataset_sizes": self.dataset_sizes}}

        elif self.cold_start_augumentation_method == "mask":
            inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        elif self.cold_start_augumentation_method == "unigram":
            if self.unigram_warmup_step > 0:
                self.unigram_warmup_step -= 1
                dis_to_sample = self.empirical_distribution_warmup
            else:
                dis_to_sample = self.empirical_distribution
            inputs[masked_indices] = torch.multinomial(dis_to_sample, num_samples=n_sample, replacement=True).type(vocab_dtype)

        elif self.cold_start_augumentation_method == "random":
            inputs[masked_indices] = torch.randint(len(self.tokenizer), (n_sample, ), dtype=vocab_dtype)

        rtd_labels = torch.logical_and(masked_indices, inputs != mlm_labels) * 1

        return {"input_ids": inputs.long(), "mlm_labels": mlm_labels.long(), "rtd_labels": rtd_labels.long(), 
                "side_info_sets": {"batch_idx": idx.long(), "dataset_sizes": self.dataset_sizes, "dynamic_masked_indices": dynamic_masked_indices, "dynamic_masked_labels": dynamic_masked_labels}}


        
