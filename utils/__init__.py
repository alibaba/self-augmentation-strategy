# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from transformers.file_utils import _BaseLazyModule, is_tf_available, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_sas": ["SAS_PRETRAINED_CONFIG_ARCHIVE_MAP", "SasConfig"],
    "tokenization_sas": ["SasTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_sas_fast"] = ["SasTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_sas"] = [
        "SAS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SasForMaskedLM",
        "SasForMultipleChoice",
        "SasForPreTraining",
        "SasForQuestionAnswering",
        "SasForSequenceClassification",
        "SasForTokenClassification",
        "SasModel",
        "SasPreTrainedModel",
        "load_tf_weights_in_sas",
    ]

if is_tf_available():
    _import_structure["modeling_tf_sas"] = [
        "TF_SAS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSasForMaskedLM",
        "TFSasForMultipleChoice",
        "TFSasForPreTraining",
        "TFSasForQuestionAnswering",
        "TFSasForSequenceClassification",
        "TFSasForTokenClassification",
        "TFSasModel",
        "TFSasPreTrainedModel",
    ]


if TYPE_CHECKING:
    from utils.configuration_sas import SAS_PRETRAINED_CONFIG_ARCHIVE_MAP, SasConfig
    from models.tokenization_sas import SasTokenizer

    if is_tokenizers_available():
        from models.tokenization_sas_fast import SasTokenizerFast

    if is_torch_available():
        from models.modeling_sas import (
            SAS_PRETRAINED_MODEL_ARCHIVE_LIST,
            SasForMaskedLM,
            SasForMultipleChoice,
            SasForPreTraining,
            SasForQuestionAnswering,
            SasForSequenceClassification,
            SasForTokenClassification,
            SasModel,
            SasPreTrainedModel,
            load_tf_weights_in_sas,
        )

    if is_tf_available():
        from .modeling_tf_sas import (
            TF_SAS_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSasForMaskedLM,
            TFSasForMultipleChoice,
            TFSasForPreTraining,
            TFSasForQuestionAnswering,
            TFSasForSequenceClassification,
            TFSasForTokenClassification,
            TFSasModel,
            TFSasPreTrainedModel,
        )

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
