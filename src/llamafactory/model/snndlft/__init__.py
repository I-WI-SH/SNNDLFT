# Copyright 2023-present the HuggingFace Inc. team.
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

from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from .config import AdaLoraConfig

from .model import SNNAdaLoraModel
from .peft_model import(
    SNNPeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
from .peft import ada_get_peft_model

# __all__ = ["AdaLoraConfig", "AdaLoraLayer", "AdaLoraModel", "SVDLinear", "RankAllocator", "SVDQuantLinear", "get_peft_model"]


def __getattr__(name):
    if (name == "SVDLinear8bitLt") and is_bnb_available():
        from .bnb import SVDLinear8bitLt

        return SVDLinear8bitLt

    if (name == "SVDLinear4bit") and is_bnb_4bit_available():
        from .bnb import SVDLinear4bit

        return SVDLinear4bit

    raise AttributeError(f"module {__name__} has no attribute {name}")
