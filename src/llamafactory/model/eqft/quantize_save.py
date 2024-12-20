import argparse
import json
import os

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from peft import TaskType, get_peft_model, LoraConfig
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class EQFTConfig:
    eqft_bits: int = field(default=4, metadata={"help": "Quantization bits for EQFT"})
    eqft_iter: int = field(default=1, metadata={"help": "Alternating iterations for EQFT"})


class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer"):
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        # get the parent of the submodule
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)
        # print(sub_module)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    print("You have unwrapped the model. Use it on your own risk.")


def print_model(model, name):
    print("=" * 10 + name + "=" * 10)
    # print(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                print(
                    name,
                    param.shape,
                    param.device,
                    param.dtype,
                    param.requires_grad,
                    param.mean().item(),
                    param.max().item(),
                )
            else:
                print(name, param.shape, param.device, param.dtype, param.requires_grad)


def lora_initialize(lora_model, eqft_config, lora_config):
    from .eqft_utils import eqft_init
    for i, layer in enumerate(lora_model.base_model.model.model.layers):
        # 遍历每个层级中的所有子模块
        for name, module in layer.named_modules():
            # 仅针对目标模块进行处理
            for target in lora_config.target_modules:
                if target in name and isinstance(module, nn.Module):
                    # 如果模块包含 base_layer 属性且 base_layer 是 nn.Linear
                    if hasattr(module, 'base_layer') and isinstance(module.base_layer, nn.Linear):
                        print(f"Processing module: {name}")
                        base_weight = module.base_layer.weight.data  # 获取基础层的权重
                        # 设置初始化的参数
                        kwargs = {
                            "num_bits": eqft_config.eqft_bits,
                            "reduced_rank": lora_config.r,
                            "num_iter": eqft_config.eqft_iter,
                        }
                        # 使用 eqft_init 初始化权重
                        qweight, lora_A, lora_B = eqft_init(base_weight, **kwargs)

                        # 如果模块有 lora_A 和 lora_B，分别初始化它们的权重
                        if hasattr(module, 'lora_A'):
                            module.lora_A['default'].weight = torch.nn.Parameter(lora_A)
                        if hasattr(module, 'lora_B'):
                            module.lora_B['default'].weight = torch.nn.Parameter(lora_B)

                        # 更新基础层的权重
                        module.base_layer.weight.data = qweight
                        # print(f"Updated {name} with quantized weights.")

    return lora_model  # 在所有层级遍历完成后再返回 lora_model


def quantize_and_save(model_args, finetuning_args):
    # Download weights and configure LoRA
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, token=finetuning_args.token, trust_remote_code=True)
    if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            token=finetuning_args.token,
            trust_remote_code=True,
            device_map="auto",
        )
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        # target_modules = finetuning_args.lora_target

    elif any(name in model_args.model_name_or_path.lower() for name in ["bart", "t5"]):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, token=finetuning_args.token)
        task_type = TaskType.SEQ_2_SEQ_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]

    elif any(name in model_args.model_name_or_path.lower() for name in ["deberta", "roberta", "bert"]):
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, token=finetuning_args.token)
        task_type = TaskType.SEQ_CLS
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]  # embeddings not supported by peft
    else:
        raise NotImplementedError("Other models not supported yet.")

    # Config of LoftQ
    eqft_config = EQFTConfig(eqft_bits=finetuning_args.bits, eqft_iter=finetuning_args.iter)

    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=True,
        r=finetuning_args.rank,
        lora_alpha=16 if task_type is TaskType.CAUSAL_LM and finetuning_args.bits == 4 else finetuning_args.rank,
        lora_dropout=0.1,
        target_modules=target_modules,
    )

    # Obtain LoftQ model
    lora_model = get_peft_model(model, lora_config)
    # Perform custom initialization based on the config
    lora_initialize(lora_model, eqft_config, lora_config)
    base_model = lora_model.get_base_model()

    # Save LoftQ model
    model_name = model_args.model_name_or_path.split("/")[-1] + f"-{finetuning_args.bits}bit" + f"-{finetuning_args.rank}rank"
    base_model_dir = os.path.join(finetuning_args.save_dir, model_name)
    lora_model_dir = os.path.join(finetuning_args.save_dir, model_name, "eqft_init")

    lora_model.save_pretrained(lora_model_dir)

    # remove lora adapters and save the backbone
    unwrap_model(base_model)
    base_model.save_pretrained(base_model_dir)
    tokenizer.save_pretrained(base_model_dir)

    # convert safetensor to bin
    tensors = {}
    # with safe_open(os.path.join(lora_model_dir, "adapter_model.safetensors"), framework="pt") as f:
    #     for k in f.keys():
    #         tensors[k] = f.get_tensor(k)
    torch.save(tensors, os.path.join(lora_model_dir, "adapter_model.bin"))

    # change adapter_config.json
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "r") as fp:
        adapter_config = json.load(fp)
        adapter_config['base_model_name_or_path'] = base_model_dir  # This can be a local path or Hub model id
        adapter_config['init_lora_weights'] = True  # Don't apply LoftQ when loading again
        fp.close()
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "w") as fp:
        json.dump(adapter_config, fp, indent=2)

    return base_model_dir, lora_model_dir
