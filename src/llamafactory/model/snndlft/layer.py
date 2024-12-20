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

import warnings
from typing import Any, List, Optional

import packaging
import torch
import transformers
from torch import nn
import numpy as np
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
from spikingjelly.activation_based import neuron,functional, encoding

if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
    from transformers.integrations import deepspeed_config
else:
    from transformers.deepspeed import deepspeed_config




class SNNAdaLoraLayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    # Note: ranknum doesn't need to be included as it is not an nn.Module
    adapter_layer_names = ("lora_A", "lora_B", "lora_E", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "ranknum")

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.lora_E = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.ranknum = nn.ParameterDict({})
        self.firing_rate = nn.ParameterDict({})
        self.spiking_neuron = nn.ModuleDict({})

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r < 0:
            # note: r == 0 is allowed for AdaLora, see #1539
            raise ValueError(f"`r` should be a positive integer or 0, but the value passed is {r}")
        # import pdb
        # pdb.set_trace()
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        # Right singular vectors
        self.lora_A[adapter_name] = nn.Parameter(torch.randn(r, self.in_features))
        # Singular values
        self.lora_E[adapter_name] = nn.Parameter(torch.randn(r, 1))
        # Left singular vectors
        self.lora_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, r))
        # The current rank
        self.ranknum[adapter_name] = nn.Parameter(torch.randn(1), requires_grad=False)
        self.ranknum[adapter_name].data.fill_(float(r))
        self.ranknum[adapter_name].requires_grad = False
        # The current firing-rate
        self.firing_rate[adapter_name] = nn.Parameter(torch.randn(1), requires_grad=False)
        self.firing_rate[adapter_name].data.fill_(float(0))
        self.firing_rate[adapter_name].requires_grad = False     
        self.spiking_neuron[adapter_name] = neuron.IFNode(step_mode='s')

        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.zeros_(self.lora_E[adapter_name])
            nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.02)


class SNNSVDLinear(nn.Module, SNNAdaLoraLayer):
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        SNNAdaLoraLayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)


    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            # import pdb
            # pdb.set_trace()
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_E = self.lora_E[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            ranknum = self.ranknum[active_adapter] + 1e-5
            spiking_nueron = self.spiking_neuron[active_adapter]

            x = x.to(lora_A.dtype)
            
            T = 10
            firing_rate = 0
            encoder = encoding.PoissonEncoder()
            for _ in range(T):
                encoded_x = encoder(dropout(x))
                firing_rate += spiking_nueron(encoded_x @ (lora_A * lora_E).T)
            
            firing_rate = firing_rate / T
            self.firing_rate[active_adapter].data.fill_(firing_rate.mean())
            result += firing_rate @ lora_B.T * scaling / ranknum
            # result += (dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * scaling / ranknum
            functional.reset_net(self)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "adalora." + rep


class DLRankAllocator:
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                self.init_bgt += p.size(0)
                self.name_set.add(n.replace("lora_A", "%s"))
        self.name_set = sorted(self.name_set)
        # The total final rank budget
        self.target_bgt = self.peft_config.target_r * len(self.name_set)


    def budget_schedule(self, step: int):
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        # Initial warmup
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        # Final fine-tuning
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            # Budget decreasing with a cubic scheduler
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int((self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt)
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return budget, mask_ind

    def update_ipt(self, model):
        # Update the sensitivity and uncertainty for every weight
        for n, p in model.named_parameters():
            if "lora_" in n and self.adapter_name in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                with torch.no_grad():
                    if p.requires_grad == True:
                        if deepspeed_config() is not None:
                            import deepspeed

                            grad = deepspeed.utils.safe_get_full_grad(p)
                            self.ipt[n] = (p * grad - 0.5 * (p * grad)**2).abs().detach()
                        else:
                            self.ipt[n] = (p * p.grad - 0.5 * (p * p.grad)**2).abs().detach()
                    else :
                        self.ipt[n] = torch.zeros_like(p)
                    # Sensitivity smoothing
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]

    def _element_score(self, n):
        return self.exp_avg_ipt[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_budget(self, model, budget, layer_wise):
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        firing_rate = {}
        # Get the importance score for A, E, B
        for n, p in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace("lora_A", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_B.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace("lora_B", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_E.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                name_m = n.replace("lora_E", "%s")
                value_ipt[name_m] = entry_ipt
            
            if f"firing_rate.{self.adapter_name}" in n:
                name_m = n.replace("firing_rate", "%s")
                firing_rate[name_m] = p.item()
        # import pdb
        # pdb.set_trace()
        all_score = []
        layers_ipt = [0]*model.config.num_hidden_layers
        layers_firing_rate = [[] for _ in range(model.config.num_hidden_layers)]
        # Calculate the score for each triplet
        for name_m in vector_ipt:
            ipt_E = value_ipt[name_m]
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_m % "lora_E"
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))

            layer_index = 0
            import re
            # 使用正则表达式提取数字
            match = re.search(r"\.(\d+)\.", name_m)

            if match:
                # 提取数字
                layer_index = int(match.group(1))
            else:
                layer_index = None
            
            if layer_index:
                layers_ipt[layer_index] += sum_ipt.sum().item()
                layers_firing_rate[layer_index].append(firing_rate[name_m])
            
        for i in range(len(layers_firing_rate)):
            sublist = layers_firing_rate[i]
            layers_firing_rate[i] = sum(sublist) / len(sublist) if sublist else 0.0
        
        combined_ipt_layers = self.layer_wise_training(model, layers_ipt, layers_firing_rate, layer_wise)
        # logger.info(f"After layer-wise choose remain traing Params: {train_params}")

        # Get the threshold by ranking ipt
        mask_threshold = torch.kthvalue(
            torch.cat(all_score),
            k=self.init_bgt - budget,
        )[0].item()

        rank_pattern = {}
        # Mask the unimportant triplets
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                    rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
        
        
        return rank_pattern, layers_firing_rate, layers_ipt, combined_ipt_layers

    
    def layer_wise_training(self, model, layers_ipt:list, layers_firing_rate:list, layer_wise):

        w1 = 0.3
        w2 = 0.7
        def min_max_normalize(lst):
            min_val = min(lst)
            max_val = max(lst)
            if min_val != max_val : 
                return np.array([(x - min_val) / (max_val - min_val) for x in lst])
            else:
                return np.array(lst)

        combined_ipt_list = w1 * min_max_normalize(layers_ipt)  + w2 * min_max_normalize(layers_firing_rate)
        if layer_wise == 'single':
            combined_ipt_layers = sorted(range(len(combined_ipt_list)), key=lambda i: combined_ipt_list[i], reverse=True)[:1]
        else:
            combined_ipt_layers = sorted(range(len(combined_ipt_list)), key=lambda i: combined_ipt_list[i], reverse=True)[:(model.config.num_hidden_layers//2)]

        traing_sum_param = 0 
        
        # in distribute env the model name will set as: module.deberta.encoder.layer.0.attention.self.query_proj.lora A so [4]
        for name, param in model.named_parameters(): 
            if 'lora' in name:
                layer_index = None
                import re
                # 使用正则表达式提取数字
                match = re.search(r"\.(\d+)\.", name)
                if match:
                    # 提取数字
                    layer_index = int(match.group(1))
                else:
                    layer_index = None   
                if 'lora_E' in name or layer_index in combined_ipt_layers:
                    param.requires_grad = True
                    sub_num_param = 1 
                    for dim in param.shape:
                        sub_num_param *= dim  
                    traing_sum_param += sub_num_param
                else:
                    param.requires_grad = False
        
        print(f"After dynamic choose layers training the total params is {traing_sum_param:,}")
        
        return combined_ipt_layers
        
        
    def update_and_allocate(self, model, global_step, layer_wise, force_mask=False):
        # import pdb
        # pdb.set_trace()
        # # Update the importance score and allocate the budget
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)
        budget, mask_ind = self.budget_schedule(global_step)
        # Allocate the budget according to importance scores
        if mask_ind or force_mask:
            rank_pattern, layers_firing_rate, layers_ipt, combined_ipt_layers = self.mask_to_budget(model, budget, layer_wise)
        else:
            rank_pattern = None
            layers_firing_rate = None 
            layers_ipt = None
            combined_ipt_layers = None
        return budget, rank_pattern, layers_firing_rate, layers_ipt, combined_ipt_layers

    def mask_using_rank_pattern(self, model, rank_pattern):
        # Mask the unimportant triplets
        is_adapter_name_truncated = False
        if self.adapter_name not in next(iter(rank_pattern.keys())):
            is_adapter_name_truncated = True

        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    key = n if not is_adapter_name_truncated else n.replace(f".{self.adapter_name}", "")
                    mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
                    p.masked_fill_(~mask.bool(), 0.0)




class RankAllocator:
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                self.init_bgt += p.size(0)
                self.name_set.add(n.replace("lora_A", "%s"))
        self.name_set = sorted(self.name_set)
        # The total final rank budget
        self.target_bgt = self.peft_config.target_r * len(self.name_set)

    def budget_schedule(self, step: int):
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        # Initial warmup
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        # Final fine-tuning
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            # Budget decreasing with a cubic scheduler
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int((self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt)
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return budget, mask_ind

    def update_ipt(self, model):
        # Update the sensitivity and uncertainty for every weight
        for n, p in model.named_parameters():
            if "lora_" in n and self.adapter_name in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    if deepspeed_config() is not None:
                        import deepspeed

                        grad = deepspeed.utils.safe_get_full_grad(p)
                        self.ipt[n] = (p * grad).abs().detach()
                    else:
                        self.ipt[n] = (p * p.grad).abs().detach()
                    # Sensitivity smoothing
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                    # Uncertainty quantification
                    self.exp_avg_unc[n] = (
                        self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    )

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_budget(self, model, budget):
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        # Get the importance score for A, E, B
        for n, p in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace("lora_A", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_B.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace("lora_B", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_E.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                name_m = n.replace("lora_E", "%s")
                value_ipt[name_m] = entry_ipt

        all_score = []
        # Calculate the score for each triplet
        for name_m in vector_ipt:
            ipt_E = value_ipt[name_m]
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_m % "lora_E"
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))

        # Get the threshold by ranking ipt
        mask_threshold = torch.kthvalue(
            torch.cat(all_score),
            k=self.init_bgt - budget,
        )[0].item()

        rank_pattern = {}
        # Mask the unimportant triplets
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                    rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
        return rank_pattern

    def update_and_allocate(self, model, global_step, force_mask=False):
        # # Update the importance score and allocate the budget
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)
        budget, mask_ind = self.budget_schedule(global_step)
        # Allocate the budget according to importance scores
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None
        
        layers_firing_rate = None 
        layers_ipt = None
        combined_ipt_layers = None
        return budget, rank_pattern, layers_firing_rate, layers_ipt, combined_ipt_layers

    def mask_using_rank_pattern(self, model, rank_pattern):
        # Mask the unimportant triplets
        is_adapter_name_truncated = False
        if self.adapter_name not in next(iter(rank_pattern.keys())):
            is_adapter_name_truncated = True

        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    key = n if not is_adapter_name_truncated else n.replace(f".{self.adapter_name}", "")
                    mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
                    p.masked_fill_(~mask.bool(), 0.0)