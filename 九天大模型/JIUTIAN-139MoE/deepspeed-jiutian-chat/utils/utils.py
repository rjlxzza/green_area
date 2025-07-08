# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import re
import os
import json
import torch
import random
import shutil
import numpy as np
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import set_seed
from transformers.modeling_utils import shard_checkpoint
from peft import LoraConfig
from safetensors.torch import save_file as safe_save_file

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
#     torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if os.path.exists(args.model_name_or_path):
        for filename in os.listdir(args.model_name_or_path):
            if filename.endswith(".py"):
                shutil.copy(os.path.join(args.model_name_or_path, filename), os.path.join(output_dir, filename))

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_as_shards(state_dict, save_directory, weights_name, index_name, max_shard_size="10GB"):
    shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)
    # Clean the folder from a previous save
    for filename in os.listdir(save_directory):
        full_filename = os.path.join(save_directory, filename)
        # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
        # in distributed settings to avoid race conditions.
        weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")
        # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
        filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
        reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")
        if (
            filename.startswith(weights_no_suffix)
            and os.path.isfile(full_filename)
            and filename not in shards.keys()
            and reg.fullmatch(filename_no_suffix) is not None
        ):
            os.remove(full_filename)

    # Save the model
    for shard_file, shard in shards.items():
        safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
    if index is None:
        path_to_weights = os.path.join(save_directory, weights_name)
    else:
        save_index_file = os.path.join(save_directory, index_name)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=4, sort_keys=True) + "\n"
            f.write(content)

def get_peft_config(lora_config):
    lora_config = json.load(open(lora_config))
    config = LoraConfig(
        r=lora_config["lora_r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["lora_target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        modules_to_save=lora_config["modules_to_save"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    return config

def save_zero_three_model(tokenizer, model_ema, global_rank, save_dir, lora_config, model_name_or_path, zero_stage=0):
    """
    Rerwited by Jiutian_Team, to support zero3+lora+shards partition saving. More robust than previous function.
    """
    use_lora = False
    if lora_config:
        use_lora = False
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    model_to_save = model_ema.module if hasattr(model_ema, 'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            model_to_save.save_pretrained(save_dir, safe_serialization=True)
    else:
        output_state_dict = {}
        lora_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]),enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0:
                output_state_dict[k] = v_p
                if 'lora' in k:
                    lora_state_dict[k] = v_p
        if global_rank == 0:
            if use_lora:
                weight_name = "adapter_model.safetensors"
                config_name = "adapter_config.json"
                peft_config = get_peft_config(lora_config)
                lora_tgt = os.path.join(save_dir, weight_name)
                safe_save_file(lora_state_dict, lora_tgt)
                # lora_config
                peft_config.save_pretrained(save_dir)
            else:
                weight_name = "model.safetensors"
                weight_index_name = "model.safetensors.index.json"
                config_name = "config.json"
                gen_cfg_name = "generation_config.json"
                config_file = os.path.join(save_dir, config_name)
                model_to_save.config.to_json_file(config_file)
                save_as_shards(output_state_dict, save_dir, weight_name, weight_index_name)
            if os.path.exists(model_name_or_path):
                for filename in os.listdir(model_name_or_path):
                    if filename.endswith(".py"):
                        shutil.copy(os.path.join(model_name_or_path, filename), os.path.join(save_dir, filename))
        del output_state_dict
        del lora_state_dict
