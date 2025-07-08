# conda #!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import math
import sys
import json
from datasets import Dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    SchedulerType,
    default_data_collator,
    get_scheduler,
    AutoTokenizer,
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq,
)
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator
from peft import LoraConfig, get_peft_model

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.ds_utils import get_train_ds_config

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--model_name_or_path", type=str, default="/llm/ckpt_jiutian_hf/jiutian")
    parser.add_argument("--train_data", type=str, default="./data/tokenizer_datasets")
    parser.add_argument("--domain_train_data", type=str, default="")
    parser.add_argument("--valid_data", type=str, default="")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--resume_step", type=str, default="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="/llm/jiutian2sftckpt")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    parser.add_argument('--offload', action='store_true', default=True)
    parser.add_argument('--zero_stage', type=int, default=3)
    ##lora
    #parser.add_argument("--lora_config", type=str, help="Lora", default="./config/lora_config.json")
    parser.add_argument("--lora_config", type=str, help="Lora", default="")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.local_rank == -1: 
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    if args.global_rank <= 0:
        print(args.__dict__)
        print(ds_config)
        log_dir = args.output_dir
        os.makedirs(log_dir, exist_ok=True)
        with open(log_dir + "/args.json", "w") as f:
            json.dump(args.__dict__, f, indent=4)
        with open(log_dir + "/ds_config.json", "w") as f:
            json.dump(ds_config, f, indent=4)

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()
    dschf = HfDeepSpeedConfig(ds_config)
    print_rank_0("model_name_or_path : " + args.model_name_or_path, args.global_rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.eos_token_id = 0
    tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                             torch_dtype=torch.bfloat16, use_flash_attention_2=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    # Lora config
    if args.lora_config:
        lora_config = json.load(open(args.lora_config))
        config = LoraConfig(
            r=lora_config["lora_r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["lora_target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            modules_to_save=lora_config["modules_to_save"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    # Prepare the data
    train_dataset = Dataset.load_from_disk(args.train_data)
    if args.domain_train_data:
        domain_train_dataset = Dataset.load_from_disk(args.domain_train_data)
        train_dataset = concatenate_datasets([train_dataset, domain_train_dataset])
    try:
        train_dataset = train_dataset.remove_columns("cnt_token")
    except:
        pass
    print_rank_0("***** Data load success! *****", args.global_rank)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    print(f"args.per_device_train_batch_size = {args.per_device_train_batch_size}")
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
        
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    
    # Train!
    if args.resume_step:
        _, client_state = model.load_checkpoint(args.output_dir, args.resume_step)
        print_rank_0(f"client state: {client_state}", args.global_rank)
        checkpoint_step = int(args.resume_step)
    else:
        checkpoint_step = -1

    cur_epoch = 0
    global_step = 0
    resume_step = -1
    if checkpoint_step != -1:
        cur_epoch = checkpoint_step // len(train_dataloader)
        global_step = checkpoint_step
        resume_step = checkpoint_step % len(train_dataloader)
        print_rank_0(f"RESUME GLOBAL STEP: {global_step}", args.global_rank)
        print_rank_0(f"RESUME CURRENT STEP: {resume_step}", args.global_rank)
        
    print_rank_0("***** Running training *****", args.global_rank)
    
    for epoch in range(args.num_train_epochs):
        if epoch < cur_epoch:
            continue
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            if step < resume_step:
                continue
            else:
                resume_step = -1
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            if args.lora_config:
                loss = outputs[0]
            else:
                loss = outputs.loss
            model.backward(loss)
            model.step()
            if global_step % (args.gradient_accumulation_steps * 1) == 0:
                loss_reduce = get_all_reduce_mean(loss).item()
                print_rank_0("Epoch: {:.2f}, step: {}, loss: {:.4f}".format(global_step / len(train_dataloader), global_step, loss_reduce), args.global_rank)
            get_accelerator().empty_cache()
            if global_step % (args.save_interval * args.gradient_accumulation_steps) == 0 and global_step != checkpoint_step and global_step > 0:
                print_rank_0(f"Save Checkpoint on Step: {global_step}", args.global_rank)
                if args.global_rank == 0:
                    save_hf_format(model, tokenizer, args, sub_folder="checkpoint-" + str(global_step))
                if args.zero_stage == 3:
                    # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                    save_zero_three_model(tokenizer,
                                          model,
                                          args.global_rank,
                                          args.output_dir + "/epoch-" + str(epoch),
                                          args.lora_config,
                                          args.model_name_or_path,
                                          zero_stage=args.zero_stage)
            global_step += 1
        model.tput_timer.update_epoch_count()
        if args.output_dir is not None:
            print_rank_0('saving the final model ...', args.global_rank)
            if args.global_rank == 0:
                save_hf_format(model, tokenizer, args, sub_folder="epoch-" + str(epoch))
            if args.zero_stage == 3:
                # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                save_zero_three_model(tokenizer,
                                      model,
                                      args.global_rank,
                                      args.output_dir + "/epoch-" + str(epoch),
                                      args.lora_config,
                                      args.model_name_or_path,
                                      zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
