# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

GLOBAL_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 1

def get_train_ds_config(offload,
                        stage=1):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        "checkpoint-activations": True,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "overlap_comm": True
    }