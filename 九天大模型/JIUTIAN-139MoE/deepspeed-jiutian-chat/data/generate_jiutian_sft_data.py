#!/usr/bin/env python
#-*- encoding:utf-8 -*-

import sys, os, re
import datasets, copy
from transformers import AutoTokenizer

IGNORE_INDEX = -100
def preprocess(example):
    """
        data format:
        {
            "conversations": [
                {"from": "human", "value": "prompt1"},
                {"from": "assistant", "value": "response1"},
                {"from": "human", "value": "prompt2"},
                {"from": "assistant", "value": "response2"},
                ...
            ]
        }
    """
    data = {"conversations": example["conversations"]}
    return data

def generate_and_tokenize_prompt(data_point, model_max_length=2048, field="conversations"):
    input_ids = []
    labels = []
    source = data_point[field]
    for sentence in source:
        sentence_from = sentence["from"].lower()
        sentence_value = "Human:\n" + sentence["value"] + "\n\nAssistant:\n" if sentence_from == "human" else sentence["value"]
        sentence_ids = tokenizer.encode(sentence_value, add_special_tokens=False)
        label = copy.deepcopy(sentence_ids) if sentence_from != "human" else [IGNORE_INDEX] * len(sentence_ids)
        input_ids += sentence_ids
        labels += label
        # add eos at every end of assistant sentence
        if sentence_from != "human":
            input_ids += [tokenizer.pad_token_id]
            labels += [tokenizer.pad_token_id]
    input_ids = input_ids[: model_max_length]
    labels = labels[: model_max_length]
    attention_mask = [1] * len(input_ids)

    tokenized_full_prompt = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return tokenized_full_prompt


if __name__ == "__main__":
    model_path = "/llm/jiutian139"
    input_file = "./data.json"
    out_file = "./tokenizer_datasets"
    ds = datasets.load_dataset("json", data_files=input_file, split="train")

    pre_ds = ds.map(preprocess, remove_columns=ds.column_names)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = "<unk>"

    # tokenize
    tokenized_dataset = pre_ds.map(generate_and_tokenize_prompt, num_proc=4, remove_columns=pre_ds.column_names)
    print("tokenized_dataset.column_names : ", tokenized_dataset.column_names)

    tokenized_dataset.save_to_disk(out_file)
    # test tokenizer result
    print(tokenizer.decode(tokenized_dataset[-1]["labels"][-10:]), "\n")
