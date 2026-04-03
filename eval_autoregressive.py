"""
Baseline evaluation using batched autoregressive generation.
Uses the same interface as main2.py (same configs, dataset, logits processor)
so tokens/s can be directly compared against speculative decoding.
"""
import os
import torch
import gc
import tqdm
import json
import pickle
import numpy as np
from transformers import set_seed
from . import autoregressive_generate_batch
from . import NucleusProcessor, GreedyProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse

SEED = 42
set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()

if args.config is not None:
    with open(args.config, 'r') as f:
        config = json.load(f)
else:
    config = {
        'dir': './results/baseline-autoregressive/',
        'num_prompts': 1000,
        'gen_len': 200,
        'logits_processor': {
            'type': 'NucleusProcessor',
            'temperature': 0.6,
            'top_p': 0.95
        },
        'dataset_name': 'openai/gsm8k',
        'models': {
            'target': 'Qwen/Qwen3-4B-Instruct-2507',
        },
        'batch_size': 16,
        'show_output': True,
    }

if not os.path.exists(config['dir']):
    os.makedirs(config['dir'])

NUM_PROMPTS = config['num_prompts']
gen_len = config['gen_len']

if config['logits_processor']['type'] == 'NucleusProcessor':
    logits_processor = NucleusProcessor(
        temperature=config['logits_processor']['temperature'],
        top_p=config['logits_processor']['top_p'],
    )
else:
    logits_processor = GreedyProcessor()

# Load dataset
if config.get('dataset_name') == 'openai/gsm8k':
    dataset = load_dataset("openai/gsm8k", "main", split='test').shuffle(seed=SEED)
else:
    dataset = load_dataset("rishabhrj11/cnn_dailymail_512", split='test').shuffle(seed=SEED)

dataset = dataset.select(list(range(NUM_PROMPTS)))

target = config['models']['target']
tokenizer = AutoTokenizer.from_pretrained(target)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

def map_to_prompts(example):
    if config.get('dataset_name') == 'openai/gsm8k':
        message = [
            {"role": "system", "content": "In math word problem given by the user, reason step by step and put your final answer within \\boxed{}"},
            {"role": "user", "content": example["question"]}
        ]
    else:
        message = [
            {"role": "system", "content": "Write a very short summary for the user's article."},
            {"role": "user", "content": example["article"]}
        ]
    return {'prompt': tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )}

dataset = dataset.map(map_to_prompts)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
target_model = AutoModelForCausalLM.from_pretrained(
    target, device_map=device, dtype=torch.bfloat16, attn_implementation="sdpa",
)
target_model.set_attn_implementation("eager")
target_model.eval()

use_cache = config.get('use_cache', False)

def evaluate_generation(prompts, max_new_tokens):
    outputs = []
    batch_size = config.get('batch_size', 16)

    for start_idx in tqdm.tqdm(range(0, len(prompts), batch_size), total=(len(prompts) + batch_size - 1) // batch_size):
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]['prompt']
        print(f"Evaluating prompts {start_idx} to {end_idx-1}...")

        tokenized = tokenizer(batch_prompts, padding='longest', return_tensors='pt')
        output_ids, _ = autoregressive_generate_batch(
            tokenized.input_ids,
            tokenized.attention_mask,
            target_model,
            logits_processor=logits_processor,
            max_gen_len=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            debug=config.get('show_output', True),
            use_cache=use_cache,
        )
        outputs.extend(output_ids)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()

    return outputs

print(f"Evaluating target model (autoregressive): {target}")
print(f"use_cache={use_cache}")
outputs = evaluate_generation(dataset, gen_len)

with open(os.path.join(config['dir'], 'autoregressive-outputs.pkl'), 'wb') as f:
    pickle.dump(outputs, f)

print(f"Saved to {config['dir']}")

with open(os.path.join(config['dir'], 'autoregressive-config.json'), 'w') as f:
    json.dump(config, f, indent=4)

del outputs
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
