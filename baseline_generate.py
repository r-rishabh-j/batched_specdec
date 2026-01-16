"""
Baseline evaluation using HuggingFace generate API.
Uses batching, KV caching, and nucleus sampling on target model only.
"""
import os
import time
import torch
import gc
import tqdm
import json
import pickle
import numpy as np
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse

SEED = 44
set_seed(SEED)

# load config from args. if none, default to file
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()

if args.config is not None:
    with open(args.config, 'r') as f:
        config = json.load(f)
else:
    config = {
        'dir': './results/baseline/',
        'num_prompts': 100,
        'gen_len': 100,
        'logits_processor': {
            'type': 'NucleusProcessor',
            'temperature': 0.6,
            'top_p': 0.95
        },
        'dataset_name': 'openai/gsm8k',
        'models': {
            'target': 'Qwen/Qwen3-4B-Instruct-2507',
        },
        'batch_size': 2,
        'show_output': True,
    }

if not os.path.exists(config['dir']):
    os.makedirs(config['dir'])

NUM_PROMPTS = config['num_prompts']
gen_len = config['gen_len']

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
tokenizer.padding_side = 'left'  # Required for batched generation

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
target_model = AutoModelForCausalLM.from_pretrained(target, device_map=device, torch_dtype=torch.bfloat16)
target_model.eval()

def evaluate_generation(prompts, max_new_tokens):
    outputs = []
    batch_size = config.get('batch_size', 4)
    total_tokens = 0
    start_time = time.perf_counter()

    # Get sampling parameters
    lp_config = config.get('logits_processor', {})
    temperature = lp_config.get('temperature', 1.0)
    top_p = lp_config.get('top_p', 1.0)
    do_sample = lp_config.get('type') != 'GreedyProcessor'

    for start_idx in tqdm.tqdm(range(0, len(prompts), batch_size), total=(len(prompts)+batch_size-1)//batch_size):
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]['prompt']
        
        # Tokenize with padding for batched generation
        inputs = tokenizer(
            batch_prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).to(device)
        
        prompt_lengths = inputs.attention_mask.sum(dim=1)
        
        # Generate using HuggingFace generate with KV caching
        with torch.no_grad():
            generated = target_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,  # Enable KV caching
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Extract only generated tokens (exclude prompt)
        batch_tokens = 0
        for i in range(len(batch_prompts)):
            prompt_len = prompt_lengths[i].item()
            output_tokens = generated[i, prompt_len:].tolist()
            # Remove padding tokens
            if tokenizer.pad_token_id in output_tokens:
                output_tokens = output_tokens[:output_tokens.index(tokenizer.pad_token_id)]
            outputs.append(output_tokens)
            batch_tokens += len(output_tokens)
        
        total_tokens += batch_tokens
        elapsed = time.perf_counter() - start_time
        tps = total_tokens / elapsed if elapsed > 0 else 0
        print(f"Batch {start_idx//batch_size + 1}: {batch_tokens} tokens | Cumulative: {tps:.2f} tokens/s", flush=True)
        
        if config.get('show_output', False):
            # Print first output of batch
            print(f"\n--- Batch {start_idx//batch_size + 1} ---")
            print(tokenizer.decode(outputs[-len(batch_prompts)], skip_special_tokens=True)[:500])
        
        # Free memory after each batch
        del inputs, generated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()

    elapsed = time.perf_counter() - start_time
    tps = total_tokens / elapsed if elapsed > 0 else 0
    print(f"\nGeneration complete: {total_tokens} tokens in {elapsed:.2f}s ({tps:.2f} tokens/s)", flush=True)

    return outputs, tps

print(f"Evaluating target model: {target}")
outputs, tokens_per_second = evaluate_generation(dataset, gen_len)

# Save outputs
with open(os.path.join(config['dir'], 'baseline-outputs.pkl'), 'wb') as f:
    pickle.dump(outputs, f)

config['results'] = {
    'tokens_per_second': tokens_per_second,
    'total_prompts': NUM_PROMPTS,
}

print(f"Tokens/s: {tokens_per_second:.2f}")
print(f"Saved to {config['dir']}")

with open(os.path.join(config['dir'], 'baseline-config.json'), 'w') as f:
    json.dump(config, f, indent=4)

# Cleanup
del outputs
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
