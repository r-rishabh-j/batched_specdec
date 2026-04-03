"""
Unified evaluation script for speculative and autoregressive decoding.

Usage:
    python -m batched_specdec.evaluate --mode v1 --config eval_configs/baselines/smollm-gsm8k.json
    python -m batched_specdec.evaluate --mode v3 --config eval_configs/baselines/smollm-gsm8k.json
    python -m batched_specdec.evaluate --mode autoregressive --config eval_configs/baselines/smollm-gsm8k.json
"""
import os
import torch
import gc
import tqdm
import json
import pickle
import numpy as np
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse

from . import speculative_generate_batch, speculative_generate_batch_v2, speculative_generate_batch_v3, autoregressive_generate_batch
from . import NucleusProcessor, GreedyProcessor

SEED = 42
set_seed(SEED)

MODES = ['v1', 'v3', 'autoregressive']

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=MODES,
                    help='v1: speculative (right-padded, DynamicCache via CacheManager), '
                         'v3: speculative (left-padded, DynamicCache), '
                         'autoregressive: batched autoregressive baseline')
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()

if args.config is not None:
    with open(args.config, 'r') as f:
        config = json.load(f)
else:
    config = {
        'dir': './results/gsm-qwen-1000/',
        'num_prompts': 1000,
        'gen_len': 200,
        'gamma': 5,
        'logits_processor': {
            'type': 'NucleusProcessor',
            'temperature': 0.6,
            'top_p': 0.95
        },
        'dataset_name': 'openai/gsm8k',
        'models': {
            'target': 'Qwen/Qwen3-4B-Instruct-2507',
            'drafts': {
                'base': 'Qwen/Qwen3-0.6B',
            }
        },
        'batch_size': 16,
        'show_output': True,
    }

mode = args.mode

if not os.path.exists(config['dir']):
    os.makedirs(config['dir'])

NUM_PROMPTS = config['num_prompts']
gen_len = config['gen_len']
gamma = config.get('gamma', 5)
use_cache = config.get('use_cache', False)

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

# Load target model
target_model = AutoModelForCausalLM.from_pretrained(
    target, device_map=device, dtype=torch.bfloat16, attn_implementation="sdpa",
)
# target_model.set_attn_implementation("eager")
target_model.eval()


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


def run_v1(draft_model, prompts, max_new_tokens):
    """Speculative decoding v1: right-padded, DynamicCache via CacheManager."""
    results, alphas, outputs = [], [], []
    batch_size = config.get('batch_size')

    for start_idx in tqdm.tqdm(range(0, len(prompts), batch_size), total=(len(prompts) + batch_size - 1) // batch_size):
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]['prompt']
        print(f"Evaluating prompts {start_idx} to {end_idx-1}...")
        input_ids = tokenizer(batch_prompts, max_length=1024, truncation=True).input_ids
        output_ids, alpha, stats = speculative_generate_batch(
            input_ids,
            draft_model,
            target_model,
            logits_processor=logits_processor,
            gamma=gamma,
            max_gen_len=max_new_tokens,
            eos_tokens_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            collect_stats=True,
            tokenizer=tokenizer,
            debug=config.get('show_output', True),
        )
        print("Acceptance rate:", np.mean(alpha))
        results.extend(stats)
        alphas.extend(alpha)
        outputs.extend(output_ids)
        del input_ids
        free_memory()

    return results, alphas, outputs


def run_v3(draft_model, prompts, max_new_tokens):
    """Speculative decoding v3: left-padded, DynamicCache with cropping."""
    results, alphas, outputs = [], [], []
    batch_size = config.get('batch_size')

    for start_idx in tqdm.tqdm(range(0, len(prompts), batch_size), total=(len(prompts) + batch_size - 1) // batch_size):
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]['prompt']
        print(f"Evaluating prompts {start_idx} to {end_idx-1}...")
        tokenized = tokenizer(batch_prompts, padding='longest', return_tensors='pt')
        output_ids, alpha, stats = speculative_generate_batch_v3(
            tokenized.input_ids,
            tokenized.attention_mask,
            draft_model,
            target_model,
            logits_processor=logits_processor,
            gamma=gamma,
            max_gen_len=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            debug=config.get('show_output', True),
            use_cache=use_cache,
        )
        print("Acceptance rate:", np.mean(alpha))
        results.extend(stats)
        alphas.extend(alpha)
        outputs.extend(output_ids)
        free_memory()

    return results, alphas, outputs


def run_autoregressive(prompts, max_new_tokens):
    """Batched autoregressive baseline using target model only."""
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
        free_memory()

    return outputs


# Run
print(f"Mode: {mode} | Target: {target} | use_cache={use_cache}")

if mode == 'autoregressive':
    outputs = run_autoregressive(dataset, gen_len)
    with open(os.path.join(config['dir'], 'autoregressive-outputs.pkl'), 'wb') as f:
        pickle.dump(outputs, f)
    del outputs

else:
    drafts = config.get('models', {}).get('drafts', {})
    if not drafts:
        print("Error: speculative modes require 'models.drafts' in config")
        exit(1)

    run_fn = run_v1 if mode == 'v1' else run_v3

    for draft_name, draft in drafts.items():
        print(f"Evaluating draft model: {draft_name} ({draft})")
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft, device_map=device, dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        # draft_model.set_attn_implementation("eager")
        draft_model.eval()
        draft_model.config.use_cache = False

        results, alphas, outputs = run_fn(draft_model, dataset, gen_len)

        with open(os.path.join(config['dir'], f'{draft_name}-stats.pkl'), 'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(config['dir'], f'{draft_name}-outputs.pkl'), 'wb') as f:
            pickle.dump(outputs, f)

        print(f'Alpha for {draft_name}: {np.mean(alphas):.4f}')
        config[f'results_{draft_name}'] = {'alpha': float(np.mean(alphas))}

        del outputs, draft_model
        free_memory()

with open(os.path.join(config['dir'], f'{mode}-config.json'), 'w') as f:
    json.dump(config, f, indent=4)

print(f"Saved to {config['dir']}")
free_memory()
