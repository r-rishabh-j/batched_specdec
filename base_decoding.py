import time
import torch
from torch.nn import Module
from transformers import DynamicCache
from .logits_processor import LogitsProcessor, GreedyProcessor
from . import printing
from typing import List
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.console import Console, Group


def _get_max_seq_length(model: Module) -> int:
    return getattr(
        model.config,
        "max_position_embeddings",
        getattr(model.config, "max_context_length", 1024),
    )


def _left_pad_batch(input_ids, attn_mask, pad_token_id: int, width: int):
    """Normalize a padded batch to left padding and truncate to the last `width` real tokens."""
    batch_size = input_ids.shape[0]
    normalized_ids = torch.full(
        (batch_size, width),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    normalized_mask = torch.zeros(
        (batch_size, width),
        dtype=attn_mask.dtype,
        device=attn_mask.device,
    )
    prompt_lens = torch.zeros(batch_size, dtype=torch.long, device=attn_mask.device)

    for row in range(batch_size):
        tokens = input_ids[row, attn_mask[row].bool()]
        if tokens.numel() > width:
            tokens = tokens[-width:]
        length = tokens.numel()
        prompt_lens[row] = length
        if length > 0:
            normalized_ids[row, width - length :] = tokens
            normalized_mask[row, width - length :] = 1

    return normalized_ids, normalized_mask, prompt_lens


@torch.no_grad()
def autoregressive_generate(
    inputs: List[int],
    model: Module,
    max_gen_len: int = 40,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    debug: bool = False,
) -> List[int]:
    """
    Generate text sequence autoregressively based on the input sequence.

    Args:
        inputs (List[int]): input sequence of batch size 1.
        model (Module): model to use for inference.
        max_gen_len (int): maximum length of the generated sequence.
        logits_processor (LogitsProcessor): logits processor for sampling.
        eos_tokens_id (int): end token id.
        pad_token_id (int): pad token id.
        use_cache (bool): whether to use cache.

    Returns:
        List[int]: generated sequence.

    Note:
        This generation methods only works for decoder-only models.
    """
    if max_gen_len <= 0:
        return []

    max_seq_length = _get_max_seq_length(model)
    prompt_tokens = torch.tensor(inputs[-max_seq_length:], dtype=torch.long, device=model.device)
    prompt_len = int(prompt_tokens.numel())
    if prompt_len == 0 or prompt_len >= max_seq_length:
        return []

    cache = None
    # prepare input tensor
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=model.device)
    input_ids[0, :prompt_len] = prompt_tokens

    list_tokens_id = (
        eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    )
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=model.device)
    generated_end = prompt_len

    for curr in range(prompt_len, total_len):
        if use_cache:
            if cache is None:
                # Prefill: feed full prompt to populate cache
                o = model(input_ids[..., :curr], past_key_values=cache, use_cache=True)
            else:
                # Decode: feed last token only, context is in cache
                o = model(input_ids[..., curr-1:curr], past_key_values=cache, use_cache=True)
            cache = o.past_key_values
        else:
            o = model(input_ids[..., :curr], use_cache=False)
        logits = o.logits[..., -1, :]  # [1, vocab_size]
        probs = logits_processor(logits)  # [1, vocab_size]
        x = logits_processor.sample(probs)  # [1, 1]
        input_ids[0, curr] = x
        generated_end = curr + 1

        # check for end token
        if torch.isin(x, stop_tokens):
            if debug:
                printing.end_token_found(curr)
            break

    return input_ids[0, prompt_len:generated_end].tolist()


@torch.no_grad()
def autoregressive_generate_batch(
    input_ids,
    attn_mask,
    model: Module,
    tokenizer=None,
    max_gen_len: int = 40,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    eos_token_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = True,
    debug: bool = True,
):
    """
    Batched autoregressive decoding for variable-length (left-padded) prompts.
    Matches speculative_generate_batch_v3's interface for fair speed comparison.
    Returns generated token IDs per sample and empty stats placeholder.
    """
    device = model.device
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    B = input_ids.shape[0]

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_tensor = torch.tensor(eos_token_id, device=device, dtype=input_ids.dtype)

    max_seq_length = _get_max_seq_length(model)
    if max_gen_len <= 0 or max_seq_length <= 0:
        return [[] for _ in range(B)], []

    prompt_width = min(input_ids.shape[1], max_seq_length)
    input_ids, attn_mask, prompt_lens = _left_pad_batch(
        input_ids,
        attn_mask,
        pad_token_id,
        prompt_width,
    )

    if torch.any(prompt_lens == 0):
        raise ValueError("autoregressive_generate_batch requires at least one prompt token per sample")

    max_total_length = int(min(max_seq_length, prompt_width + max_gen_len))
    batch_position = prompt_width

    if batch_position >= max_total_length:
        return [[] for _ in range(B)], []

    # Pad for generation space
    pad_len = max_total_length - prompt_width
    if pad_len > 0:
        pad_ids = torch.full((B, pad_len), pad_token_id, dtype=input_ids.dtype, device=device)
        attn_pad = torch.zeros((B, pad_len), dtype=attn_mask.dtype, device=device)
        input_ids = torch.cat([input_ids, pad_ids], dim=1)
        attn_mask = torch.cat([attn_mask, attn_pad], dim=1)

    cache = DynamicCache() if use_cache else None
    debug = debug and tokenizer is not None

    # PREFILL
    mask = attn_mask[:, :batch_position]
    pos_id = (mask.cumsum(dim=1) - 1).clamp(min=0)
    output = model(
        input_ids=input_ids[:, :batch_position],
        attention_mask=mask,
        position_ids=pos_id,
        past_key_values=cache,
        use_cache=use_cache,
    )
    logits = output.logits[:, -1]
    if use_cache:
        cache = output.past_key_values

    probs = logits_processor(logits)
    new_token = logits_processor.sample(probs)
    input_ids[:, batch_position] = new_token
    attn_mask[:, batch_position] = 1
    batch_position += 1

    start_time = time.perf_counter()
    total_tokens = B

    active_status = torch.ones(B, dtype=torch.bool, device=device)
    active_status &= ~(new_token.unsqueeze(-1) == eos_tensor).any(-1)

    if debug:
        console = Console()
        header = Text(f"Batch Autoregressive | 0.00 tokens/s", style="bold magenta")
        texts = [Text(tokenizer.decode(new_token[b].item(), skip_special_tokens=True)) for b in range(B)]
        panels = [Panel(texts[i], border_style="cyan", title=f"Prompt {i+1}") for i in range(B)]
        group = Group(header, *panels)
        live = Live(group, console=console, refresh_per_second=10, transient=True)
        live.start()

    while torch.any(active_status) and batch_position < max_total_length:
        active_indices = active_status.nonzero().flatten()

        if use_cache:
            mask = attn_mask[:, :batch_position]
            pos_id = (mask.sum(dim=1) - 1).unsqueeze(1)
            output = model(
                input_ids=input_ids[:, batch_position - 1].unsqueeze(1),
                attention_mask=mask,
                position_ids=pos_id,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
        else:
            mask = attn_mask[:, :batch_position]
            pos_id = (mask.cumsum(dim=1) - 1).clamp(min=0)
            output = model(
                input_ids=input_ids[:, :batch_position],
                attention_mask=mask,
                position_ids=pos_id,
                use_cache=False,
            )

        logits = output.logits[:, -1]
        probs = logits_processor(logits)
        new_token = logits_processor.sample(probs)

        input_ids[active_indices, batch_position] = new_token[active_indices]
        attn_mask[active_indices, batch_position] = 1

        is_eos = (new_token.unsqueeze(-1) == eos_tensor).any(-1)
        active_status &= ~is_eos

        num_active = len(active_indices)
        total_tokens += num_active

        if debug:
            elapsed = time.perf_counter() - start_time
            tps = total_tokens / elapsed if elapsed > 0 else 0
            header.plain = f"Batch Autoregressive | {tps:.2f} tokens/s"
            for b in active_indices:
                decoded = tokenizer.decode(new_token[b].item(), skip_special_tokens=True)
                texts[b].append(decoded)
            live.refresh()

        batch_position += 1

    elapsed = time.perf_counter() - start_time
    tps = total_tokens / elapsed if elapsed > 0 else 0
    print(f"\nGeneration complete: {total_tokens} tokens in {elapsed:.2f}s ({tps:.2f} tokens/s)", flush=True)

    if debug:
        live.stop()
        texts.clear()
        del texts, group, live, console

    outputs = []
    for i in range(B):
        valid_pos = torch.where(attn_mask[i])[0]
        full_real = input_ids[i, valid_pos]
        generated = full_real[int(prompt_lens[i].item()):]
        outputs.append(generated.tolist())

    return outputs, []
