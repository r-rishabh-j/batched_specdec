import time
import torch
from torch.nn import Module
from transformers import StaticCache, DynamicCache
from .logits_processor import LogitsProcessor, GreedyProcessor
from .cache_manager import CacheManager
from typing import List, Tuple
from torch.nn import functional as F
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.console import Console, Group

@torch.no_grad()
def speculative_generate_batch(
    prompts: List[List[int]],
    draft_model: Module,
    target_model: Module,
    tokenizer = None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    skip_sample_adjustment: bool = False,
    collect_stats: bool = False,
    use_cache: bool = True,
    debug: bool = True ):
    """
    Batched speculative decoding for variable-length prompts.
    Returns generated sequences, per-sample acceptance rates, and optional stats.
    """
    device = target_model.device
    vocab_size = int(target_model.config.vocab_size)
    B = len(prompts)
    batch_indexer = torch.arange(B, device=device)
    
    # configure lengths
    max_seq_length = target_model.config.max_position_embeddings if hasattr(target_model.config, 'max_position_embeddings') \
        else (target_model.config.max_context_length if hasattr(target_model.config, 'max_context_length') else 1024)
    prompt_lens = torch.tensor([len(prompt) for prompt in prompts], device=device)
    max_prompt_len = int(prompt_lens.max().item())
    max_total_length = int(min(max_seq_length, max_prompt_len+max_gen_len))

    # hold start pointers for next token fill for each batch element
    start_positions = prompt_lens.clone()
    
    # create input tensor for the model in input_ids, +1 for possible bonus token in the end
    input_ids = torch.full((B, max_total_length+1), pad_token_id, dtype=torch.long, device=device)

    padded_prompts = [prompt + [pad_token_id] * (max_prompt_len - len(prompt)) for prompt in prompts]
    input_ids[:, :max_prompt_len] = torch.tensor(padded_prompts, dtype=torch.long, device=device)

    # attention mask
    positions = torch.arange(max_total_length, device=device).unsqueeze(0)
    attn_mask = (positions < start_positions.unsqueeze(1)).to(dtype=torch.int).contiguous()

    # performance counters
    drafts_accepted = torch.zeros(B, device=device)
    drafts_speculated = torch.zeros(B, device=device)
    # for block efficiency calculation
    num_blocks = torch.zeros(B, device=device)


    cache_manager = CacheManager(use_cache)
    
    # generate the first token from target_model
    # max_prompt_len, attn_mask = attention_mask(start_positions)
    max_prompt_len = int(start_positions.max().item())
    target_model_output = target_model(
        input_ids = input_ids[:, :max_prompt_len],
        attention_mask = attn_mask[:, :max_prompt_len],
        use_cache=use_cache
    )
    target_model_logits = target_model_output.logits
    cache_manager.load_target_cache(target_model_output.past_key_values)
    # prefill draft model
    draft_output = draft_model(
        input_ids = input_ids[:, :max_prompt_len],
        attention_mask = attn_mask[:, :max_prompt_len],
        use_cache=use_cache,
    )
    cache_manager.load_draft_cache(draft_output.past_key_values)
    # get new token from end of target_model output and place in input_ids
    new_tok_probs = logits_processor(target_model_logits[batch_indexer, start_positions-1])
    # sample from target_model distribution
    new_token_batch = logits_processor.sample(new_tok_probs)
    input_ids[batch_indexer, start_positions]=new_token_batch
    attn_mask[batch_indexer, start_positions]=1
    # move pointers one step ahead
    start_positions += 1
    # Crop both caches to min position after first token
    cache_manager.prune_cache_to_min(start_positions)

    # Token speed tracking (always enabled)
    start_time = time.perf_counter()
    total_tokens = B  # Count first token for each batch element

    if debug:
        console = Console()
        header = Text(f"Batch Speculative decoding | 0.00 tokens/s", style="bold magenta")
        texts = [Text(tokenizer.decode(new_token_batch[b], skip_special_tokens=True)) for b in range(B)]
        panels = [Panel(texts[i], border_style="cyan", title=f"Prompt {i+1}") for i in range(B)]
        group = Group(header, *panels)
        live = Live(group, console=console, refresh_per_second=10, transient=True)
        live.start()

    # store which prompts are now terminated, either by exceeding max len 
    # or that eos token has been reached
    active_status = start_positions < max_total_length
    active_status &= (new_token_batch != eos_tokens_id)
    
    # Evict any sequences that finished on first token from cache
    cache_manager.evict_inactive(active_status)

    # buffers to store prob outputs from draft and target_model, redundant +1 for batching
    Q = torch.zeros(size=(B, gamma+1, vocab_size), device=device, dtype=target_model.dtype) # draft

    # run loop untill all sequences terminated
    while torch.any(active_status):
        # process only those sequences which are active
        active_indices = active_status.nonzero().flatten()  # Use flatten() to avoid scalar when B=1
        active_start_positions = start_positions[active_indices]
        max_length = int(active_start_positions.max().item())
        if max_length >= max_total_length:
            break
        num_active = len(active_indices)
        active_indexer = torch.arange(num_active, device=device)
        draft_steps = min(gamma, max_total_length - max_length - 1)
        if draft_steps <= 0:
            break

        Q.fill_(0)

        # use draft_model to generate draft_steps tokens
        draft_active = torch.full((num_active,), True, device=device)
        draft_lengths = torch.full_like(active_indices, 0, device=device)

        # next positions to be filled in drafting
        # draft cache is cropped to minimum seq length
        min_seq_len=cache_manager.get_draft_len()
        
        for k in range(draft_steps):
            # draft a token
            draft_output = draft_model(
                input_ids=input_ids[active_indices, min_seq_len+k:max_length+k],
                attention_mask=attn_mask[active_indices, :max_length+k],
                use_cache=use_cache,
                past_key_values=cache_manager.draft_cache
            )
            
            # get and process logits
            draft_logits = draft_output.logits
            Q[active_indices, k] = logits_processor(draft_logits[active_indexer, active_start_positions-min_seq_len-1])
            
            # sample tokens
            new_token_batch = logits_processor.sample(Q[active_indices, k])
            new_token_batch = torch.where(draft_active, new_token_batch, eos_tokens_id)

            # find eos token and mark inactive
            input_ids[active_indices, active_start_positions+k] = new_token_batch
            # update attention mask to include new token
            attn_mask[active_indices, active_start_positions+k] = 1
            draft_lengths += draft_active
            draft_active &= (new_token_batch != eos_tokens_id)

            # need to crop cache to seq with smallest length to avoid accessing invalid positions
            cache_manager.crop(min_seq_len+k+1, which='draft')

        drafts_speculated[active_indices] += draft_lengths

        # parallel verification of drafted tokens
        target_model_cache_len = cache_manager.get_target_len()
        target_model_output = target_model(
            input_ids=input_ids[active_indices, target_model_cache_len:max_length+draft_steps],
            attention_mask=attn_mask[active_indices, :max_length+draft_steps],
            use_cache=use_cache,
            past_key_values=cache_manager.target_cache
        )
        target_model_logits = target_model_output.logits
        del target_model_output  # Free model output container

        # eps for avoiding NaN
        eps = 1e-12

        # gather target_model probs for drafted tokens
        q = Q[active_indices, :draft_steps+1].clamp(min=eps) # [num_active, draft_steps+1, vocab_size]
        # create an index array to gather draft probs and tokens. Include an extra position for bonus token
        draft_indexer = torch.arange(draft_steps+1, device=device)

        drafted_indices=active_start_positions.unsqueeze(1) + draft_indexer # [num_active, draft_steps+1]
        drafted_tokens=input_ids[active_indices.unsqueeze(1), drafted_indices[:,:-1]] # [num_active, draft_steps]
        # -1 for left shift of model probs, use target_model_cache_len for offset
        p = logits_processor(target_model_logits[active_indexer.unsqueeze(1), drafted_indices-target_model_cache_len-1]).clamp(min=eps) # [num_active, draft_steps+1, vocab_size]
        p_tok = p[:, :draft_steps].gather(dim=2, index=drafted_tokens.unsqueeze(-1)).squeeze(-1) # [num_active, draft_steps]
        q_tok = q[:, :draft_steps].gather(dim=2, index=drafted_tokens.unsqueeze(-1)).squeeze(-1) # [num_active, draft_steps]
        del target_model_logits

        # compute log ratios for rejection sampling
        log_ratio = torch.log(p_tok) - torch.log(q_tok)
        log_r = torch.empty(log_ratio.shape, device=device).uniform_().log_()

        # Find first rejection (vectorized cumulative product)
        # take cumsum of accepted_mask, the first false breaks the chain
        acceptance_status = (log_r<=log_ratio).cumprod(dim=1).to(dtype=bool)  # [num_active, draft_steps]
        num_accepted = acceptance_status.sum(dim=1)  # [num_active,]
        active_idx_rejected, rejected_idx = torch.where(~acceptance_status)
        rejected_positions = drafted_indices[active_idx_rejected, rejected_idx]
        input_ids[active_indices[active_idx_rejected], rejected_positions] = pad_token_id
        # mask rejected tokens
        attn_mask[active_indices[active_idx_rejected], rejected_positions] = 0

        # perform resampling of last token - bonus or extra
        # mask for those sequences in which first extra token needs re-sampling
        extra_token_prob = p[active_indexer, num_accepted]
        if not skip_sample_adjustment:
            q_at_rejection = q[active_indexer, num_accepted]
            extra_token_prob = torch.where(
                (num_accepted<draft_steps).unsqueeze(1),
                torch.nn.functional.relu(extra_token_prob-q_at_rejection)+eps, # max(0, p-q)
                extra_token_prob
            )
        # sample extra tokens, includes bonus tokens
        extra_tokens = logits_processor.sample(extra_token_prob)
        input_ids[active_indices, active_start_positions+num_accepted]=extra_tokens
        # set bonus/extra token mask
        attn_mask[active_indices, active_start_positions+num_accepted]=1

        # find eos token and update active status, start positions
        drafted_tokens = input_ids[active_indices.unsqueeze(1), drafted_indices] # includes bonus token
        # Simple eos check without torch.isin
        eos_hits = (drafted_tokens == eos_tokens_id)
        eos_positions = torch.where(
            eos_hits,
            draft_indexer,
            max_seq_length
        ).min(dim=1).values
        has_eos = eos_positions <= num_accepted
        # update active sequences
        active_status[active_indices]&=~has_eos
        # correct lengths
        accepted_draft_length = torch.where(
            has_eos,
            eos_positions, 
            num_accepted
        )
        drafts_accepted[active_indices]+=accepted_draft_length
        num_blocks[active_indices] += 1

        # Update tokens count: accepted_draft_length + 1 (bonus token) for each active sequence
        tokens_this_iter = int((accepted_draft_length + 1).sum().item())
        total_tokens += tokens_this_iter

        if debug:
            elapsed = time.perf_counter() - start_time
            tps = total_tokens / elapsed if elapsed > 0 else 0
            header.plain = f"Batch Speculative decoding | {tps:.2f} tokens/s"
            
            active_idx = 0
            for b in active_indices:
                decoded = tokenizer.decode(drafted_tokens[active_idx, :accepted_draft_length[active_idx]+1].tolist(), skip_special_tokens=True)
                texts[b].append(decoded)
                active_idx+=1
            live.refresh()

        start_positions[active_indices]+=accepted_draft_length+1 # includes extra/bonus token
        
        # Evict newly inactive sequences from cache
        keep_mask = active_status[active_indices]
        cache_manager.evict_inactive(keep_mask)
        # Prune cache sequence length to minimum across remaining active sequences
        new_active_indices = active_status.nonzero().flatten()
        if new_active_indices.numel() > 0:
            new_start_positions = start_positions[new_active_indices].view(-1)
            cache_manager.prune_cache_to_min(new_start_positions)
        
    # Print final tokens/s and block efficiency
    elapsed = time.perf_counter() - start_time
    tps = total_tokens / elapsed if elapsed > 0 else 0
    total_blocks = int(num_blocks.sum().item())
    block_efficiency = total_tokens / total_blocks if total_blocks > 0 else 0
    print(f"\nGeneration complete: {total_tokens} tokens in {elapsed:.2f}s ({tps:.2f} tokens/s) | Block efficiency: {block_efficiency:.2f} tokens/block", flush=True)

    if debug:
        live.stop()
        texts.clear()
        del texts, group, live, console
    
    outputs = []
    acc_rates = []
    for i in range(B):
        end = min(int(start_positions[i].item()), max_total_length)
        outputs.append(input_ids[i, prompt_lens[i]:end].detach().cpu().tolist())
        denom = drafts_speculated[i].item() if drafts_speculated[i].item() > 0 else 1e-10
        acc_rates.append((drafts_accepted[i] / denom).item())
    
    # Cleanup
    del input_ids, attn_mask, Q, drafts_accepted, drafts_speculated, prompt_lens, start_positions, batch_indexer, active_status
    
    return outputs, acc_rates, []

@torch.no_grad()
def speculative_generate_batch_v2(
    input_ids,
    attn_mask,
    draft_model: Module,
    target_model: Module,
    tokenizer = None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    skip_sample_adjustment: bool = False,
    collect_stats: bool = False,
    use_cache: bool = True,
    debug: bool = True ):
    """
    Batched speculative decoding for variable-length prompts.
    Returns generated sequences, per-sample acceptance rates, and optional stats.
    """
    device = target_model.device
    vocab_size = int(target_model.config.vocab_size)
    B = input_ids.shape[0]
    
    # configure lengths
    max_seq_length = target_model.config.max_position_embeddings if hasattr(target_model.config, 'max_position_embeddings') \
        else (target_model.config.max_context_length if hasattr(target_model.config, 'max_context_length') else 1024)
    max_prompt_len = input_ids.shape[1]
    max_total_length = int(min(max_seq_length, max_prompt_len+max_gen_len*2))

    batch_position = input_ids.shape[1]

    pad_ids = torch.full((B, max_total_length-max_prompt_len+1), pad_token_id, dtype=input_ids.dtype)
    attn_pad = torch.zeros((B, max_total_length-max_prompt_len+1), dtype=attn_mask.dtype)
    input_ids = torch.cat([input_ids, pad_ids], dim=1).to(device)
    attn_mask = torch.cat([attn_mask, attn_pad], dim=1).to(device)

    # performance counters
    drafts_accepted = torch.zeros(B, device=device)
    drafts_speculated = torch.zeros(B, device=device)
    # for block efficiency calculation
    num_blocks = torch.zeros(B, device=device)

    # allocate model caches
    target_cache = StaticCache(
        config=target_model.config,
        max_cache_len=max_total_length,
        device=device,
        dtype=target_model.dtype
    )

    draft_cache = StaticCache(
        config=draft_model.config,
        max_cache_len=max_total_length,
        device=device,
        dtype=draft_model.dtype
    )
    
    # PREFILL
    # generate the first token from target_model
    # sequences are left padded, hence all last tokens are at batch_position-1
    mask = attn_mask[:, :batch_position]
    pos_id = (mask.cumsum(dim=1) - 1).clamp(min=0)
    cache_pos = torch.arange(batch_position, device=device)
    target_model_output = target_model(
        input_ids = input_ids[:, :batch_position],
        attention_mask = mask,
        position_ids = pos_id,
        cache_position = cache_pos,
        past_key_values = target_cache,
        use_cache = use_cache,
        output_hidden_states = False,
        output_attentions = False,
    )
    target_model_logits = target_model_output.logits
    # prefill draft model
    draft_output = draft_model(
        input_ids = input_ids[:, :batch_position],
        attention_mask = mask,
        position_ids = pos_id,
        cache_position = cache_pos,
        past_key_values = draft_cache,
        use_cache = use_cache,
        output_hidden_states=False,
        output_attentions=False,
    )
    # get new token from end of target_model output and place in input_ids
    new_tok_probs = logits_processor(target_model_logits[:, -1])
    # sample from target_model distribution
    new_token_batch = logits_processor.sample(new_tok_probs)
    input_ids[:, batch_position] = new_token_batch
    attn_mask[:, batch_position] = 1
    # move pointers one step ahead
    batch_position += 1

    # Token speed tracking (always enabled)
    start_time = time.perf_counter()
    total_tokens = B  # Count first token for each batch element

    if debug:
        console = Console()
        header = Text(f"Batch Speculative decoding | 0.00 tokens/s", style="bold magenta")
        texts = [Text(tokenizer.decode(new_token_batch[b], skip_special_tokens=True)) for b in range(B)]
        panels = [Panel(texts[i], border_style="cyan", title=f"Prompt {i+1}") for i in range(B)]
        group = Group(header, *panels)
        live = Live(group, console=console, refresh_per_second=10, transient=True)
        live.start()

    active_status = torch.full((B, ), True, dtype=torch.bool, device=device)
    active_status &= (new_token_batch != eos_tokens_id)

    # buffers to store prob outputs from draft model, redundant +1 for batched resample
    Q = torch.zeros(size=(B, gamma+1, vocab_size), device=device, dtype=target_model.dtype) # draft

    # run loop untill all sequences terminated
    while torch.any(active_status):
        # process only those sequences which are active
        active_indices = active_status.nonzero().flatten()  # Use flatten() to avoid scalar when B=1
        if batch_position >= max_total_length:
            break
        num_active = len(active_indices)
        draft_steps = min(gamma, max_total_length - batch_position - 1)
        if draft_steps <= 0:
            break
        Q.fill_(0)
        # use draft_model to generate draft_steps tokens
        draft_active = torch.full((num_active,), True, device=device)
        draft_lengths = torch.full_like(active_indices, 0, device=device)
        cache_position = torch.tensor([batch_position - 1], device=device)

        pos_id = (attn_mask[:, :batch_position].sum(dim=1)-1).unsqueeze(1)
        # Force high precision for the position calculation
        for k in range(draft_steps):
            # draft a token
            if k == 0:
                draft_output = draft_model(
                    input_ids=input_ids[:, batch_position-2:batch_position],
                    attention_mask=attn_mask[:, :batch_position],
                    use_cache=use_cache,
                    past_key_values=draft_cache,
                    cache_position=torch.tensor([batch_position - 2, batch_position - 1], device=device),
                    position_ids=(attn_mask[:, :batch_position].cumsum(dim=1)-1)[:, batch_position-2:batch_position],
                    output_hidden_states=False,
                    output_attentions=False,
                )
            else:
                draft_output = draft_model(
                    input_ids=input_ids[:, batch_position - 1 + k].unsqueeze(1),
                    attention_mask=attn_mask[:, :batch_position + k],
                    use_cache=use_cache,
                    past_key_values=draft_cache,
                    cache_position=cache_position + k,
                    position_ids=pos_id + k,
                    output_hidden_states=False,
                    output_attentions=False,
                )
            # get and process logits
            draft_logits = draft_output.logits[active_indices]
            Q[active_indices, k] = logits_processor(draft_logits[:, -1])

            # sample tokens
            new_token_batch = logits_processor.sample(Q[active_indices, k])
            new_token_batch = torch.where(draft_active, new_token_batch, eos_tokens_id)

            # find eos token and mark inactive
            input_ids[active_indices, batch_position+k] = new_token_batch
            # update attention mask to include new token
            attn_mask[active_indices, batch_position+k] = 1
            draft_lengths += draft_active
            draft_active &= (new_token_batch != eos_tokens_id)

        drafts_speculated[active_indices] += draft_lengths

        # parallel verification of drafted tokens
        # cache_position must match the number of tokens being processed
        target_cache_position = torch.arange(batch_position-1, batch_position+draft_steps, device=device)
        mask = attn_mask[:, :batch_position+draft_steps]
        pos_id = (mask.cumsum(dim=1) - 1).clamp(min=0)
        # TODO: check if mask alleviates problem of garbage values in cache 
        target_model_output = target_model(
            input_ids=input_ids[:, batch_position-1:batch_position+draft_steps],
            position_ids=pos_id[:, batch_position-1:batch_position+draft_steps],
            attention_mask=mask,
            use_cache=use_cache,
            past_key_values=target_cache,
            cache_position=target_cache_position,
            output_hidden_states=False,
            output_attentions=False,
        )
        # del target_model_output  # Free model output container

        # eps for avoiding NaN
        eps = 1e-15

        # gather target_model probs for drafted tokens
        target_model_logits = target_model_output.logits[active_indices]
        drafted_tokens=input_ids[active_indices, batch_position:batch_position+draft_steps] # [num_active, draft_steps]
        p = logits_processor(target_model_logits).clamp(min=eps) # [num_active, draft_steps+1, vocab_size]
        q = Q[active_indices, :draft_steps+1].clamp(min=eps) # [num_active, draft_steps+1, vocab_size]
        p_tok = p[:, :draft_steps].gather(dim=2, index=drafted_tokens.unsqueeze(-1)).squeeze(-1) # [num_active, draft_steps]
        q_tok = q[:, :draft_steps].gather(dim=2, index=drafted_tokens.unsqueeze(-1)).squeeze(-1) # [num_active, draft_steps]
        del target_model_logits

        # compute log ratios for rejection sampling
        log_ratio = torch.log(p_tok) - torch.log(q_tok)
        log_r = torch.empty(log_ratio.shape, device=device).uniform_().log_()

        # Find first rejection (vectorized cumulative product)
        # take cumsum of accepted_mask, the first false breaks the chain
        acceptance_status = (log_r<=log_ratio).cumprod(dim=1).to(dtype=bool)  # [num_active, draft_steps]
        num_accepted = acceptance_status.sum(dim=1)  # [num_active,]
        active_idx_rejected, rejected_idx = torch.where(~acceptance_status)
        input_ids[active_indices[active_idx_rejected], batch_position+rejected_idx] = pad_token_id
        # mask rejected tokens
        attn_mask[active_indices[active_idx_rejected], batch_position+rejected_idx] = 0

        # perform resampling of last token - bonus or extra
        # mask for those sequences in which first extra token needs re-sampling
        active_indexer = torch.arange(num_active, device=device)
        extra_token_prob = p[active_indexer, num_accepted]
        if not skip_sample_adjustment:
            q_at_rejection = q[active_indexer, num_accepted]
            extra_token_prob = torch.where(
                (num_accepted<draft_steps).unsqueeze(1),
                torch.nn.functional.relu(extra_token_prob-q_at_rejection)+eps, # max(0, p-q)
                extra_token_prob
            )
        # sample extra tokens, includes bonus tokens
        extra_tokens = logits_processor.sample(extra_token_prob)

        # find eos token and update active status, start positions
        drafted_tokens=input_ids[active_indices, batch_position:batch_position+draft_steps]
        eos_positions = torch.where(
            drafted_tokens == eos_tokens_id,
            torch.arange(draft_steps, device=device),
            max_seq_length
        ).min(dim=1).values
        has_eos = (eos_positions < num_accepted)
        # correct lengths
        accepted_draft_length = torch.where(
            has_eos,
            eos_positions+1,
            num_accepted
        )
        # update active sequences
        active_status[active_indices] &= ~( has_eos | (extra_tokens == eos_tokens_id) )
        drafts_accepted[active_indices] += accepted_draft_length
        num_blocks[active_indices] += 1

        tokens_this_iter = int((accepted_draft_length + 1).sum().item())
        total_tokens += tokens_this_iter

        # put extra/bonus token at the max position of accepted draft lengths
        # this is so that in next iteration correct logits are fetched when this token is fed as input
        batch_pos_shift = accepted_draft_length.max().item()
        input_ids[active_indices, batch_position+batch_pos_shift]=extra_tokens
        # set bonus/extra token mask
        attn_mask[active_indices, batch_position+batch_pos_shift]=1
        if debug:
            elapsed = time.perf_counter() - start_time
            tps = total_tokens / elapsed if elapsed > 0 else 0
            header.plain = f"Batch Speculative decoding | {tps:.2f} tokens/s"
            active_idx = 0
            for b in active_indices:
                a=input_ids[b, batch_position:batch_position+batch_pos_shift+1]
                decoded = tokenizer.decode(a.tolist(), skip_special_tokens=True)
                texts[b].append(decoded)
                active_idx+=1
            live.refresh()
        batch_position += batch_pos_shift + 1

        
    # Print final tokens/s and block efficiency
    elapsed = time.perf_counter() - start_time
    tps = total_tokens / elapsed if elapsed > 0 else 0
    total_blocks = int(num_blocks.sum().item())
    block_efficiency = total_tokens / total_blocks if total_blocks > 0 else 0
    print(f"\nGeneration complete: {total_tokens} tokens in {elapsed:.2f}s ({tps:.2f} tokens/s) | Block efficiency: {block_efficiency:.2f} tokens/block", flush=True)

    if debug:
        live.stop()
        texts.clear()
        del texts, group, live, console
    
    outputs = []
    acc_rates = []
    for i in range(B):
        denom = drafts_speculated[i].item() if drafts_speculated[i].item() > 0 else 1e-10
        acc_rates.append((drafts_accepted[i] / denom).item())

    return outputs, acc_rates, []



@torch.no_grad()
def speculative_generate_batch_v3(
    input_ids,
    attn_mask,
    draft_model: Module,
    target_model: Module,
    tokenizer = None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    skip_sample_adjustment: bool = False,
    collect_stats: bool = False,
    use_cache: bool = True,
    debug: bool = True ):
    """
    Batched speculative decoding with DynamicCache.
    Same padding/masking/placement as v2 but uses DynamicCache with cropping.
    """
    device = target_model.device
    vocab_size = int(target_model.config.vocab_size)
    B = input_ids.shape[0]

    # configure lengths
    max_seq_length = target_model.config.max_position_embeddings if hasattr(target_model.config, 'max_position_embeddings') \
        else (target_model.config.max_context_length if hasattr(target_model.config, 'max_context_length') else 1024)
    max_prompt_len = input_ids.shape[1]
    max_total_length = int(min(max_seq_length, max_prompt_len+max_gen_len*2))

    batch_position = input_ids.shape[1]

    pad_ids = torch.full((B, max_total_length-max_prompt_len+1), pad_token_id, dtype=input_ids.dtype)
    attn_pad = torch.zeros((B, max_total_length-max_prompt_len+1), dtype=attn_mask.dtype)
    input_ids = torch.cat([input_ids, pad_ids], dim=1).to(device)
    attn_mask = torch.cat([attn_mask, attn_pad], dim=1).to(device)

    # performance counters
    drafts_accepted = torch.zeros(B, device=device)
    drafts_speculated = torch.zeros(B, device=device)
    num_blocks = torch.zeros(B, device=device)

    # Use DynamicCache instead of StaticCache
    target_cache = DynamicCache()
    draft_cache = DynamicCache()

    # PREFILL
    mask = attn_mask[:, :batch_position]
    pos_id = (mask.cumsum(dim=1) - 1).clamp(min=0)
    target_model_output = target_model(
        input_ids = input_ids[:, :batch_position],
        attention_mask = mask,
        position_ids = pos_id,
        past_key_values = target_cache,
        use_cache = use_cache,
    )
    target_model_logits = target_model_output.logits
    target_cache = target_model_output.past_key_values

    draft_output = draft_model(
        input_ids = input_ids[:, :batch_position],
        attention_mask = mask,
        position_ids = pos_id,
        past_key_values = draft_cache,
        use_cache = use_cache,
    )
    draft_cache = draft_output.past_key_values

    # get new token from end of target_model output
    new_tok_probs = logits_processor(target_model_logits[:, -1])
    new_token_batch = logits_processor.sample(new_tok_probs)
    input_ids[:, batch_position] = new_token_batch
    attn_mask[:, batch_position] = 1
    batch_position += 1

    start_time = time.perf_counter()
    total_tokens = B

    if debug:
        console = Console()
        header = Text(f"Batch Speculative decoding v3 | 0.00 tokens/s", style="bold magenta")
        texts = [Text(tokenizer.decode(new_token_batch[b], skip_special_tokens=True)) for b in range(B)]
        panels = [Panel(texts[i], border_style="cyan", title=f"Prompt {i+1}") for i in range(B)]
        group = Group(header, *panels)
        live = Live(group, console=console, refresh_per_second=10, transient=True)
        live.start()

    active_status = torch.full((B, ), True, dtype=torch.bool, device=device)
    active_status &= (new_token_batch != eos_tokens_id)

    Q = torch.zeros(size=(B, gamma+1, vocab_size), device=device, dtype=target_model.dtype)

    while torch.any(active_status):
        active_indices = active_status.nonzero().flatten()
        if batch_position >= max_total_length:
            break
        num_active = len(active_indices)
        draft_steps = min(gamma, max_total_length - batch_position - 1)
        if draft_steps <= 0:
            break
        Q.fill_(0)

        draft_active = torch.full((num_active,), True, device=device)
        draft_lengths = torch.full_like(active_indices, 0, device=device)

        # Crop both caches to batch_position - 1
        target_cache.crop(batch_position - 1)
        draft_cache.crop(batch_position - 2)

        pos_id = (attn_mask[:, :batch_position].sum(dim=1) - 1).unsqueeze(1)

        for k in range(draft_steps):
            if k == 0:
                draft_output = draft_model(
                    input_ids=input_ids[:, batch_position-2:batch_position],
                    attention_mask=attn_mask[:, :batch_position],
                    use_cache=use_cache,
                    past_key_values=draft_cache,
                    # cache_position=torch.tensor([batch_position - 2, batch_position - 1], device=device),
                    position_ids=(attn_mask[:, :batch_position].cumsum(dim=1)-1)[:, batch_position-2:batch_position],
                    output_hidden_states=False,
                    output_attentions=False,
                )
            else:
                draft_output = draft_model(
                    input_ids=input_ids[:, batch_position - 1 + k].unsqueeze(1),
                    attention_mask=attn_mask[:, :batch_position + k],
                    position_ids=pos_id + k,
                    past_key_values=draft_cache,
                    use_cache=use_cache,
                )
            # draft_cache = draft_output.past_key_values

            draft_logits = draft_output.logits[active_indices]
            Q[active_indices, k] = logits_processor(draft_logits[:, -1])

            new_token_batch = logits_processor.sample(Q[active_indices, k])
            new_token_batch = torch.where(draft_active, new_token_batch, eos_tokens_id)

            input_ids[active_indices, batch_position+k] = new_token_batch
            attn_mask[active_indices, batch_position+k] = 1
            draft_lengths += draft_active
            draft_active &= (new_token_batch != eos_tokens_id)

        drafts_speculated[active_indices] += draft_lengths

        # Target model verification
        mask = attn_mask[:, :batch_position+draft_steps]
        pos_id = (mask.cumsum(dim=1) - 1).clamp(min=0)
        target_cache_len = target_cache.get_seq_length()

        target_model_output = target_model(
            input_ids=input_ids[:, target_cache_len:batch_position+draft_steps],
            attention_mask=mask,
            position_ids=pos_id[:, target_cache_len:batch_position+draft_steps],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = target_model_output.past_key_values

        # Extract logits for verification positions
        start_offset = batch_position - 1 - target_cache_len
        target_model_logits = target_model_output.logits[active_indices, start_offset:]

        eps = 1e-15
        drafted_tokens = input_ids[active_indices, batch_position:batch_position+draft_steps]
        p = logits_processor(target_model_logits).clamp(min=eps)
        q = Q[active_indices, :draft_steps+1].clamp(min=eps)
        p_tok = p[:, :draft_steps].gather(dim=2, index=drafted_tokens.unsqueeze(-1)).squeeze(-1)
        q_tok = q[:, :draft_steps].gather(dim=2, index=drafted_tokens.unsqueeze(-1)).squeeze(-1)

        log_ratio = torch.log(p_tok) - torch.log(q_tok)
        log_r = torch.empty(log_ratio.shape, device=device).uniform_().log_()

        acceptance_status = (log_r<=log_ratio).cumprod(dim=1).to(dtype=bool)
        num_accepted = acceptance_status.sum(dim=1)
        active_idx_rejected, rejected_idx = torch.where(~acceptance_status)
        input_ids[active_indices[active_idx_rejected], batch_position+rejected_idx] = pad_token_id
        attn_mask[active_indices[active_idx_rejected], batch_position+rejected_idx] = 0

        active_indexer = torch.arange(num_active, device=device)
        extra_token_prob = p[active_indexer, num_accepted]
        if not skip_sample_adjustment:
            q_at_rejection = q[active_indexer, num_accepted]
            extra_token_prob = torch.where(
                (num_accepted<draft_steps).unsqueeze(1),
                torch.nn.functional.relu(extra_token_prob-q_at_rejection)+eps,
                extra_token_prob
            )
        extra_tokens = logits_processor.sample(extra_token_prob)

        drafted_tokens = input_ids[active_indices, batch_position:batch_position+draft_steps]
        eos_positions = torch.where(
            drafted_tokens == eos_tokens_id,
            torch.arange(draft_steps, device=device),
            max_seq_length
        ).min(dim=1).values
        has_eos = (eos_positions < num_accepted)

        accepted_draft_length = torch.where(
            has_eos,
            eos_positions+1,
            num_accepted
        )
        active_status[active_indices] &= ~(has_eos | (extra_tokens == eos_tokens_id))
        drafts_accepted[active_indices] += accepted_draft_length
        num_blocks[active_indices] += 1

        tokens_this_iter = int((accepted_draft_length + 1).sum().item())
        total_tokens += tokens_this_iter

        # Place bonus/extra token at max position
        batch_pos_shift = accepted_draft_length.max().item()
        input_ids[active_indices, batch_position+batch_pos_shift] = extra_tokens
        attn_mask[active_indices, batch_position+batch_pos_shift] = 1

        if debug:
            elapsed = time.perf_counter() - start_time
            tps = total_tokens / elapsed if elapsed > 0 else 0
            header.plain = f"Batch Speculative decoding v3 | {tps:.2f} tokens/s"
            active_idx = 0
            for b in active_indices:
                a = input_ids[b, batch_position:batch_position+batch_pos_shift+1]
                decoded = tokenizer.decode(a.tolist(), skip_special_tokens=True)
                texts[b].append(decoded)
                active_idx += 1
            live.refresh()

        batch_position += batch_pos_shift + 1

    elapsed = time.perf_counter() - start_time
    tps = total_tokens / elapsed if elapsed > 0 else 0
    total_blocks = int(num_blocks.sum().item())
    block_efficiency = total_tokens / total_blocks if total_blocks > 0 else 0
    print(f"\nGeneration complete: {total_tokens} tokens in {elapsed:.2f}s ({tps:.2f} tokens/s) | Block efficiency: {block_efficiency:.2f} tokens/block", flush=True)

    if debug:
        live.stop()
        texts.clear()
        del texts, group, live, console

    outputs = []
    acc_rates = []
    for i in range(B):
        denom = drafts_speculated[i].item() if drafts_speculated[i].item() > 0 else 1e-10
        acc_rates.append((drafts_accepted[i] / denom).item())

    return outputs, acc_rates, []

@torch.no_grad()
def speculative_generate_batch_v4(
    input_ids,
    attn_mask,
    draft_model: Module,
    target_model: Module,
    tokenizer=None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_token_id: int | List[int] = 1,
    pad_token_id: int = 0,
    skip_sample_adjustment: bool = False,
    use_cache: bool = True,
    debug: bool = True,
):
    """
    Batched speculative decoding for variable-length (left-padded) prompts.
    Returns generated token IDs (new tokens per sample), acceptance rates, and empty stats.
    """
    device = target_model.device
    vocab_size = int(target_model.config.vocab_size)
    B = input_ids.shape[0]

    # EOS handling
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_tensor = torch.tensor(eos_token_id, device=device, dtype=input_ids.dtype)
    force_eos_id = eos_tensor[0].item()  # arbitrary choice for forcing

    # Length config
    max_seq_length = getattr(target_model.config, "max_position_embeddings",
                             getattr(target_model.config, "max_context_length", 1024))
    max_prompt_len = input_ids.shape[1]
    max_total_length = min(max_seq_length, max_prompt_len + max_gen_len * 2)

    prompt_lens = attn_mask.sum(dim=1).clone()  # initial prompt lengths

    batch_position = max_prompt_len

    # Pad to exact max_total_length
    pad_len = max_total_length - max_prompt_len
    pad_ids = torch.full((B, pad_len), pad_token_id, dtype=input_ids.dtype,)
    attn_pad = torch.zeros((B, pad_len), dtype=attn_mask.dtype)
    input_ids = torch.cat([input_ids, pad_ids], dim=1).to(device)
    attn_mask = torch.cat([attn_mask, attn_pad], dim=1).to(device)

    # Counters
    drafts_accepted = torch.zeros(B, device=device)
    drafts_speculated = torch.zeros(B, device=device)
    num_blocks = torch.zeros(B, device=device)

    # Caches
    target_cache = StaticCache(
        config=target_model.config,
        max_cache_len=max_total_length,
        device=device,
        dtype=target_model.dtype,
    )
    draft_cache = StaticCache(
        config=draft_model.config,
        max_cache_len=max_total_length,
        device=device,
        dtype=draft_model.dtype,
    )

    # PREFILL prompt
    prompt_mask = attn_mask[:, :batch_position]
    pos_id = (prompt_mask.cumsum(dim=1) - 1).clamp(min=0)
    cache_pos = torch.arange(batch_position, device=device)

    target_output = target_model(
        input_ids=input_ids[:, :batch_position],
        attention_mask=prompt_mask,
        position_ids=pos_id,
        cache_position=cache_pos,
        past_key_values=target_cache,
        use_cache=use_cache,
    )
    draft_model(
        input_ids=input_ids[:, :batch_position],
        attention_mask=prompt_mask,
        position_ids=pos_id,
        cache_position=cache_pos,
        past_key_values=draft_cache,
        use_cache=use_cache,
    )

    # First token from target
    new_tok_probs = logits_processor(target_output.logits[:, -1])
    new_token_batch = logits_processor.sample(new_tok_probs)
    input_ids[:, batch_position] = new_token_batch
    attn_mask[:, batch_position] = 1
    batch_position += 1

    # Stats
    start_time = time.perf_counter()
    total_tokens = B

    active_status = torch.full((B,), True, dtype=torch.bool, device=device)
    is_eos = (new_token_batch.unsqueeze(-1) == eos_tensor).any(-1)
    active_status &= ~is_eos

    Q = torch.zeros((B, gamma, vocab_size), device=device, dtype=target_model.dtype)

    if debug:
        console = Console()
        header = Text(f"Batch Speculative decoding v3 | 0.00 tokens/s", style="bold magenta")
        texts = [Text(tokenizer.decode(new_token_batch[b], skip_special_tokens=True)) for b in range(B)]
        panels = [Panel(texts[i], border_style="cyan", title=f"Prompt {i+1}") for i in range(B)]
        group = Group(header, *panels)
        live = Live(group, console=console, refresh_per_second=10, transient=True)
        live.start()
        # your Live setup here


    while torch.any(active_status):
        active_indices = active_status.nonzero().flatten()
        num_active = len(active_indices)
        if batch_position >= max_total_length:
            break
        draft_steps = min(gamma, max_total_length - batch_position)
        if draft_steps <= 0:
            break

        Q.fill_(0)

        draft_active = torch.ones(num_active, dtype=torch.bool, device=device)
        draft_lengths = torch.zeros(num_active, device=device)

        # Uniform single-token drafting
        for k in range(draft_steps):
            input_pos = batch_position - 1 + k
            attn_len = input_pos + 1

            input_slice = input_ids[:, input_pos].unsqueeze(1)
            cache_position = torch.tensor([input_pos], device=device)
            pos_id = (attn_mask[:, :attn_len].cumsum(dim=1) - 1).clamp(min=0)[:, -1:]

            draft_output = draft_model(
                input_ids=input_slice,
                attention_mask=attn_mask[:, :attn_len],
                position_ids=pos_id,
                cache_position=cache_position,
                past_key_values=draft_cache,
                use_cache=use_cache,
            )
            draft_logits = draft_output.logits[:, -1]
            processed = logits_processor(draft_logits)
            Q[:, k] = processed

            processed_active = processed[active_indices]
            tentative_tokens = logits_processor.sample(processed_active)
            new_tokens = torch.where(draft_active, tentative_tokens, force_eos_id)

            place_pos = batch_position + k
            input_ids[active_indices, place_pos] = new_tokens
            attn_mask[active_indices, place_pos] = 1

            draft_lengths += draft_active.float()
            is_eos_this = (new_tokens.unsqueeze(-1) == eos_tensor).any(-1)
            draft_active &= ~is_eos_this

        drafts_speculated[active_indices] += draft_lengths

        # Verification
        target_input_len = draft_steps + 1
        target_cache_pos = torch.arange(batch_position - 1, batch_position - 1 + target_input_len, device=device)
        target_attn_len = batch_position + draft_steps
        target_pos_id = (attn_mask[:, :target_attn_len].cumsum(dim=1) - 1).clamp(min=0)

        target_output = target_model(
            input_ids=input_ids[:, batch_position - 1:batch_position - 1 + target_input_len],
            attention_mask=attn_mask[:, :target_attn_len],
            position_ids=target_pos_id[:, batch_position - 1:batch_position - 1 + target_input_len],
            cache_position=target_cache_pos,
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_logits = target_output.logits[active_indices]  # [num_active, draft_steps+1, V]

        eps = 1e-15
        p = logits_processor(target_logits).clamp(min=eps)
        q = Q[active_indices, :draft_steps].clamp(min=eps)

        drafted_tokens = input_ids[active_indices, batch_position:batch_position + draft_steps]

        p_tok = p[:, :draft_steps].gather(2, drafted_tokens.unsqueeze(-1)).squeeze(-1)
        q_tok = q.gather(2, drafted_tokens.unsqueeze(-1)).squeeze(-1)

        log_ratio = torch.log(p_tok) - torch.log(q_tok)
        log_r = torch.rand_like(log_ratio).log()

        acceptance = (log_r <= log_ratio).cumprod(dim=1).bool()
        num_accepted = acceptance.sum(dim=1)

        # Mask rejected
        rej_seq, rej_k = torch.where(~acceptance)
        input_ids[active_indices[rej_seq], batch_position + rej_k] = pad_token_id
        attn_mask[active_indices[rej_seq], batch_position + rej_k] = 0

        # Bonus/extra
                # Bonus/extra sampling
        extra_prob = p[torch.arange(num_active, device=device), num_accepted]

        if not skip_sample_adjustment:
            reject_mask = (num_accepted < draft_steps)
            if reject_mask.any():
                reject_active_idx = torch.arange(num_active, device=device)[reject_mask]
                q_rej = q[reject_active_idx, num_accepted[reject_mask]]
                extra_prob[reject_mask] = torch.nn.functional.relu(
                    extra_prob[reject_mask] - q_rej
                ) + eps

        extra_tokens = logits_processor.sample(extra_prob)

        # EOS in accepted drafts
        drafted_is_eos = (drafted_tokens.unsqueeze(-1) == eos_tensor).any(-1)
        eos_pos = torch.where(drafted_is_eos, torch.arange(draft_steps, device=device), draft_steps).min(dim=1).values
        has_eos_in_draft = eos_pos < num_accepted
        accepted_len = torch.where(has_eos_in_draft, eos_pos + 1, num_accepted)

        # Update
        drafts_accepted[active_indices] += accepted_len
        num_blocks[active_indices] += 1
        is_extra_eos = (extra_tokens.unsqueeze(-1) == eos_tensor).any(-1)
        active_status[active_indices] &= ~(has_eos_in_draft | is_extra_eos)

        tokens_this_iter = (accepted_len + 1).sum().item()
        total_tokens += tokens_this_iter

        # Place extra at max accepted position
        max_shift = accepted_len.max().item()
        extra_pos = batch_position + max_shift
        input_ids[active_indices, extra_pos] = extra_tokens
        attn_mask[active_indices, extra_pos] = 1

        if debug:
            elapsed = time.perf_counter() - start_time
            tps = total_tokens / elapsed if elapsed > 0 else 0
            header.plain = f"Batch Speculative decoding v3 | {tps:.2f} tokens/s"
            active_idx = 0
            for b in active_indices:
                a = input_ids[b, batch_position:batch_position+max_shift+1]
                decoded = tokenizer.decode(a.tolist(), skip_special_tokens=True)
                texts[b].append(decoded)
                active_idx += 1
            live.refresh()
            # update Live

        batch_position += max_shift + 1

    # Final stats
    elapsed = time.perf_counter() - start_time
    tps = total_tokens / elapsed if elapsed > 0 else 0
    block_eff = total_tokens / num_blocks.sum() if num_blocks.sum() > 0 else 0
    print(f"\nGeneration complete: {total_tokens} tokens in {elapsed:.2f}s ({tps:.2f} tokens/s) "
          f"| Block efficiency: {block_eff:.2f}", flush=True)

    if debug:
        # stop Live
        live.stop()
        texts.clear()
        del texts, group, live, console

    # Extract generated
    outputs = []
    for i in range(B):
        valid_pos = torch.where(attn_mask[i])[0]
        full_real = input_ids[i, valid_pos]
        generated = full_real[prompt_lens[i]:]
        outputs.append(generated.tolist())

    acc_rates = [(drafts_accepted[i] / drafts_speculated[i]).item() if drafts_speculated[i] > 0 else 0.0
                 for i in range(B)]

    return outputs, acc_rates, []