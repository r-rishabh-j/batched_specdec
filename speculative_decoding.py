import gc
import torch
from torch.nn import Module
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
    batch_indexer = torch.arange(B)
    
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
    for b, prompt in enumerate(prompts):
        input_ids[b, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)

    # attention mask
    positions = torch.arange(max_total_length, device=device).unsqueeze(0)
    attn_mask = (positions < start_positions.unsqueeze(1)).to(dtype=torch.int).contiguous()

    # performance counters
    drafts_accepted = torch.zeros(B, device=device)
    drafts_speculated = torch.zeros(B, device=device)
    # for block efficiency calculation
    draft_steps = torch.zeros(B, device=device)

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
    cache_manager.prune_cache_to_min(start_positions)

    if debug:
        console = Console()
        header = Text("Batch Speculative decoding", style="bold magenta")
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
    while torch.any(active_status) and int(start_positions.max().item()) < max_total_length:
        # process only those sequences which are active
        active_indices = active_status.nonzero().flatten()  # Use flatten() to avoid scalar when B=1
        num_active = len(active_indices)
        active_indexer = torch.arange(num_active)
        max_current = start_positions[active_indices].max().item()
        draft_steps = int(min(gamma, max_total_length - max_current - 1))
        if draft_steps <= 0:
            break

        Q.fill_(0)
        # use draft_model to generate draft_steps tokens
        draft_active = torch.full((num_active,), True, device=device)
        draft_lengths = torch.full_like(active_indices, 0, device=device)

        # next positions to be filled in drafting
        next_indices = start_positions[active_indices].clone()
        # attn mask for autoregressive drafting. mask and len updated in loop
        # max_length, attn_mask = attention_mask(next_indices, margin=draft_steps)
        max_length = int(next_indices.max().item())
        draft_cache_len=cache_manager.get_draft_len()
        for k in range(draft_steps):
            # draft a token
            draft_output = draft_model(
                input_ids=input_ids[active_indices, draft_cache_len:max_length],
                attention_mask=attn_mask[active_indices, :max_length],
                use_cache=use_cache,
                past_key_values=cache_manager.draft_cache
            )
            
            # get and process logits
            draft_logits = draft_output.logits
            Q[active_indices, k] = logits_processor(draft_logits[active_indexer, next_indices-draft_cache_len-1])
            
            # sample tokens
            new_token_batch = logits_processor.sample(Q[active_indices, k])
            new_token_batch = torch.where(draft_active, new_token_batch, eos_tokens_id)

            # find eos token and mark inactive
            draft_active &= (new_token_batch != eos_tokens_id)
            input_ids[active_indices, next_indices] = new_token_batch
            draft_lengths += draft_active

            # update attention mask to include new token
            attn_mask[active_indices, next_indices] = 1

            # update lengths
            next_indices += 1
            max_length += 1
            # need to crop cache to seq with smallest length to avoid accessing invalid positions
            draft_cache_len += 1
            cache_manager.crop(draft_cache_len, which='draft')

        drafts_speculated[active_indices] += draft_lengths

        # parallel verification of drafted tokens
        target_model_cache_len = cache_manager.get_target_len()
        target_model_output = target_model(
            input_ids=input_ids[active_indices, target_model_cache_len:max_length],
            attention_mask=attn_mask[active_indices, :max_length],
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
        drafted_indices=start_positions[active_indices].unsqueeze(1) + draft_indexer # [num_active, draft_steps+1]
        drafted_tokens=input_ids[active_indices.unsqueeze(1), drafted_indices[:,:-1]] # [num_active, draft_steps]
        # -1 for left shift of model probs, use target_model_cache_len for offset
        p = logits_processor(target_model_logits[active_indexer.unsqueeze(1), drafted_indices-target_model_cache_len-1]).clamp(min=eps) # [num_active, draft_steps+1, vocab_size]
        p_tok = p[:, :draft_steps].gather(dim=2, index=drafted_tokens.unsqueeze(-1)).squeeze(-1) # [num_active, draft_steps]
        q_tok = q[:, :draft_steps].gather(dim=2, index=drafted_tokens.unsqueeze(-1)).squeeze(-1) # [num_active, draft_steps]
        del target_model_logits

        # compute log ratios for rejection sampling
        log_ratio = torch.log(p_tok) - torch.log(q_tok)
        log_r = torch.log(torch.rand(log_ratio.shape, device=device))

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
        input_ids[active_indices, start_positions[active_indices]+num_accepted]=extra_tokens
        # set bonus/extra token mask
        attn_mask[active_indices, start_positions[active_indices]+num_accepted]=1

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

        if debug:
            active_idx = 0
            for b in range(B):
                if active_status[b]:
                    decoded = tokenizer.decode(drafted_tokens[active_idx, :accepted_draft_length[active_idx]+1].detach().cpu().tolist(), skip_special_tokens=True)
                    texts[b].append(decoded)
                    active_idx+=1
            live.refresh()

        start_positions[active_indices]+=accepted_draft_length+1 # includes extra/bonus token
        
        # Evict newly inactive sequences from cache
        # active_status[active_indices] gives which of the current batch elements are still active
        keep_mask = active_status[active_indices]
        cache_manager.evict_inactive(keep_mask)
        # Prune cache sequence length to minimum across remaining active sequences
        new_active_indices = active_status.nonzero().flatten()
        if len(new_active_indices) > 0:
            cache_manager.prune_cache_to_min(start_positions[new_active_indices])
        
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
    del input_ids, Q, drafts_accepted, drafts_speculated, prompt_lens, start_positions, batch_indexer, active_status
    
    return outputs, acc_rates, []
