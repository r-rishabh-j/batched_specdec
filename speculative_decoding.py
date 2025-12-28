import torch
from torch.nn import Module
from .logits_processor import LogitsProcessor, GreedyProcessor
from . import printing
from typing import List, Tuple
from torch.nn import functional as F

def prune_cache(cache, cache_len):
    if cache is None:
        return None
    cache.crop(cache_len)
    return cache

def prob_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Max function.
        x: input tensor.
    Returns:
        tensor norm(max(0, x)).
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum

@torch.no_grad()
def speculative_generate_batch(
    batch_inputs: List[List[int]],
    drafter: Module,
    target: Module,
    tokenizer = None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    skip_sample_adjustment: bool = False,
    collect_stats: bool = False,
    debug: bool = True ):
    """
    Batched speculative decoding for variable-length prompts.
    This implementation ignores kv caching and instead relies on attention masks.
    Returns generated sequences, per-sample acceptance rates, and optional stats.
    """
    B = len(batch_inputs)
    batch_indexer = torch.arange(B)
    # fix our constants
    device = target.device
    max_seq_length = target.config.max_position_embeddings if hasattr(target.config, 'max_position_embeddings') else (target.config.max_context_length if hasattr(target.config, 'max_context_length') else 1024)
    prompt_lens = torch.tensor([len(prompt) for prompt in batch_inputs], device=device)
    max_prompt_len = int(prompt_lens.max().item())
    max_total_length = int(min(max_seq_length, max_prompt_len+max_gen_len))

    start_positions = prompt_lens.clone() # hold start pointers for next token fill for each batch element
    # create input tensor for the model in input_ids
    input_ids = torch.full((B, max_total_length+1), pad_token_id, dtype=torch.long, device=device)
    for b, prompt in enumerate(batch_inputs):
        input_ids[b,:len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)

    drafts_accepted = torch.zeros(B, device=device)
    drafts_speculated = torch.zeros(B, device=device)

    def attention_mask(start_positions: torch.Tensor):
        """
        create an attention mask for batch of prompt of length given by start_positions
        """
        max_prompt_len = start_positions.max().item()
        # uses broadcasting here to expand positions to batch dim then the length dim
        positions = torch.arange(max_prompt_len, device=device).unsqueeze(0)
        mask = positions < (start_positions).unsqueeze(1)
        return max_prompt_len, mask

    # generate the first token from target
    max_prompt_len, attn_mask = attention_mask(start_positions)
    target_logits = target(
        input_ids = input_ids[:, :max_prompt_len],
        attention_mask = attn_mask,
        use_cache=False
    ).logits
    # get new token from end of output Mp and put in input_ids
    new_tok_probs = logits_processor(target_logits[batch_indexer, start_positions-1])
    # sample from target distribution
    new_token_batch = logits_processor.sample(new_tok_probs)
    input_ids[batch_indexer, start_positions]=new_token_batch
    # move pointers one step ahead
    start_positions+=1

    # store which prompts are now terminated, either by exceeding max len 
    # or that eos token has been reached
    active_status = start_positions < max_total_length
    active_status &= (new_token_batch != eos_tokens_id)

    vocab_size = int(target.config.vocab_size)

    # buffers to store prob outputs from draft and target, redundant +1 for batching
    Q = torch.zeros(size=(B, gamma+1, vocab_size), device=device, dtype=target.dtype) # draft

    # run loop untill all sequences terminated
    while torch.any(active_status) and int(start_positions.max().item()) < max_total_length:
        # process only those sequences which are active
        active_indices = active_status.nonzero().squeeze(-1)
        num_active = len(active_indices)
        max_current = start_positions[active_indices].max().item()
        draft_steps = int(min(gamma, max_total_length - max_current - 1))
        if draft_steps <= 0:
            break

        Q.fill_(0)
        # use drafter to generate draft_steps tokens
        # draft_active = torch.clone(active_indices)
        for k in range(draft_steps):
            gen_indices = start_positions[active_indices]+k
            max_length, attn_mask = attention_mask(start_positions[active_indices] + k) # add k to include draft step
            draft_logits = drafter(
                input_ids=input_ids[active_indices, :max_length],
                attn_mask=attn_mask,
                use_cache=False
            ).logits
            # get the last logit batch from model output
            Q[active_indices, k] = logits_processor(draft_logits[torch.arange(num_active), gen_indices-1]) # -1 for left shift
            # sample from this batch of next tokens
            # new_token_batch = logits_processor.sample(Q[active_indices, k])
            new_token_batch = logits_processor.sample(Q[active_indices, k])
            # new_token_batch = torch.where(draft_active, new_token_batch, pad_token_id)
            # write to input IDs
            input_ids[active_indices, gen_indices]=new_token_batch

        drafts_speculated[active_indices] += draft_steps

        # run the generated drafts through the target model
        # the drafts are contained in input_ids
        max_length, verification_mask = attention_mask(start_positions[active_indices] + draft_steps) # add draft steps to include the draft

        target_logits = target(
            input_ids=input_ids[active_indices, :max_length],
            attn_mask=verification_mask,
            use_cache=False
        ).logits

        eps = 1e-12

        q = Q[active_indices, :draft_steps+1].clamp(min=eps)
        # create an index array to gather draft probs and tokens. Include an extra position for bonus token
        drafted_indices = start_positions[active_indices].unsqueeze(1) + torch.arange(draft_steps+1, device=device)
        drafted_tokens=input_ids[active_indices.unsqueeze(1), drafted_indices[:,:-1]]
        # expand to match vocab dim for collecting target probs, -1 for left shift of model probs
        p = logits_processor(target_logits[torch.arange(num_active).unsqueeze(1), drafted_indices-1]).clamp(min=eps)
        p_tok = p[torch.arange(num_active).unsqueeze(1), torch.arange(draft_steps).unsqueeze(0), drafted_tokens]
        q_tok = q[torch.arange(num_active).unsqueeze(1), torch.arange(draft_steps).unsqueeze(0), drafted_tokens]

        log_ratio = torch.log(p_tok) - torch.log(q_tok)
        log_r = torch.log(torch.rand(log_ratio.shape, device=device))
        # Find first rejection (vectorized cumulative product)
        # take cumsum of accepted_mask, the first false breaks the chain
        acceptance_status = (log_r<=log_ratio).cumprod(dim=1).to(dtype=bool)  # [num_active, draft_steps+1]
        num_accepted = acceptance_status.sum(dim=1)  # [num_active]
        drafts_accepted[active_indices] += num_accepted

        active_idx_rejected, rejected_idx = torch.where(~acceptance_status)
        rejected_positions = drafted_indices[active_idx_rejected, rejected_idx]
        input_ids[active_indices[active_idx_rejected], rejected_positions] = pad_token_id

        # perform resampling of last token - bonus or extra
        # mask for those sequences in which first extra token needs re-sampling
        extra_token_prob = p[torch.arange(num_active), num_accepted]
        if not skip_sample_adjustment:
            q_at_rejection = q[torch.arange(num_active), num_accepted]
            extra_token_prob = torch.where((num_accepted<draft_steps).unsqueeze(1), extra_token_prob, prob_norm(extra_token_prob-q_at_rejection))
        # sample extra tokens, includes bonus tokens
        extra_tokens = logits_processor.sample(extra_token_prob)
        input_ids[active_indices, start_positions[active_indices]+num_accepted]=extra_tokens

        # find eos token and update active status, start positions
        drafted_tokens=input_ids[active_indices.unsqueeze(1), drafted_indices]
        eos_hits = torch.isin(drafted_tokens, eos_tokens_id)
        eos_positions = torch.where(eos_hits, torch.arange(draft_steps+1, device=device).unsqueeze(0), max_seq_length).min(dim=1).values
        has_eos = eos_positions <= draft_steps
        # correct lengths
        start_positions[active_indices] = torch.where(
            has_eos,
            start_positions[active_indices]+eos_positions+1, 
            start_positions[active_indices]+num_accepted+1
        )
        active_status[active_indices] &= ~has_eos

        
    outputs = []
    acc_rates = []
    for i in range(B):
        end = min(int(start_positions[i].item()), max_total_length)
        outputs.append(input_ids[i, prompt_lens[i]:end].tolist())
        denom = drafts_speculated[i].item() if drafts_speculated[i].item() > 0 else 1e-10
        acc_rates.append((drafts_accepted[i] / denom).item())
    
    return outputs, acc_rates, []


@torch.no_grad()
def speculative_generate(
    inputs: List[int],
    drafter: Module,
    target: Module,
    tokenizer = None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = True,
    first_target: bool = True,
    debug: bool = False,
) -> Tuple[List[int], float]:
    """
    Generate text sequence using the speculative decoding algorithm.
    Implementation of Speculative Decoding. (https://arxiv.org/pdf/2211.17192.pdf)
    
    Args:
        inputs (List[int]): input sequence of batch size 1.
        drafter (Module): drafter model.
        target (Module): target model.
        tokenizer: tokenizer (used for debugging).
        gamma (int): number of drafts generated by the drafter at each step.
        logits_processor (LogitsProcessor): logits processor for sampling.
        max_gen_len (int): maximum length of the generated sequence.
        eos_tokens_id (int or List[int]): end token id (could be multiple).
        pad_token_id (int): pad token id.
        use_cache (bool): whether to use cache.
        skip_sample_adjustment (bool): whether to skip the sample adjustment step when some drafts are discarded.
        first_target (bool): whether to run the target model before the speculative algorithm.
        debug (bool): debug mode.
    
    Returns:
        List[int]: generated sequence.
        float: acceptance rate (number of accepted drafts divided by the number of total drafts).
        
    Note: This generation methods only works for decoder-only models.
    Note bis: The drafter and target models should output the same logits shape.
    Note ter: NgramModels are currently not supported.
    """
    
    drafter_cache, target_cache = None, None

    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)
    
    drafts_accepted, drafts_speculated = .0, .0
    
    vocab_size = target.config.vocab_size    
        
    # prepare input tensor
    prompt_len = len(inputs)
    max_seq_length = target.config.max_position_embeddings if hasattr(target.config, 'max_position_embeddings') else (target.config.max_context_length if hasattr(target.config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=target.device)
    
    current_position = prompt_len
    
    if first_target:
        # run the target model before the speculative algorithm. Allows to prefill the kvcache and get a first token.
        Mp = target(
            input_ids=input_ids[..., :current_position],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        p_p = logits_processor(Mp.logits[..., -1, :])
        t = logits_processor.sample(p_p)
        input_ids[0, current_position] = t
        current_position += 1
        
        if torch.isin(t, stop_tokens):
            if debug:
                printing.end_token_found(0)
            return input_ids[0, prompt_len:current_position].tolist(), 0
        
        if debug:
            printing.initial_step(t, tokenizer)


    while current_position < total_len:
        corrected_gamma = min(gamma, total_len - current_position - 1)
        q = torch.zeros((1, corrected_gamma, vocab_size), device=target.device)
        
        input_ids = input_ids.to(drafter.device)
        
        # generate gamma drafts
        for k in range(corrected_gamma):
            Mq = drafter(
                input_ids=input_ids[..., :current_position + k],
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Mq.past_key_values
            
            draft_logits = Mq.logits[..., -1, :]
            draft_probs = logits_processor(draft_logits)
            q[0, k] = draft_probs.to(target.device)
            xi = logits_processor.sample(draft_probs)
            input_ids[0, current_position + k] = xi
        drafts_speculated += corrected_gamma
        input_ids = input_ids.to(target.device)
        
        # run target model on drafts and get logits of the previous tokens plus one more token
        Mp = target(
            input_ids=input_ids[..., :current_position + corrected_gamma],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        draft_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :] # [1, corrected_gamma, vocab_size]
        p = logits_processor(draft_logits) # [1, gamma, vocab_size]
        
        # compute the last accepted draft position (rejection sampling)
        r = torch.rand(corrected_gamma, device=target.device)
        fractions = p / q
        n = corrected_gamma
        for i in range(corrected_gamma):
            if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
                n = i
                break
        
        drafts_accepted += n
        
        # check if the end token is in the drafts
        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:current_position + n], stop_tokens))
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[0, 1].item()
            if debug:
                printing.end_token_found(stop_location)
            return input_ids[0, prompt_len:current_position + stop_location + 1].tolist(), drafts_accepted / drafts_speculated

        # adjust the distribution from Mp
        if n == corrected_gamma:
            p_p = Mp.logits[..., current_position + corrected_gamma - 1, :]
            p_p = logits_processor(p_p)
        else:
            # prune the cache
            if use_cache:
                drafter_cache = prune_cache(drafter_cache, current_position + n)
                target_cache = prune_cache(target_cache, current_position + n +1)

            if not skip_sample_adjustment:
                p_p = prob_norm(p[..., n, :] - q[0, n, :])
            else:
                p_p = p[..., n, :]
        x = logits_processor.sample(p_p)
        
        if debug:
            generated = input_ids.clone().detach()
            
        input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id
        input_ids[0, current_position + n] = x
        
        if debug:
            printing.speculative_step(tokenizer, generated, input_ids, n, prompt_len, current_position, corrected_gamma)
            
        current_position += n + 1
        
        if torch.isin(x, stop_tokens):
            if debug:
                printing.end_token_found(n)
            return input_ids[0, prompt_len:current_position].tolist(), drafts_accepted / drafts_speculated
    
    return input_ids[0, prompt_len:].tolist(), drafts_accepted / drafts_speculated
