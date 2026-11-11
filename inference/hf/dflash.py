import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature
import qwen35_tree


DFLASH_STAGE_ORDER = ("draft", "verify", "commit")


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    draft_sampling: bool = False,
    ar_tree_path: bool = False,
) -> SimpleNamespace:
    # draft_sampling (T > 0 only) samples the drafted chain from the draft's
    # distribution q and verifies it with standard speculative sampling (accept x_i with
    # probability min(1, p_i(x_i) / q_i(x_i)), else resample from max(p_i - q_i, 0)); without
    # it the chain is drafted greedily and a token is
    # accepted when the target's sample equals it (DDTree's sample-and-match).
    spec_sampling = draft_sampling and temperature > 0 and block_size > 1
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    # A hybrid (Gated-DeltaNet) target verifies the chain as a one-path tree
    # (qwen35_tree), since cropping the cache cannot roll back its recurrent state.
    # ar_tree_path runs autoregressive decoding (block_size 1) through the same kernels, so
    # that the baseline can be timed on the speculative methods' verification path as well.
    hybrid = (block_size > 1 or ar_tree_path) and qwen35_tree.is_hybrid(target)
    past_key_values_target = qwen35_tree.new_cache(target)
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DFLASH_STAGE_ORDER)
    if hybrid:
        from ddtree import compact_dynamic_cache  # lazy: ddtree imports this module

        chain_parents, chain_visibility = qwen35_tree.chain_tree(block_size)
        kernel = qwen35_tree.conv_kernel(target)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            draft_stage_start = cuda_time()
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_hidden = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )
            past_key_values_draft.crop(start)
            if spec_sampling:
                draft_q = torch.softmax(target.lm_head(draft_hidden[:, -block_size + 1 :, :]).float() / temperature, dim=-1)
                block_output_ids[:, 1:] = torch.multinomial(draft_q[0], num_samples=1).view(1, -1)
            else:
                draft_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
                block_output_ids[:, 1:] = sample(draft_logits)
            draft_stage_elapsed = cuda_time() - draft_stage_start
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()
            else:
                stage_times["draft"] += draft_stage_elapsed

        verify_stage_start = cuda_time()
        if hybrid:
            with qwen35_tree.tree_verify(chain_parents, chain_visibility, kernel, model.device) as tree_ctx:
                output = target(block_output_ids, position_ids=block_position_ids,
                                past_key_values=past_key_values_target, use_cache=True,
                                output_hidden_states=block_size > 1)
        else:
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True if block_size > 1 else False,
            )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        if spec_sampling:
            target_p = torch.softmax(output.logits[0].float() / temperature, dim=-1)  # [block, V]
            drafted = block_output_ids[0, 1:]
            p_x = target_p[:-1].gather(-1, drafted[:, None]).squeeze(-1)
            q_x = draft_q[0].gather(-1, drafted[:, None]).squeeze(-1)
            accept = torch.rand_like(p_x) < torch.clamp(p_x / q_x, max=1.0)
            acceptance_length = int(torch.cumprod(accept.long(), dim=0).sum().item())
            if acceptance_length < drafted.numel():
                residual = torch.clamp(target_p[acceptance_length] - draft_q[0, acceptance_length], min=0.0)
                if residual.sum() <= 0:
                    residual = target_p[acceptance_length]
                next_token = torch.multinomial(residual / residual.sum(), num_samples=1)
            else:
                next_token = torch.multinomial(target_p[-1], num_samples=1)
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = next_token
        else:
            posterior = sample(output.logits, temperature)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        if hybrid:
            qwen35_tree.commit(past_key_values_target, tree_ctx, list(range(acceptance_length + 1)), start,
                               compact_dynamic_cache)
        start += acceptance_length + 1
        if not hybrid:
            past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, : acceptance_length + 1, :]
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - acceptance_length - 1 : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
    )


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def empty_stage_times(stage_names: tuple[str, ...]) -> dict[str, float]:
    return {stage_name: 0.0 for stage_name in stage_names}
