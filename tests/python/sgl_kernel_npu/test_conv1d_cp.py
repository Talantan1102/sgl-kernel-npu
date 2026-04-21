# -*- coding: utf-8 -*-
"""Unit test for context-parallel support in causal_conv1d_fn_npu.

Simulates a 2-rank CP group in a single process by passing a FakeCPGroup
via the ``cp_group`` kwarg, and verifies that splitting a sequence across
ranks + CP state stitching reproduces the single-rank result (both output
and conv_states cache)."""
import pytest
import torch

from sgl_kernel_npu.mamba.causal_conv1d import (
    _extract_last_width,
    causal_conv1d_fn_npu,
)


class FakeCPGroup:
    def __init__(self, world_size, rank, all_tails):
        self.world_size = world_size
        self.rank_in_group = rank
        self._all_tails = all_tails

    def all_gather(self, x, dim=0):
        return self._all_tails


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.mark.parametrize(
    ("num_seqs", "seq_len", "dim", "width"),
    [
        (1, 16, 32, 4),
        (1, 32, 64, 4),
        (2, 12, 32, 4),
        (1, 24, 48, 3),
    ],
)
@torch.no_grad()
def test_conv1d_cp_matches_single_rank(num_seqs, seq_len, dim, width, device):
    torch.manual_seed(0)
    dtype = torch.float32
    assert seq_len % 2 == 0, "need even seq_len to split into 2 ranks"
    half = seq_len // 2
    state_len = width - 1

    # Single-rank reference: run full sequence, no CP.
    x_full = torch.randn(dim, num_seqs * seq_len, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype) * 0.1
    bias = torch.randn(dim, device=device, dtype=dtype) * 0.1
    qsl_full = torch.arange(
        0, num_seqs * seq_len + 1, seq_len, device=device, dtype=torch.int32
    )
    cache_indices = torch.arange(num_seqs, device=device, dtype=torch.int32)
    num_cache_lines = num_seqs + 2

    conv_states_ref = torch.zeros(
        num_cache_lines, dim, state_len, device=device, dtype=dtype
    )
    has_init_ref = torch.zeros(num_seqs, dtype=torch.bool, device=device)

    y_ref = causal_conv1d_fn_npu(
        x_full,
        weight,
        bias,
        query_start_loc=qsl_full,
        cache_indices=cache_indices,
        has_initial_state=has_init_ref,
        conv_states=conv_states_ref,
        activation="silu",
    )

    # 2-rank CP simulation: split each sequence contiguously at `half`.
    x_r0 = torch.cat(
        [x_full[:, s * seq_len : s * seq_len + half] for s in range(num_seqs)], dim=-1
    ).contiguous()
    x_r1 = torch.cat(
        [
            x_full[:, s * seq_len + half : (s + 1) * seq_len]
            for s in range(num_seqs)
        ],
        dim=-1,
    ).contiguous()
    qsl_half = torch.arange(
        0, num_seqs * half + 1, half, device=device, dtype=torch.int32
    )

    # Pre-compute what all_gather would produce on each rank.
    # Shape: (world, num_seqs, dim, state_len), matching real cp_group.all_gather.
    tail_r0 = _extract_last_width(x_r0, qsl_half, state_len)
    tail_r1 = _extract_last_width(x_r1, qsl_half, state_len)
    all_tails = torch.stack([tail_r0, tail_r1], dim=0)

    conv_states_r0 = torch.zeros_like(conv_states_ref)
    conv_states_r1 = torch.zeros_like(conv_states_ref)
    has_init_r0 = torch.zeros(num_seqs, dtype=torch.bool, device=device)
    has_init_r1 = torch.zeros(num_seqs, dtype=torch.bool, device=device)

    y_r0 = causal_conv1d_fn_npu(
        x_r0,
        weight,
        bias,
        query_start_loc=qsl_half,
        cache_indices=cache_indices,
        has_initial_state=has_init_r0,
        conv_states=conv_states_r0,
        activation="silu",
        cp_group=FakeCPGroup(2, 0, all_tails),
    )
    y_r1 = causal_conv1d_fn_npu(
        x_r1,
        weight,
        bias,
        query_start_loc=qsl_half,
        cache_indices=cache_indices,
        has_initial_state=has_init_r1,
        conv_states=conv_states_r1,
        activation="silu",
        cp_group=FakeCPGroup(2, 1, all_tails),
    )

    # Reassemble the per-rank outputs in full-sequence order.
    y_cp = torch.cat(
        [
            torch.cat([y_r0[:, s * half : (s + 1) * half], y_r1[:, s * half : (s + 1) * half]], dim=-1)
            for s in range(num_seqs)
        ],
        dim=-1,
    )

    torch.testing.assert_close(y_cp, y_ref, atol=1e-5, rtol=1e-5)
    # Both ranks' caches must hold the global sequence tail after CP.
    torch.testing.assert_close(
        conv_states_r0[cache_indices], conv_states_ref[cache_indices], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        conv_states_r1[cache_indices], conv_states_ref[cache_indices], atol=1e-6, rtol=1e-6
    )


@torch.no_grad()
def test_conv1d_cp_disabled_is_noop(device):
    """cp_group=None (or world_size == 1) must leave has_initial_state untouched."""
    torch.manual_seed(0)
    dim, width, seq_len = 16, 4, 8
    state_len = width - 1
    x = torch.randn(dim, seq_len, device=device)
    weight = torch.randn(dim, width, device=device) * 0.1
    qsl = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    cache_indices = torch.tensor([0], device=device, dtype=torch.int32)
    has_init = torch.zeros(1, dtype=torch.bool, device=device)
    conv_states = torch.randn(2, dim, state_len, device=device)
    conv_states_snapshot = conv_states.clone()

    causal_conv1d_fn_npu(
        x,
        weight,
        bias=None,
        query_start_loc=qsl,
        cache_indices=cache_indices,
        has_initial_state=has_init,
        conv_states=conv_states,
        activation="silu",
        # cp_group omitted -> CP branch must short-circuit
    )

    assert has_init.tolist() == [False]
    assert not torch.equal(conv_states, conv_states_snapshot)  # kernel did write tail
