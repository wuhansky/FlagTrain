# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Expert-grouped MoE GEMM: the forward matrix multiplication of a MoE layer.

This file is a documented copy of ``moe_gemm.py``.  The code is unchanged; only
the docstrings and comments that describe the algorithm have been added.

THE PROBLEM IT SOLVES
    A Mixture-of-Experts layer splits the token batch across ``E`` experts.
    Routing (which token goes to which expert, and how the top-k outputs are
    combined) happens in the caller.  By the time this operator runs, the token
    rows have already been physically sorted by expert, and the vector
    ``total_rows_before_expert`` records where each expert's block of rows ends.
    So the whole layer collapses into one batched multiplication:

        out[r, :] = act( x[r, :] @ W[expert(r), :, :] + b[expert(r), :] )

    where ``expert(r)`` is the expert that owns row ``r``.

DATA LAYOUT
    ``hidden_states``  [tokens, K]                     fp16 / bf16
    ``output``         [tokens, N]                     same dtype, written in place
    ``weights``        [experts, K, N]                 fp16 / bf16, contiguous
    ``biases``         [experts, N]                    optional
    ``total_rows_before_expert``  [experts]            int64, inclusive row ends

    Because ``weights`` is contiguous, expert ``e``'s matrix starts at element
    ``e * K * N``, so no per-expert pointer table is needed.  Expert ``e`` owns
    rows ``[row_ends[e-1], row_ends[e])``; expert 0 starts at row 0.  Repeated
    ends mean an expert received no tokens and does no work at all.

WHY NO ROUTING AND NO COMBINE
    Keeping this operator a pure GEMM keeps it reusable: the router changes
    between model families, the grouped GEMM does not.  It also means the
    kernel never writes to the same row twice, so no atomic accumulation or
    second reduction pass is required.

FORWARD ONLY
    There is no autograd support: the operator writes into a caller-owned
    buffer and returns it by identity, which is not something an autograd
    Function can track safely.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train import runtime
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

__all__ = ["moe_gemm"]


@libentry()
@triton.jit
def moe_gemm_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    row_ends_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_IM: tl.constexpr,
    STRIDE_IK: tl.constexpr,
    STRIDE_OM: tl.constexpr,
    STRIDE_ON: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one ``BLOCK_M x BLOCK_N`` output tile of one expert.

    GRID
        ``grid = (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), num_experts)``.  The
        linear tile index on axis 0 is split into a row tile ``tile_m`` and a
        column tile ``tile_n``; the expert index is axis 1.

        Why a rectangular (tile, expert) grid instead of one flattened list of
        (tile, expert) work items: every program can find its own token range
        from ``row_ends_ptr`` with two scalar loads.  A flattened list would
        need an integer prefix scan across lanes to map work item -> expert,
        which vendor compilers lower unevenly, and the extra index arithmetic
        buys nothing here.

    WORK SKIPPING
        A tile whose row offset lies past the expert's row count exits
        immediately.  That is what makes empty and ragged experts cheap: a
        zero-row expert launches programs that read two scalars and return,
        and the row mask trims the partially filled last tile of an expert.

    K LOOP
        The reduction over ``K`` is tiled by ``BLOCK_K`` and accumulated in
        FP32.  Inputs keep their storage dtype so the tensor cores can consume
        them, but every partial product is widened before the add, so the
        accumulation error does not grow with K in half precision.

    BIAS AND ACTIVATION
        Both are applied to the finished FP32 accumulator, before the single
        masked store.  Each output element is therefore written exactly once,
        and the store mask is the intersection of the row and column masks,
        which is also what allows N to be any value, not only a multiple of
        ``BLOCK_N``.
    """
    tile = tle.program_id(0)
    expert = tle.program_id(1)

    # Map the linear tile index to the row and column tile indices.
    tiles_n = tl.cdiv(N, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile % tiles_n

    # Obtain the token row range assigned to the current expert.
    # The cumulative row ends are inclusive, so expert 0 must be treated as
    # starting at 0 rather than reading row_ends_ptr[-1]; the load is predicated
    # on that and supplies the 0.0 default instead of relying on masking.
    first_row = tl.load(row_ends_ptr + expert - 1, expert > 0, other=0)
    last_row = tl.load(row_ends_ptr + expert)

    # Skip tiles that do not contain any rows assigned to this expert.
    # Empty experts therefore cost one program per tile that returns without a
    # single load from input, weight or output memory.
    if tile_m * BLOCK_M < last_row - first_row:
        rows = first_row + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        # row_mask folds together the two ways a row can be out of range: past
        # the end of the expert's block (ragged split) or past the end of the
        # whole token batch (M is not a multiple of BLOCK_M).
        row_mask = (rows < last_row) & (rows < M)
        col_mask = cols < N
        # Widen the expert id before it enters the weight address: expert * K * N
        # can exceed 2^31 for a large expert count with a large matrix, and the
        # arithmetic would silently wrap in 32 bits.
        expert_offset = expert.to(tl.int64)
        # Accumulate matrix multiplication results in FP32.
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for block_k in range(tl.cdiv(K, BLOCK_K)):
            ks = block_k * BLOCK_K + offsets_k
            inputs = tl.load(
                input_ptr + rows[:, None] * STRIDE_IM + ks[None, :] * STRIDE_IK,
                row_mask[:, None] & (ks[None, :] < K),
                other=0,
            )
            # weights is [experts, K, N] and contiguous, so this expert's slice
            # starts at expert * K * N and row ks of it is ks * N inside it.
            weight_offsets = (
                expert_offset * K * N + ks[:, None].to(tl.int64) * N + cols[None, :]
            )
            # Casting the weight tile to the activation dtype keeps the dot in
            # the storage dtype (tensor cores) while the accumulator stays FP32.
            weights = tl.load(
                weight_ptr + weight_offsets,
                (ks[:, None] < K) & col_mask[None, :],
                other=0,
            ).to(inputs.dtype)
            accumulator += tl.dot(inputs, weights)

        if HAS_BIAS:
            # One bias row per expert, added broadcast over the tile's rows.
            biases = tl.load(bias_ptr + expert_offset * N + cols, col_mask, other=0).to(
                tl.float32
            )
            accumulator += biases[None, :]
        # ACTIVATION mirrors DeepSpeed's ActivationType numbering; code 6
        # (identity) matches no branch and leaves the accumulator untouched, so
        # the "no activation" case is a compile-time no-op rather than a runtime
        # test. GELU here is the erf formulation, not the tanh approximation.
        if ACTIVATION == 0:
            accumulator = (
                0.5
                * accumulator
                * (1.0 + tl_extra_shim.erf(accumulator * 0.7071067811865476))
            )
        elif ACTIVATION == 1:
            accumulator = tl.maximum(accumulator, 0.0)
        elif ACTIVATION == 2:
            accumulator = accumulator * tl.sigmoid(accumulator)

        # The store is masked, so rows past the expert's range and columns past
        # N are never touched; the ownership of each output row by exactly one
        # expert is what makes an overwrite-free in-place write safe.
        output_offsets = rows[:, None] * STRIDE_OM + cols[None, :] * STRIDE_ON
        tl.store(
            output_ptr + output_offsets,
            accumulator,
            row_mask[:, None] & col_mask[None, :],
        )


def _validate_inputs(
    output,
    hidden_states,
    weights,
    biases,
    total_rows_before_expert,
    activation,
):
    """Check the operands and return ``(M, N, K, num_experts)``.

    Every check here is metadata only -- dtype, shape, device, contiguity -- so
    validation never reads device memory and therefore never synchronizes the
    accelerator.  The *values* of ``total_rows_before_expert`` (nondecreasing,
    within ``[0, M]``, ending at ``M``) are a precondition of the operator, not
    a runtime check, for exactly the same reason: reading them on the host
    would stall the pipeline for a tiny operator.

    Strides are deliberately left unchecked: inputs and outputs may be
    non-overlapping views, and the kernel receives their strides explicitly.

    Args:
        output (Tensor): destination, shape ``[M, N]``.
        hidden_states (Tensor): activations, shape ``[M, K]``.
        weights (Tensor): expert matrices, shape ``[experts, K, N]``.
        biases (Tensor or None): optional per-expert bias, shape ``[experts, N]``.
        total_rows_before_expert (Tensor): inclusive cumulative row ends.
        activation (int): one of 0 (GELU), 1 (RELU), 2 (SILU), 6 (identity).

    Returns:
        Tuple[int, int, int, int]: ``M``, ``N``, ``K`` and the expert count.

    Raises:
        RuntimeError: the accelerator vendor has no compiled kernel here.
        ValueError, TypeError: shape, dtype, device or contiguity mismatch.
    """
    supported_vendors = ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend")
    if runtime.device.vendor_name not in supported_vendors:
        raise RuntimeError("moe_gemm does not support this accelerator backend")
    if activation not in (0, 1, 2, 6):
        raise ValueError("activation must be GELU=0, RELU=1, SILU=2, or IDENTITY=6")
    if hidden_states.ndim != 2 or output.ndim != 2 or weights.ndim != 3:
        raise ValueError("input/output must be 2D and weights must be 3D")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("hidden_states must have dtype float16 or bfloat16")
    if output.dtype != hidden_states.dtype:
        raise TypeError("output and hidden_states must have the same dtype")
    if weights.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("weights must have dtype float16 or bfloat16")

    m, k = hidden_states.shape
    n = output.shape[1]
    num_experts = weights.shape[0]
    if num_experts == 0:
        raise ValueError("at least one expert is required")
    if output.shape[0] != m or weights.shape[1:] != (k, n):
        raise ValueError("input, output, and weight dimensions do not match")
    # int64 cumulative ends are required because M can exceed 2^31 tokens at
    # large batch sizes; a narrower dtype would cap the batch, not just the index.
    if (
        total_rows_before_expert.shape != (num_experts,)
        or total_rows_before_expert.dtype != torch.int64
    ):
        raise ValueError("row ends must have shape [experts] and dtype int64")
    if biases is not None:
        if biases.shape != (num_experts, n) or biases.dtype != hidden_states.dtype:
            raise ValueError("biases must have shape [experts, N] and the input dtype")

    tensors = [output, hidden_states, weights, total_rows_before_expert]
    if biases is not None:
        tensors.append(biases)
    if hidden_states.device.type != runtime.device.name or any(
        t.device != hidden_states.device for t in tensors
    ):
        raise ValueError("all tensors must be on the same accelerator device")
    # Only the tensors the kernel walks linearly with scalar strides have to be
    # contiguous; output and input may be views and carry their strides along.
    contiguous_tensors = [weights, total_rows_before_expert]
    if biases is not None:
        contiguous_tensors.append(biases)
    if any(not t.is_contiguous() for t in contiguous_tensors):
        raise ValueError("weights, biases, and row ends must be contiguous")
    return m, n, k, num_experts


def moe_gemm(
    output,
    hidden_states,
    weights,
    biases,
    total_rows_before_expert,
    activation=6,
):
    """Compute the expert-grouped MoE GEMM, writing into ``output``.

    The operator is the grouped matrix multiplication of a Mixture-of-Experts
    layer, after routing: ``hidden_states`` rows are already ordered by expert
    and ``total_rows_before_expert`` marks each expert's block of rows, so each
    row is multiplied by its own expert's weight slice and nothing else.  It
    does not route tokens and does not combine the top-k expert outputs; both
    stay with the caller.

    All of the arithmetic happens in one kernel launch: ``num_experts`` grids
    are folded into one grid axis, so experts with few rows do not each cost a
    launch, and no intermediate tensor is allocated.  Precision is FP16/BF16
    for the inputs and the stored result, FP32 for the accumulation, the bias
    and the activation.

    Args:
        output (Tensor): output tensor with the same dtype as ``hidden_states``,
            shape ``[tokens, N]``; written in place and returned by identity.
        hidden_states (Tensor): input activations, dtype ``torch.float16`` or
            ``torch.bfloat16``, shape ``[tokens, K]``.  Rows must be sorted by
            expert.
        weights (Tensor): per-expert weight matrices, same dtype as the input,
            shape ``[experts, K, N]``, contiguous.
        biases (Tensor or None): optional bias, shape ``[experts, N]``, same
            dtype as ``hidden_states``.
        total_rows_before_expert (Tensor): inclusive cumulative row ends,
            shape ``[experts]``, dtype ``torch.int64``.  Values must be
            nondecreasing, in ``[0, tokens]``, with the last value equal to
            ``tokens``; repeated ends denote empty experts.  These value
            preconditions are not checked on the host, so calling the operator
            does not synchronize the device.
        activation (int): activation identifier, ``GELU=0`` (erf form),
            ``RELU=1``, ``SILU=2``, ``IDENTITY=6`` (default: no activation).

    Returns:
        Tensor: ``output``, holding the result of one grouped GEMM followed by
            the optional bias and activation.

    Input and output may be non-overlapping strided views; all tensors must
    live on the same accelerator device.  This is a forward-only operator:
    ``output`` must not overlap any input and no autograd graph is built.
    """
    logger.debug("TRAIN MOE_GEMM")
    m, n, k, num_experts = _validate_inputs(
        output,
        hidden_states,
        weights,
        biases,
        total_rows_before_expert,
        activation,
    )
    if m == 0 or n == 0:
        return output

    # A 32-row minimum keeps matrix fragments valid on every supported backend
    # (Ascend included); the wider 64-row tile is only used once there are
    # enough rows per expert to fill it, since a half-empty tile wastes half the
    # tensor-core work while a 32-row tile still fills the pipeline.
    block_m = 32 if m <= 64 * num_experts else 64
    # One program per (row/column tile, expert) pair: the tile count along M and
    # N is flattened onto axis 0, the expert becomes axis 1.  BLOCK_N and
    # BLOCK_K are fixed at 64, the shape the shared-memory footprint and the
    # pipelining depth are tuned for.
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, 64), num_experts)
    with torch_device_fn.device(hidden_states.device):
        moe_gemm_kernel[grid](
            output,
            hidden_states,
            weights,
            biases,
            total_rows_before_expert,
            m,
            n,
            k,
            hidden_states.stride(0),
            hidden_states.stride(1),
            output.stride(0),
            output.stride(1),
            ACTIVATION=int(activation),
            HAS_BIAS=biases is not None,
            BLOCK_M=block_m,
            BLOCK_N=64,
            BLOCK_K=64,
            # num_stages=3 double-buffers the K loop: the next input and weight
            # tiles are in flight while the current pair feeds tl.dot, which is
            # what hides the global-memory latency of the reduction.
            num_warps=4,
            num_stages=3,
        )
    return output
