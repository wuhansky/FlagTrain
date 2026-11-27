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
"""Correctness tests for the MoE GEMM operator.

This file is a documented copy of ``test_moe_gemm.py``.  The code is unchanged;
only the docstrings and comments that describe the tests have been added.

WHAT IS BEING TESTED
    ``moe_gemm`` multiplies each expert-ordered block of token rows by that
    expert's weight slice and writes the result in place.  The contract has four
    parts that can break independently:

        1. routing/boundaries  -- which rows belong to which expert
        2. tiling              -- does the last partial tile of M, and of N, get
                                  computed instead of dropped or overrun
        3. epilogue            -- optional bias and the activation
        4. aliasing/layout     -- strided views, in-place output, zero-size dims

    Each test below targets one or more of those, and the parametrization is
    what makes the failure visible: an off-by-one boundary bug is invisible on
    a single expert, and a dropped bias looks like tolerance noise on one shape.

TEST MATRIX
    ``test_moe_gemm`` alone runs 4 shapes x 2 dtypes x 4 activations x
    {no bias, bias} = 64 cases.  The shapes are picked for boundary behaviour,
    not to be representative (see ``_CASES``).

ONE ORACLE
    ``_reference`` -- a plain-torch, CPU FP64 composition of the same
    contract. It runs on the CPU in double precision, so the tiled FP32
    reduction and the activation approximations of the kernel cannot agree with
    it by sharing the same device arithmetic; the comparison helpers move the
    operator's own output across afterwards.

    The oracle is deliberately not a second kernel reading: expert boundaries,
    the tail of a non-contiguous output, bias handling and the activation are
    all re-derived here from ``counts`` alone, so a shared misreading of the
    contract would have to be made twice, independently, to survive.

RUNNING
    ``pytest -m moe_gemm`` selects these tests; the module-level ``skipif``
    keeps them off backends that do not ship the fused kernel.
"""

import warnings

import pytest
import torch
import torch.nn.functional as F

import flag_train
from flag_train.runtime import torch_device_fn

from .. import accuracy_utils as utils

# The module-level marks apply to every test below: the operator is only
# meaningful on backends that ship the fused MoE GEMM, and each test is tagged
# so it can be selected with ``-m moe_gemm``.
pytestmark = [
    pytest.mark.moe_gemm,
    pytest.mark.skipif(
        flag_train.vendor_name
        not in ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend"),
        reason="Unsupported accelerator",
    ),
]

# Activation codes as the operator's ABI numbers them, mirroring the dispatch in
# the kernel. Code 6 is the identity, used where a test wants the matmul alone.
# The dict doubles as the oracle's epilogue dispatch, so a change to the ABI
# numbering has to be made here as well as in the kernel.
_ACTIVATIONS = {0: F.gelu, 1: F.relu, 2: F.silu, 6: lambda x: x}

# Shapes chosen to exercise different boundary conditions rather than to be
# representative: a single expert, leading/trailing empty experts, a ragged
# 64-way split, and K/N that are not multiples of the usual tile sizes.
#
# (counts, K, N) with ``tokens == sum(counts)``:
#   [37]                     -> one expert: the simplest case, and the one an
#                               expert-index bug can still pass
#   [0, 3, 0, 19, 1, 0, 12, 0] -> leading, interior and trailing empty experts,
#                               so first_row/last_row both land mid-batch
#   [i % 5 for i in range(64)] -> 64 experts of 0..4 rows each: mostly-empty,
#                               and every expert's block is smaller than BLOCK_M
#   [7, 35, 129]             -> ragged sizes and K=127/N=129, neither a multiple
#                               of BLOCK_K=64/BLOCK_N=64
_CASES = [
    ([37], 128, 64),
    ([0, 3, 0, 19, 1, 0, 12, 0], 129, 65),
    ([i % 5 for i in range(64)], 33, 31),
    ([7, 35, 129], 127, 129),
]

# ---------------------------------------------------------------------------
# Reference implementation and shared fixtures
#
# Exists so the operator can be checked against a plain-torch, CPU FP64
# composition of the same contract on any supported backend. The reference runs
# in double precision on the CPU, so it is an independent oracle rather than a
# second reading of the same device arithmetic.
# ---------------------------------------------------------------------------


def _inputs(counts, k, n, dtype, strided=False):
    """Build one operator invocation's operands.

    ``counts[e]`` is the number of rows routed to expert ``e``; ``rows`` is their
    cumulative sum, i.e. the expert boundaries the kernel reads. Operands are
    created on ``flag_train.device`` so the operator runs on the backend under
    test.

    With ``strided=True`` the input and output are non-contiguous views into
    larger storages: a stride test must observe the kernel's handling of the
    gaps, so the storage is filled with NaN and the untouched padding is checked
    afterwards by the caller.
    """
    # Seeded so a failure is reproducible from the reported shape alone.
    torch.manual_seed(0)
    # tokens = total routed rows; experts = length of the routing table.
    m, experts = sum(counts), len(counts)
    # Weights are per expert: [experts, K, N], contiguous, as the kernel requires.
    weights = torch.randn(
        (experts, k, n),
        dtype=dtype,
        device=flag_train.device,
    )
    # counts -> inclusive cumulative ends, which is exactly the representation
    # the operator consumes; an empty expert simply repeats the previous value.
    rows = torch.tensor(counts, dtype=torch.int64, device=flag_train.device).cumsum(0)
    # A bias is always built.  Tests that want the no-bias path pass None to the
    # operator instead of zeroing this tensor: that selects HAS_BIAS=False at
    # compile time, whereas zeros would only add zero at run time and hide a
    # bias-load bug.
    biases = torch.randn((experts, n), dtype=dtype, device=flag_train.device)
    if strided:
        # Every other row and every other column: a view whose row stride is
        # 2*k elements and column stride 2, neither matching the dense layout
        # the kernel would assume if it ignored the stride arguments.
        x = torch.randn((2 * m, 2 * k), dtype=dtype, device=flag_train.device)[::2, ::2]
        # NaN, not zeros: any element the kernel writes outside the view shows
        # up as a non-NaN value in the padding and is caught by the caller.
        output = torch.full(
            (2 * m, 2 * n), float("nan"), dtype=dtype, device=flag_train.device
        )[::2, ::2]
    else:
        x = torch.randn((m, k), dtype=dtype, device=flag_train.device)
        output = torch.full((m, n), float("nan"), dtype=dtype, device=flag_train.device)
    return output, x, weights, biases, rows


def _reference(x, weights, biases, counts, activation):
    """Reference for the MoE GEMM operator, composed from plain torch ops.

    Slices ``x`` by the ``counts``-derived expert boundaries and applies each
    expert's weight, optional bias and activation, all in CPU FP64. The double
    precision is what makes the oracle independent of the kernel's FP32 tiled
    reduction: rounding differences must stay within the shared tolerance rather
    than cancel by construction.

    The result is moved back to ``x``'s dtype and device before
    ``utils.to_reference``, so the comparison happens on whichever device the
    test framework asked the reference to run on.
    """
    # Move to the CPU and widen to FP64: the oracle must use a different
    # arithmetic path from the kernel, otherwise the two share rounding errors.
    x_cpu = x.cpu().double()
    weights_cpu = weights.cpu().double()
    result = torch.empty((x.shape[0], weights.shape[2]), dtype=torch.float64)
    # Walk the routing table instead of reading rows: boundaries are rebuilt
    # from counts here, from scratch, which is what makes the oracle independent.
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        values = torch.matmul(x_cpu[start:end], weights_cpu[expert])
        if biases is not None:
            values = values + biases[expert].cpu().double()
        # The activation is applied per expert block, in FP64, before the slice
        # is stored back into the shared result buffer.
        result[start:end] = _ACTIVATIONS[activation](values)
        start = end
    # Narrow only at the very end, so the tolerance has to absorb exactly one
    # rounding, matching the single rounding the kernel's output store performs.
    return utils.to_reference(result.to(x.dtype).to(x.device))


def _reference_with_deepspeed(x, weights, biases, counts, activation):
    """Prefer DeepSpeed's native MoE GEMM as the reference implementation.

    If the DeepSpeed native implementation is unavailable or does not support
    the current case, fall back to the independent CPU FP64 reference.
    """
    rows = torch.tensor(
        counts,
        dtype=torch.int64,
        device=x.device,
    ).cumsum(0)

    try:
        from deepspeed.inference.v2.inference_utils import ActivationType
        from deepspeed.ops.op_builder import InferenceCutlassBuilder

        # Load the native DeepSpeed CUTLASS implementation.
        inf_module = InferenceCutlassBuilder().load()
        moe_gemm_op = inf_module.moe_gemm

        # DeepSpeed writes the result into the supplied output buffer.
        reference_output = torch.empty(
            (x.shape[0], weights.shape[2]),
            dtype=x.dtype,
            device=x.device,
        )

        moe_gemm_op(
            reference_output,
            x,
            weights,
            biases,
            rows,
            ActivationType(activation),
        )

        return utils.to_reference(reference_output)

    except Exception as exc:
        warnings.warn(
            "DeepSpeed native MoE GEMM is unavailable for this case. "
            f"Falling back to the CPU FP64 reference: {exc}",
            RuntimeWarning,
        )
        return _reference(x, weights, biases, counts, activation)


def _assert_close(output, expected):
    """Compare the operator's output to the CPU FP64 oracle.

    The CPU FP64 oracle and the tiled FP32 kernel need not round exactly alike,
    so the shared floating-point defaults are kept rather than a tightened
    tolerance; the tolerances themselves are dtype-derived from ``output``.
    """
    # Tiled FP32 reduction and activation approximations need not round exactly
    # like the CPU FP64 oracle. Keep the shared floating-point defaults.
    utils.train_assert_close(output, expected, output.dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# Each test's docstring names the failure it exists to catch, which is the
# convention in this suite: a test is written against a bug, not against a
# behaviour.


@pytest.mark.parametrize("counts,k,n", _CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", [0, 1, 2, 6])
@pytest.mark.parametrize("has_bias", [False, True])
def test_moe_gemm(counts, k, n, dtype, activation, has_bias):
    """Catch wrong expert boundaries, skipped tails, bias, or activation.

    The full cross product of shapes, dtypes, activations and bias presence is
    what separates the failure modes: an off-by-one boundary is invisible on a
    single expert, and a dropped bias looks like a tolerance wobble on one
    shape only. The operator returns its output tensor, which is asserted by
    identity so a copy cannot hide the contract.
    """
    # dtype and activation are passed through to the data builder and the
    # operator; only has_bias changes the argument list.
    output, x, weights, biases, rows = _inputs(counts, k, n, dtype)
    # None selects the no-bias kernel path; see the note in ``_inputs``.
    if not has_bias:
        biases = None
    # Build the oracle first: the operator writes into ``output`` in place, so
    # any comparison computed afterwards would be against already-written data.
    expected = _reference_with_deepspeed(x, weights, biases, counts, activation)
    result = flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows, activation)
    # Identity, not value equality: the contract is that the caller's buffer is
    # filled and returned, and an internal copy would pass a value comparison.
    assert result is output
    _assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_gemm_strides(dtype):
    """Catch assuming contiguous input/output or overwriting view padding.

    Both operands are non-contiguous views: the kernel must index by strides
    rather than assume a packed layout, and must not write into the padding
    between the viewed elements, which is checked as NaN afterwards.
    """
    # Empty experts around a non-empty one, so the first expert's row offset is
    # 0 but the interesting work starts at a non-zero row.
    counts = [0, 1, 17, 0, 5]
    output, x, weights, biases, rows = _inputs(counts, 71, 35, dtype, strided=True)
    expected = _reference_with_deepspeed(x, weights, biases, counts, 6)
    # The strided output returned above is discarded and rebuilt here so that
    # the padding storage is reachable from the test body; the shapes are the
    # same as in ``_inputs``: 70 columns hold the 35 viewed ones, 2 rows hold 1.
    storage = torch.full(
        (2 * sum(counts), 70), float("nan"), dtype=dtype, device=x.device
    )
    output = storage[::2, ::2]
    # Identity activation: this test is about addressing, not the epilogue.
    flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows)
    _assert_close(output, expected)
    # The padding must be untouched: a kernel that writes a full contiguous tile
    # would fill these with values instead of leaving the NaN sentinel.
    # Odd rows are padding, and within even rows the odd columns are padding.
    assert torch.isnan(storage[1::2]).all()
    assert torch.isnan(storage[::2, 1::2]).all()


@pytest.mark.parametrize(
    "counts,k,n", [([0, 0], 17, 7), ([3, 0, 2], 0, 7), ([3, 0, 2], 17, 0)]
)
def test_moe_gemm_empty_dimensions(counts, k, n):
    """Catch invalid zero-size launches or losing the bias when K is zero.

    Empty routing, zero K and zero N reach different degenerate paths in the
    kernel; a launch with a zero grid dimension can fail outright, and with
    K == 0 the matmul contributes nothing, so the bias must still be produced.
    """
    output, x, weights, biases, rows = _inputs(counts, k, n, torch.float16)
    # SiLU on purpose: with K == 0 the output is act(bias), so the activation
    # must still run on a matrix product that contributed nothing.
    expected = _reference_with_deepspeed(x, weights, biases, counts, 2)
    result = flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows, 2)
    assert result is output
    _assert_close(output, expected)


def test_moe_gemm_graph_capture():
    """Catch hidden host synchronization; changed routing must work on replay.

    A kernel that synchronizes, allocates, or reads ``rows`` on the host cannot
    be captured; one that caches the routing at capture time would replay the
    old boundaries. ``rows`` is rewritten between capture and replay precisely
    so the second failure mode shows up as a mismatch rather than as silently
    correct output for the wrong routing.
    """
    # An empty expert is included so replay has to follow a changed boundary
    # pattern rather than a mere shift of the same block sizes.
    counts = [1, 3, 0, 2]
    output, x, weights, biases, rows = _inputs(counts, 33, 17, torch.float16)
    # The capture API has different names per backend, so probe for it rather
    # than assuming one; a backend without either skips instead of failing.
    graph_type = getattr(torch_device_fn, "CUDAGraph", None)
    if graph_type is None:
        graph_type = getattr(torch_device_fn, "NPUGraph", None)
    if graph_type is None:
        pytest.skip("The backend has no graph capture API")
    # Warm the build up on a side stream first -- a JIT-compiling or autotuning
    # op cannot be captured on its first invocation.
    # The replayed allocation must also be complete before the main stream
    # resumes, hence the wait_stream pair around the warm-up.
    stream = torch_device_fn.Stream()
    stream.wait_stream(torch_device_fn.current_stream())
    with torch_device_fn.stream(stream):
        for _ in range(3):
            flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows)
    torch_device_fn.current_stream().wait_stream(stream)
    graph = graph_type()
    with torch_device_fn.graph(graph):
        flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows)
    # New routing, same replay: the result must follow ``rows`` as it is at
    # replay time, not as it was at capture time.
    # 6 rows across 4 experts, with the first two experts empty.
    new_counts = [0, 0, 5, 1]
    rows.copy_(torch.tensor(new_counts, dtype=torch.int64, device=x.device).cumsum(0))
    graph.replay()
    # Identity activation, so the only thing under test is the routing read
    # from the mutated ``rows`` buffer.
    expected = _reference_with_deepspeed(x, weights, biases, new_counts, 6)
    _assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_moe_gemm_large_k(dtype):
    """Catch a kernel that compiles for small K but hangs or miscomputes at K=2048.

    K=2048 forces the reduction loop past the point where the small-shape tests
    stop; a mis-tiled or unguarded loop can still pass those and fail here.
    """
    # One expert with 128 rows: K=2048 means 32 iterations of the BLOCK_K=64
    # loop, which is where an off-by-one loop bound or a missing K guard breaks.
    counts = [128]
    output, x, weights, biases, rows = _inputs(counts, 2048, 2048, dtype)
    expected = _reference_with_deepspeed(x, weights, biases, counts, 2)
    flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows, 2)
    _assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_gemm_two_layers(dtype):
    """Catch per-layer state (workspace, routing) leaking between invocations.

    The second invocation's input is the first's output, so a kernel that
    reuses an internal buffer incorrectly corrupts the chain rather than a
    single step.
    """
    # First layer: [tokens, 33] x [33, 47] with an empty leading expert.
    counts = [0, 2, 17, 1]
    hidden, x, weights, biases, rows = _inputs(counts, 33, 47, dtype)
    # Second layer: [tokens, 47] x [47, 19]; the first output buffer doubles as
    # this layer's input, so the shapes are built to chain (47 -> 47).
    output, _, weights2, biases2, _ = _inputs(counts, 47, 19, dtype)
    # Run the two layers back to back without an intervening host sync, which is
    # how the real model uses the operator.
    flag_train.deepspeed.moe_gemm(hidden, x, weights, biases, rows, 2)
    flag_train.deepspeed.moe_gemm(output, hidden, weights2, biases2, rows)
    # The oracle is composed the same way, layer by layer, in FP64.
    hidden_ref = _reference_with_deepspeed(x, weights, biases, counts, 2)
    expected = _reference_with_deepspeed(hidden_ref, weights2, biases2, counts, 6)
    _assert_close(output, expected)


@pytest.mark.parametrize(
    "invalid",
    [
        "activation",
        "weight_dtype",
        "rows",
        "bias",
        "weight_shape",
        "device",
    ],
)
def test_moe_gemm_rejects_invalid_inputs(invalid):
    """Catch a missing precondition check on each individually invalid operand.

    One operand is corrupted at a time so the failure names the offending
    argument; the contract is that a bad operand raises ``ValueError`` or
    ``TypeError`` rather than producing a plausible-looking result.
    """
    # A valid baseline invocation; each branch below breaks exactly one operand
    # and leaves the rest intact, so a raise can only come from that operand.
    output, x, weights, biases, rows = _inputs([3, 1], 17, 9, torch.float16)
    activation = 6
    if invalid == "activation":
        # 3 is not one of GELU/RELU/SILU/IDENTITY: rejected before launch.
        activation = 3
    elif invalid == "weight_dtype":
        # FP32 weights are unsupported; only FP16/BF16 reach the kernel.
        weights = weights.float()
    elif invalid == "rows":
        # int32 cumulative ends cannot index a batch above 2^31 tokens.
        rows = rows.int()
    elif invalid == "bias":
        # Bias must share the input dtype, so an FP32 bias must not be added
        # through an implicit cast.
        biases = biases.float()
    elif invalid == "weight_shape":
        # Dropping the last N-1 columns makes [experts, K, N] disagree with the
        # output's N; the shapes must be cross-checked, not assumed.
        weights = weights[:, :, :1]
    elif invalid == "device":
        # A tensor left on the host would be read from memory the accelerator
        # cannot see.
        weights = weights.cpu()
    with pytest.raises((ValueError, TypeError)):
        flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows, activation)
