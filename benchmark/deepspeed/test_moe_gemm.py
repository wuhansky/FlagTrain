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
"""Performance benchmark for the MoE GEMM operator.

This file is a documented copy of ``test_moe_gemm_1.py``.  The code is
unchanged; only the docstrings and comments that describe the benchmark have
been added.

WHAT IS MEASURED
    The fused Triton kernel is timed against ``torch_op``, a plain-torch loop
    that calls ``torch.matmul`` once per expert.  The independent variable is
    the shape -- experts, tokens per expert, K, N -- because that is what
    decides the launch geometry (one program per tile per expert) and how much
    work each program has.  For each shape the harness reports kernel time and
    the speedup over the baseline, for FP16 and BF16, on the identity (code 6)
    and SiLU (code 2) epilogues.

THE BASELINE IS NOT A COMPETITOR
    The baseline is ``torch_op``, a plain-torch loop over experts evaluated in
    FP32. It is not a competitor and is not meant to be: the reported speedup
    says how far the fused kernel is from *a* correct reference rather than from
    the best implementation. The upcasting to FP32 in the baseline is
    deliberate -- it keeps the precision of the reference independent of the
    half-precision operands, so the timing measures routing/tiling overhead
    rather than the cost of a different arithmetic path.

VALIDATE BEFORE TIME
    Every input is validated against the baseline before it is handed to the
    benchmark; see ``input_fn``. A perf test that silently benchmarks a wrong
    kernel is worse than one that fails.

RUNNING
    ``pytest -m moe_gemm`` selects the benchmark; the module-level ``skipif``
    keeps it off backends without the fused kernel.
"""

import warnings
from functools import lru_cache

import pytest
import torch
import torch.nn.functional as F

import flag_train

from .. import base

# Shape is ``experts, tokens_per_expert, K, N``. The values are chosen to span the
# regimes the kernel dispatches on: one expert against many, tokens-per-expert
# small against large, and square against rectangular K/N.
#
# What each row stresses, in terms of the launch geometry in ``moe_gemm``:
#   (1, 128, 2048, 2048) -> one expert with 128 rows: a single expert axis, the
#                           closest thing to a plain GEMM, and the shape where a
#                           per-expert launch would have cost nothing anyway
#   (8, 16, 1024, 2048)  -> 8 experts x 16 rows: each expert is far below one
#                           tile, so the early-exit path and the masked tail
#                           dominate -- the worst case for launch overhead
#   (64, 1, 1024, 1024)  -> 64 experts with a single row each: one non-empty
#                           tile per expert, i.e. maximum program count for
#                           minimum useful work
#   (8, 128, 1024, 2048) -> 1024 tokens over 8 experts: the only shape here
#                           where tokens exceed 64 x experts, so it is the only
#                           one that takes the wider 64-row tile rather than the
#                           32-row minimum
_SHAPES = [
    (1, 128, 2048, 2048),
    (8, 16, 1024, 2048),
    (64, 1, 1024, 1024),
    (8, 128, 1024, 2048),
]

# ---------------------------------------------------------------------------
# Benchmark plumbing
#
# The benchmark class only carries the shape list; the operands are built in
# ``input_fn`` so the check can happen on the same values the benchmark times.
# ---------------------------------------------------------------------------


class MoEGEMMBenchmark(base.GenericBenchmark):
    # The harness reads these to know which cases to iterate and how to print
    # them; the shape description is the tuple layout of ``_SHAPES``.
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "experts, tokens_per_expert, K, N"

    def set_shapes(self, shape_file=None):
        # The shapes are fixed rather than swept: each one corresponds to a
        # dispatch regime listed above, so a different set would time different
        # code paths rather than the same paths at another size.
        # ``shape_file`` is accepted for the base-class signature but ignored:
        # this operator's interesting cases are a handful of regimes, not a
        # continuous sweep.
        self.shapes = list(_SHAPES)

    def set_more_shapes(self):
        # No extra shapes: the fixed list already covers the dispatch regimes,
        # and a larger suite would only lengthen the run without new coverage.
        return []


def torch_op(output, x, weights, biases, rows, activation):
    """PyTorch baseline for standard FP16/BF16 MoE GEMM.

    Mirrors the operator's contract: equal-sized expert slices derived from
    ``rows``, optional bias, then the activation. ``output`` is written in place
    and returned, so the baseline can be called with the same argument tuple as
    the operator.
    """
    # The benchmark shapes always split the tokens equally, so ``rows`` is
    # exactly (expert + 1) * tokens_per_expert and plain slicing reproduces the
    # routing without touching the row-end tensor at all.
    experts = weights.shape[0]
    tokens_per_expert = x.shape[0] // experts
    for expert in range(experts):
        start = expert * tokens_per_expert
        end = start + tokens_per_expert
        # Upcast both operands: the reference arithmetic is FP32 regardless of
        # the storage dtype, so the comparison is not biased by the baseline
        # being slower or less accurate than the kernel.
        result = torch.matmul(
            x[start:end].float(),
            weights[expert].float(),
        )
        if biases is not None:
            result = result + biases[expert].float()
        # Same activation numbering as the operator's ABI; code 6 (identity)
        # matches no branch and is left as the raw matmul.
        if activation == 0:
            result = F.gelu(result)
        elif activation == 1:
            result = F.relu(result)
        elif activation == 2:
            result = F.silu(result)
        # Write back in place so the caller's ``output`` ends up holding the
        # baseline result, exactly as the operator's contract says it should.
        output[start:end].copy_(result)
    return output


@lru_cache(maxsize=1)
def _load_deepspeed_moe_gemm():
    """Load DeepSpeed only once to exclude module loading time from benchmark timings."""
    try:
        from deepspeed.inference.v2.inference_utils import ActivationType
        from deepspeed.ops.op_builder import InferenceCutlassBuilder

        inf_module = InferenceCutlassBuilder().load()
        return inf_module.moe_gemm, ActivationType

    except Exception as exc:
        warnings.warn(
            "DeepSpeed native MoE GEMM is unavailable. "
            f"Falling back to the PyTorch baseline: {exc}",
            RuntimeWarning,
        )
        return None


def baseline_op(output, x, weights, biases, rows, activation):
    """Prefer DeepSpeed and fall back to the original PyTorch implementation on failure."""
    native = _load_deepspeed_moe_gemm()

    if native is not None:
        moe_gemm_op, activation_type = native

        try:
            moe_gemm_op(
                output,
                x,
                weights,
                biases,
                rows,
                activation_type(activation),
            )
            return output

        except Exception as exc:
            warnings.warn(
                "DeepSpeed native MoE GEMM failed for this case. "
                f"Falling back to the PyTorch baseline: {exc}",
                RuntimeWarning,
            )

    return torch_op(
        output,
        x,
        weights,
        biases,
        rows,
        activation,
    )


def input_fn(shape, dtype, device, activation):
    """Build one benchmark case and check it against the baseline first.

    The check runs here rather than in a separate correctness test so the timing
    measures values known to be correct: an operator that is faster because it
    computes the wrong thing would otherwise be reported as a speedup. The
    tolerances are set per dtype because BF16 loses more mantissa than FP16.

    ``expected`` is cloned before the operator writes ``output`` in place, so
    the two buffers do not alias. The same ``args`` tuple is yielded to the
    benchmark afterwards.
    """
    # Seeded per case so timings and tolerances are reproducible; the harness
    # reuses this generator for warm-up and for the timed iterations.
    torch.manual_seed(0)
    experts, tokens, k, n = shape
    # Activations: [experts * tokens, K].
    x = torch.randn(
        (experts * tokens, k),
        dtype=dtype,
        device=device,
    )
    # One weight matrix per expert: [experts, K, N], contiguous, as required.
    weights = torch.randn(
        (experts, k, n),
        dtype=dtype,
        device=device,
    )
    biases = torch.randn(
        (experts, n),
        dtype=dtype,
        device=device,
    )
    # Equal split: expert e owns rows [e*tokens, (e+1)*tokens), which is what
    # the arange-based cumulative ends encode.
    rows = torch.arange(1, experts + 1, dtype=torch.int64, device=device) * tokens
    output = torch.empty(
        (experts * tokens, n),
        dtype=dtype,
        device=device,
    )
    args = (output, x, weights, biases, rows, activation)
    # The baseline writes ``output`` in place; clone so the operator's call
    # below cannot overwrite the values being compared against.
    expected = baseline_op(*args).clone()
    flag_train.deepspeed.moe_gemm(*args)
    # BF16 has fewer mantissa bits than FP16, so its tolerance is one order of
    # magnitude looser; both are chosen to admit the kernel's tiled FP32
    # reduction and fused activation without admitting a wrong result.
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(
        output,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    # Hand the same buffers to the benchmark: they have already been through
    # one call, so no first-call compilation is charged to the first timed run.
    yield args


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# The suite has a single entry point: the harness runs the full shape x dtype
# matrix per activation, so "more tests" here would mean "more shapes", not
# "more assertions".


@pytest.mark.moe_gemm
@pytest.mark.skipif(
    flag_train.vendor_name
    not in ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend"),
    reason="Unsupported accelerator",
)
@pytest.mark.parametrize("activation", [6, 2], ids=["identity", "silu"])
def test_moe_gemm_perf(activation):
    """Catch a kernel that is fast on the identity path but not on SiLU.

    Identity (code 6) and SiLU (code 2) exercise different epilogues; timing
    only the identity would leave a slow fused activation unmeasured. Each case
    is validated by ``input_fn`` before it is timed.
    """

    # ``input_fn`` needs the activation too, but the harness calls the input
    # factory with only (shape, dtype, device); binding it here instead of
    # threading it through the harness keeps the closure flat.
    def make_inputs(shape, dtype, device):
        yield from input_fn(shape, dtype, device, activation)

    # The harness owns warm-up, iteration counting and timing; it needs only the
    # input factory, the two callables to compare, and the dtypes to run.
    bench = MoEGEMMBenchmark(
        input_fn=make_inputs,
        op_name="moe_gemm",
        torch_op=baseline_op,
        dtypes=[torch.float16, torch.bfloat16],
    )
    # ``set_train`` registers the implementation under test; the same callable
    # was already exercised by ``input_fn``'s validation call.
    bench.set_train(flag_train.deepspeed.moe_gemm)
    # Runs shapes x dtypes x activations and reports the timing table.
    bench.run()
