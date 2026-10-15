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
"""Iluvatar (IxUDA) tuned blocked flash attention.

The generic implementation is ``flag_train.deepspeed.blocked_flash`` and is what
every backend without an override runs. This one exists because three of that
launcher's assumptions about the device are false on IxUDA, and all three are
failures rather than preferences -- the tests, not the benchmark, are what
settled them. The launcher is not tuned differently here; it is rewritten against
a device the generic one rejects, sizes for, and miscompiles for:

* **The generic launcher rejects the device outright.** It mirrors the reference
  ``TORCH_CHECK`` for compute capability 8.0 or newer, which is how a CUDA-only
  launcher spells "this needs tensor cores". IxUDA presents itself as
  ``torch.cuda`` and answers that question in NVIDIA's terms -- a BI-V150 returns
  ``(7, 1)``, which on an NVIDIA part means Volta, no tensor cores -- so the
  check rejects a device its own kernels run on. The number describes an
  architecture this part is not, which is worse than no answer because it reads
  as one; the launcher here therefore hands the generic check a device that will
  not say, which is the case that check already provides for. See
  ``_capability_gate``, the one place in this module that reaches into the
  generic one.

* **The generic tiling does not fit in this device's shared memory.** It assumes
  the 163 KB an A100 has. A BI-V150 has 128 KB, and the generic tiling at
  ``head_size=256`` asks for 196608 bytes, so ``test_head_size[256]`` fails with
  ``OutOfResources`` from inside the compiler -- a message about shared memory
  that names neither the query tile nor the parameter that decides it. The tiling
  here is checked against the device's limit, with the compiler's own figure for
  it, before there is a launch to refuse.

* **The generic kernel's two-loop KV walk is miscompiled by this part's
  Triton.** The generic kernel peels the leading cache blocks that neither the
  causal mask nor the extent mask can touch and walks them unmasked. That is
  arithmetic-preserving, and here it is also worth 3-6% on the long shapes. It is
  given up anyway, because on this backend it does not compute: with the generic
  kernel unchanged except for forcing ``peeled`` to zero -- one line, the same
  arithmetic -- ``test_kv_block_size[256]`` goes from 1021 wrong elements and
  3.19 of absolute error to passing. The trigger is neither the unmasked loads
  nor the warp count nor ``num_stages``, all three of which were ruled out by
  measurement; it is two sequential loops over one accumulator. So the KV walk
  here is a single always-masked loop.

The rest is tuning, and it is this part's own. The tiling was chosen by
measurement on a BI-V150; the benchmark's own shapes, one sequence each, with the
generic launcher in the same process:

* the KV tile is 16 -- the narrowest ``tl.dot`` compiles with, and also the
  fastest, by a wide margin: at 4096 tokens and head 128 a 64-wide tile is
  13124 us and a 32-wide one 8371, against 4671 at 16. Sixteen compute units is
  few enough that what runs out is the KV loop's staging buffers, not the tile's
  arithmetic;
* a flat 1-D grid with the head index varying fastest and the atoms numbered
  widest-first. Programs are dispatched in linear-id order, so numbering decides
  what runs together, and on this device both halves of that are worth having:
  head-fastest beats atom-fastest by 9% across the eight shapes it was measured
  on (234 us against 368 at 1024 tokens and head 64), and widest-first beats
  narrowest-first by 1-17% wherever there is a tail to fill, costing about 6% at
  128 tokens where there is not. The flat grid is what lets both hold at once --
  a 2-D grid would have to put one of them on the slow axis;
* ``num_warps`` sized by *threads* rather than warps: this part's warps are 64
  threads against NVIDIA's 32, so a warp count carried over from an NVIDIA
  measurement is twice the CTA it was chosen for. The count measured flat from 4
  to 8 here, which is why this is the generic launcher's rule with the warp width
  read from the device rather than a tuning of its own.

The result is 2.04x the generic launcher over the benchmark's eleven shapes
(37009 us against 75524), winning nine of them by 1.25-2.75x. Both of the two it
loses are the 128-token shapes, at 0.64 and 0.69: a single atom is sixteen
programs on sixteen compute units, no occupancy to buy, and the narrow tile's
loop depth is then all there is -- the one regime where this tiling is the wrong
answer, and it is a property of the shape rather than of the cache.

Everything shared is imported from the generic module so the two cannot drift:
the atom-field constants (``_ATOM_*``, which are the kernel's view of the
layout), ``_LOG2E``, and ``_check_launch_arguments`` -- the input checks are one
implementation used by both, so a backend cannot accept something another
rejects. Importing ``_check_launch_arguments`` rather than restating it is also
what keeps the reference-launcher tests working: they monkeypatch the capability
reader in the module it is defined in, and that has to be the generic module
whichever backend dispatched here -- which is also the module ``_capability_gate``
has to reach into, and the only such reach in this file.

Registration is by name. ``_iluvatar/ops/deepspeed/__init__.py`` re-exports this
module's ``blocked_flash`` straight from here, and ``SpecOpRegistrar`` writes it
over the same-named generic function in ``flag_train.deepspeed`` when that
package is imported -- the namespace the tests and the benchmark read the
operator from.
"""

import functools
import importlib
import logging

import triton
import triton.language as tl

from flag_train.deepspeed.blocked_flash import (
    _ATOM_BLOCK_OFFSET,
    _ATOM_GLOBAL_Q_IDX,
    _ATOM_KV_BLOCKS,
    _ATOM_Q_LEN,
    _ATOM_Q_START,
    _ATOM_STRIDE,
    _ATOM_TOTAL_EXTENT,
    _LOG2E,
    _check_launch_arguments,
)
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

# The generic module as a *module*. Importing ``flag_train.deepspeed.blocked_flash``
# by name yields the function once registration has written this backend's
# operator over it, so the module object -- which is what ``_capability_gate``
# needs -- is reached through ``sys.modules`` like this.
_generic_module = importlib.import_module("flag_train.deepspeed.blocked_flash")

# The capability reader the generic module was imported with, which on this
# device is the one that answers in CUDA's terms. See ``_capability_gate``.
_READER_AS_IMPORTED = _generic_module._device_capability

# The KV tile, for every cache block size this kernel can walk. ``tl.dot``
# requires N to be at least 16, and 16 is also where this part measures best --
# see ``_block_n_for``.
_BLOCK_N = 16


@functools.lru_cache(maxsize=None)
def _device_properties(device_index):
    """The device's property dict, or ``None`` if it cannot be read.

    Cached, and it has to be: the driver call is host-side and this launcher
    makes it twice per invocation, against kernels that run in tens of
    microseconds -- uncached it does not size the tiling, it *is* the timing.
    Device properties do not change under a running process, so there is nothing
    to invalidate.

    ``None`` is not an error path to be avoided at all costs: the callers below
    each fall back to the behaviour that existed before the device could be
    asked at all.
    """
    try:
        from triton.runtime.driver import driver as triton_driver

        return triton_driver.active.utils.get_device_properties(device_index)
    except Exception:
        return None


def _max_shared_memory(device_index):
    """The device's shared-memory limit in bytes, or ``None`` if unreadable."""
    props = _device_properties(device_index)
    limit = props.get("max_shared_mem") if props else None
    return int(limit) if limit else None


def _warp_size(device_index):
    """Threads per warp, or 32 when the device will not say.

    A warp is not a fixed width across devices -- 32 on the NVIDIA parts this
    operator was first tuned on, 64 on the BI-V150 -- and the per-CTA sizing
    below is a count of *threads*, so it has to know which. 32 is the value that
    sizing already assumed before it could ask, so an unreadable device keeps
    that behaviour rather than getting a new one.
    """
    props = _device_properties(device_index)
    size = props.get("warpSize") if props else None
    return int(size) if size else 32


def _block_n_for(kv_block_size):
    """The KV tile this cache's block size is walked in.

    Sixteen, whatever the block size -- which is two statements.

    The first is a limit: ``tl.dot`` needs N of at least 16, so a cache block
    narrower than that cannot be tiled at all and has to be refused rather than
    launched into a compiler error.

    The second is the measurement, and it is the same 16. On a BI-V150 the
    narrowest tile is also the fastest by a wide margin -- at 4096 tokens, head
    128, a 64-wide tile is 13124 us and a 32-wide one 8371, against 4671 at 16
    -- which is not what a tile that narrow usually means. It is the shared
    memory, and through it the resident program count: 16 compute units is few
    enough that the KV loop's staging buffers, not the tile's arithmetic, are
    what the device runs out of room for. The narrow tile also keeps the tile a
    power of two, for a reason the limit above does not cover: the inner loop's
    lane mask is ``kv_pos < total_extent``, which says nothing about the block
    boundary. A tile that divides the block cannot reach past it; a tile that
    does not -- 16 on a 24-wide block -- loads rows belonging to the next
    physical block and attends to KV the query was never given. Paged caches are
    powers of two, and the generic launcher's tiling is a power of two for the
    same reason, so this is a property of the layout rather than something to
    guard here. It is why the tile is 16 rather than the smallest of a set of
    constants to choose between.
    """
    if kv_block_size < _BLOCK_N:
        raise ValueError(
            "blocked flash walks a cache block in KV tiles of at least %d rows, "
            "so the key cache's block size must be at least that; this one is %d"
            % (_BLOCK_N, kv_block_size)
        )
    return _BLOCK_N


def _allocated_shared_memory(
    out,
    q,
    k,
    v,
    attention_atoms,
    kv_block_idx,
    softmax_scale,
    is_causal,
    kv_block_size,
    head_size,
    h_h_k_ratio,
    num_heads_q,
    num_warps,
    block_m,
    block_n,
    block_d,
):
    """Shared memory Triton allocates for one tiling, in bytes, without launching.

    ``warmup`` compiles and hands back the compiled kernel without running it,
    which is the whole point: a tiling past the device's limit still *compiles* --
    only the launch refuses it, inside ``CompiledKernel._init_handles``. So the
    allocator's own answer is readable for a tiling that could never be launched,
    and the check in front of the launch needs neither a failed launch nor a
    closed form.

    A closed form is what this is replacing, and it is not available here: the
    allocation is the allocator's answer about buffer reuse, which is a function
    of the layouts it picked for the two dots, and that is a function of the
    compiler. The expression the other backends fit to their parts -- a query
    tile against the KV loop's live set -- was checked against this one and
    under-states the wide-head tilings: it puts a 32-row tile at head 256 with
    ``BLOCK_N`` 64 at 69632 bytes, where this part's compiler allocates 196608.
    Wrong in the direction that lets a doomed launch through is the failure being
    replaced, so the figure is asked for rather than derived.

    The arguments are the launch's, because the kernel is compiled per signature
    and the strides are part of it; they are not read.

    ``LibEntry`` does not forward ``warmup``, so it is reached through the
    ``JITFunction`` it wraps -- the same one the launch compiles.
    """
    compiled = _blocked_flash_fwd_kernel.jit_function.warmup(
        out,
        q,
        k,
        v,
        attention_atoms,
        kv_block_idx,
        softmax_scale,
        q.stride(0),
        out.stride(0),
        k.stride(1),
        h_h_k_ratio,
        num_heads_q,
        grid=(1,),
        IS_CAUSAL=is_causal,
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_SIZE=head_size,
        num_warps=num_warps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    # ``warmup`` returns the same ``(kernel, hooks)`` pair a launch does.
    compiled = compiled[0] if isinstance(compiled, tuple) else compiled
    return compiled.metadata.shared


def _check_fits(needed, limit, block_m, head_size):
    """Refuse a query tile the device cannot hold, before the launch that would.

    The KV tile is already as narrow as it compiles at -- see ``_block_n_for`` --
    so the only way left to be over the limit is the query tile, and no caller
    can be served by shrinking that here: ``BLOCK_M`` has to span the widest atom
    or query rows are dropped silently. So this raises with the two figures it
    measured and leaves the fix with whoever sizes the atoms.

    The alternative is what the generic launcher does, which is nothing: the
    launch is refused inside the compiler with ``Required: 196608, Hardware
    limit: 131072``, a message about shared memory that names neither the query
    tile nor the parameter that decides it.
    """
    if needed <= limit:
        return
    raise RuntimeError(
        "blocked flash needs %d bytes of shared memory for a %d-row query tile "
        "at head_size %d, and the device has %d. That is already at the narrowest "
        "KV tile this kernel compiles with (16 rows); build the atoms with a "
        "smaller q_block_size." % (needed, block_m, head_size, limit)
    )


@libentry()
@triton.jit
def _blocked_flash_fwd_kernel(
    out_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    atoms_ptr,
    block_idx_ptr,
    qk_scale,
    q_row_stride,
    o_row_stride,
    k_row_stride,
    h_h_k_ratio,
    n_heads_q,
    IS_CAUSAL: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Which (atom, head) pair this program runs is a scheduling decision, and it
    # is worth about 9% of the kernel: programs are dispatched in linear-id
    # order, so the order they are numbered in decides which of them run at the
    # same time. Both halves are measured on the BI-V150, both are order-only --
    # permuting independent programs cannot change a result -- and both are
    # cheap enough to be worth taking unconditionally:
    #
    #   * heads vary fastest, worth 9% across the eight shapes measured and 36%
    #     at 1024 tokens (234 us against 368). The programs resident at any
    #     moment are then the same atom on several heads, which for
    #     grouped-query attention are reading the same cache blocks; inside a
    #     block the heads are contiguous, so memory sees one sequential stream
    #     instead of ``n_heads_q`` strided ones.
    #   * atoms run from the widest to the narrowest, worth 1-17% wherever there
    #     is a tail to fill, and 6% the wrong way at 128 tokens, where there is
    #     not. An atom's work grows with the query it carries, and under the
    #     causal layout the query grows with the index: in the benchmark's
    #     1024-token sequence each atom covers 128 queries and atom *i* walks
    #     2(i+1) cache blocks, so the last atom does eight times the work of the
    #     first. Numbering them backwards hands the long programs to the device
    #     first and leaves it the short ones to drain on -- the tail is where the
    #     imbalance would otherwise sit. For atoms built in any other order the
    #     reversal is an arbitrary permutation, which is why it costs nothing to
    #     apply unconditionally.
    #
    # The flat 1-D grid is what lets both hold at once. A 2-D ``(heads, atoms)``
    # grid would put the atoms on one of them whether or not that is the fast
    # axis, and would need ``num_atoms`` to fit in grid dimension 1, capping a
    # batch at 65535 atoms; flattening keeps the fast-varying axis under our
    # control while leaving the grid bound at 2**31 programs.
    pid = tl.program_id(0)
    hq = pid % n_heads_q
    atom = tl.num_programs(0) // n_heads_q - 1 - pid // n_heads_q

    # --- atom metadata -----------------------------------------------------
    atom_base = atoms_ptr + atom * _ATOM_STRIDE
    block_offset = tl.load(atom_base + _ATOM_BLOCK_OFFSET)
    q_start = tl.load(atom_base + _ATOM_Q_START)
    q_len = tl.load(atom_base + _ATOM_Q_LEN)
    kv_blocks = tl.load(atom_base + _ATOM_KV_BLOCKS)
    total_extent = tl.load(atom_base + _ATOM_TOTAL_EXTENT)
    global_q_idx = tl.load(atom_base + _ATOM_GLOBAL_Q_IDX)

    # Grouped-query attention: query heads are grouped in contiguous runs over one
    # KV head, so dividing maps 32 query heads and 8 KV heads to runs of four
    # (heads 0-3 read KV head 0, and so on). At h_h_k_ratio == 1 the division is
    # the identity and this is plain multi-head attention.
    hk = hq // h_h_k_ratio

    # --- queries -----------------------------------------------------------
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < q_len
    d_mask = offs_d < HEAD_SIZE

    q_rows = q_start + offs_m
    q_ptrs = q_ptr + q_rows[:, None] * q_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    # Kept in the input dtype so it matches ``k`` in the dot; the scale is applied
    # to the fp32 scores instead, which is where it belongs numerically.
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    # A query token's position in the sequence, not in the tensor: a continuation
    # starts at global_q_idx, which is what the causal mask compares against.
    global_q_pos = global_q_idx + offs_m

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # One loop, every tile masked. The causal and extent comparisons below are
    # the same ones for every tile of every cache block, so a tile that lies
    # wholly before the program's first query row could skip them -- and the
    # generic kernel peels exactly that prefix, for a few percent. It is not
    # peeled here because on this part it does not compute: two sequential loops
    # over one accumulator return wrong numbers, and one always-masked loop is
    # the whole of the difference. The module docstring has the measurement.
    for b in range(0, kv_blocks):
        block = tl.load(block_idx_ptr + block_offset + b)
        block_row_base = block * k_row_stride * KV_BLOCK_SIZE

        # A KV block holds kv_block_size tokens; walk it in BLOCK_N tiles so the
        # tile does not have to know the cache's block size.
        for t0 in range(0, KV_BLOCK_SIZE, BLOCK_N):
            offs_n = t0 + tl.arange(0, BLOCK_N)
            kv_pos = b * KV_BLOCK_SIZE + offs_n
            n_mask = kv_pos < total_extent

            k_ptrs = (
                k_ptr
                + block_row_base
                + offs_n[:, None] * k_row_stride
                + hk * HEAD_SIZE
                + offs_d[None, :]
            )
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * qk_scale
            if IS_CAUSAL:
                # A query attends to every KV position up to and including its own.
                qk = tl.where(
                    kv_pos[None, :] <= global_q_pos[:, None], qk, float("-inf")
                )
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])

            v_ptrs = (
                v_ptr
                + block_row_base
                + offs_n[:, None] * k_row_stride
                + hk * HEAD_SIZE
                + offs_d[None, :]
            )
            v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            l_i = l_i * alpha + tl.sum(p, 1)
            # ``p`` is rounded into the compute dtype here. That is deliberate:
            # the bar for this operator is the reference implementation's own
            # tolerance, which this clears with room to spare, and carrying the
            # rounding residual in a second dot buys accuracy the requirement
            # does not ask for at a cost the benchmark would show.
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

    # A row whose keys were all masked out has no contribution to renormalize.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    o_ptrs = out_ptr + q_rows[:, None] * o_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :]
    )


def _no_capability_reported():
    """What ``_device_capability`` answers on this backend. See ``_capability_gate``."""
    return None


def _capability_gate():
    """Leave the generic launcher's capability gate to the device.

    ``_check_launch_arguments`` reads the capability from the module it is
    defined in, and that module's reader answers in CUDA's terms here: ``(7, 1)``,
    Volta in NVIDIA's numbering, on a part that has tensor cores and runs this
    kernel. A translated number is not evidence about this hardware, so the gate
    is given the "will not say" its own check already provides for -- by replacing
    the reader, which is the only way to reach a value it reads for itself. The
    registrar writes this backend's operator over the generic one the same way;
    this is that move applied to the one hook that decides whether the launch
    happens at all.

    Installed once, on the first launch from this backend, and left in place: a
    scope that restored the original would leave a window in which another
    thread's launch reads the number this device's API answers with, which is the
    number being rejected here.

    A reader that has already been replaced is not touched. A replacement is a
    deliberate statement about the device -- the tests that pin this rule replace
    it, one with a too-old number and one with none -- so it is honoured, and this
    cannot be used to hide from a limit someone else has stated.
    """
    if _generic_module._device_capability is _READER_AS_IMPORTED:
        _generic_module._device_capability = _no_capability_reported


def blocked_flash(
    out,
    q,
    k,
    v,
    attention_atoms,
    kv_block_idx,
    softmax_scale,
    is_causal=True,
    q_block_size=None,
):
    """Flash attention forward over the blocked KV-cache described by ``atoms``.

    Args:
        out (Tensor): output of shape ``[total_q, n_heads_q * head_size]``.
        q (Tensor): queries of shape ``[total_q, n_heads_q * head_size]``.
        k (Tensor): key cache of shape ``[n_blocks, block_size, n_heads_kv, head_size]``,
            contiguous in the last dimension.
        v (Tensor): value cache, same shape and layout as ``k``.
        attention_atoms (Tensor): ``[num_atoms, 8]`` int32; see the generic
            module's docstring for the layout.
        kv_block_idx (Tensor): int32 physical block indices, addressed by each
            atom's ``block_offset`` field.
        softmax_scale (float): scale applied to the query/key dot product.
        is_causal (bool): mask each query against its position in the sequence.
        q_block_size (int, optional): an upper bound on the atoms' ``q_len``,
            normally the ``q_block_size`` the atoms were built with. Supplying it
            avoids a device sync -- see below. ``None`` reads the maximum out of
            ``attention_atoms``, which is always correct and is the default.
            It is a **bound, not a hint**: an atom wider than it would have its
            extra rows masked away, silently.

    Returns:
        Tensor: ``out``, updated in place.
    """
    logger.debug("TRAIN_ILUVATAR BLOCKED_FLASH")

    # The query tile must span the widest atom. Taking that maximum off the
    # device costs a device sync -- about 70 us on this part, plus the copy --
    # which is several times the 41 us kernel it is sizing at 128 tokens. So a
    # caller who already knows the bound hands it over and the sync never
    # happens.
    head_size = k.size(-1)
    num_heads_q = q.size(-1) // head_size
    num_heads_kv = k.size(-2)
    num_atoms = attention_atoms.size(0)

    _capability_gate()
    _check_launch_arguments(q, k, v, head_size, num_heads_q, num_heads_kv)

    # k is [n_blocks, block_size, n_heads_kv, head_size] with contiguous heads, so
    # stride(0) / stride(1) is the cache's block size -- the same quantity the
    # DeepSpeed launcher leaves implicit.
    kv_block_size = k.stride(0) // k.stride(1)
    if kv_block_size * k.stride(1) != k.stride(0):
        raise ValueError("key cache must be contiguous across the block dimension")

    if q_block_size is None:
        # ``int()`` because the constants are constexpr for the kernel's benefit
        # and torch will not index a tensor with one; ``item()`` is the sync.
        q_len_max = (
            int(attention_atoms[:, int(_ATOM_Q_LEN)].max().item()) if num_atoms else 0
        )
    else:
        q_len_max = q_block_size

    # One program per (atom, query head), so the query tile has to span the widest
    # atom in the batch -- the rest mask off their padding. BLOCK_N is the KV tile
    # walked inside a single cache block. BLOCK_D is the whole head, padded up.
    BLOCK_M = max(triton.next_power_of_2(q_len_max), 16)
    BLOCK_N = _block_n_for(kv_block_size)
    BLOCK_D = triton.next_power_of_2(head_size)

    # Size the CTA to the accumulator. ``acc`` is BLOCK_M x BLOCK_D of fp32 and
    # lives in registers for the whole KV walk, so the per-thread share of it is
    # what decides whether the kernel spills.
    #
    # The rule targets 64 accumulator elements per thread. The quantity being
    # held fixed is the per-*thread* share, so the constant has to scale with the
    # warp width: the NVIDIA measurement this started from was 8 warps of 32
    # threads, and this part's warps are 64 threads, so the same target is half
    # the warp count. Four warps is the floor.
    device_index = q.device.index if q.device.index is not None else 0
    num_warps = max(4, (BLOCK_M * BLOCK_D) // (64 * _warp_size(device_index)))

    # Shared memory is the limit that actually bites here, and it is a property
    # of the *device*: a BI-V150 has 128 KB against the 163 KB the generic
    # launcher's tiling assumes, so the same tiling that launches elsewhere is
    # refused here -- deep inside the compiler, where the caller cannot act on
    # it. So check the tiling against the device first, and against the
    # compiler's own figure for it rather than an expression standing in.
    limit = _max_shared_memory(device_index)
    if limit is not None:
        _check_fits(
            _allocated_shared_memory(
                out,
                q,
                k,
                v,
                attention_atoms,
                kv_block_idx,
                softmax_scale * _LOG2E,
                is_causal,
                kv_block_size,
                head_size,
                num_heads_q // num_heads_kv,
                num_heads_q,
                num_warps,
                BLOCK_M,
                BLOCK_N,
                BLOCK_D,
            ),
            limit,
            BLOCK_M,
            head_size,
        )

    _blocked_flash_fwd_kernel[(num_atoms * num_heads_q,)](
        out,
        q,
        k,
        v,
        attention_atoms,
        kv_block_idx,
        # The kernel's softmax runs on exp2, so log2(e) rides on the scale rather
        # than costing an exp() per score.
        softmax_scale * _LOG2E,
        q.stride(0),
        out.stride(0),
        k.stride(1),
        # Query heads that share one KV head; the kernel divides by it to find
        # this program's KV head.
        num_heads_q // num_heads_kv,
        num_heads_q,
        IS_CAUSAL=is_causal,
        # Both are host-side values that the kernel only ever uses as a loop
        # bound and a mask extent. As runtime scalars they force the inner KV
        # loop to stay dynamic and every load mask to carry a term that is
        # statically true; as constexpr they fold away.
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_SIZE=head_size,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out
