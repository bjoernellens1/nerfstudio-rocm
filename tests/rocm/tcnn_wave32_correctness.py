"""Wave32 (RDNA / gfx1151) numerical-correctness check for tiny-rocm-nn (tinycudann).

Purpose
-------
tiny-cuda-nn was written for NVIDIA hardware, where a warp is 32 lanes.  AMD CDNA
(gfx90a/gfx942) runs wave64; AMD RDNA (gfx10xx/gfx11xx, incl. gfx1151) runs wave32.
Kernels that hard-code warp-size assumptions -- especially `FullyFusedMLP`, whose
matmuls and weight-gradient reductions are built on warp-level shuffles and
cooperative fragment layouts -- can produce *silently wrong* results rather than
crashing.  That is exactly the failure mode gsplat exhibited on this same GPU
earlier in this project.  "Training runs without crashing" is therefore NOT
evidence of correctness; this script produces the actual numbers.

Method
------
For each component we build an independent reference in pure PyTorch that
implements *tcnn's own documented semantics* (transcribed from the tiny-rocm-nn
headers, see citations inline), feed it tcnn's *own* parameters, and compare.

Two references are computed for every check:

  (a) fp32 reference  -- the mathematically intended result.
  (b) precision-emulated reference -- the same computation with tcnn's internal
      half-precision rounding applied at the same points the kernels apply it.

The verdict rule is self-calibrating and needs no magic tolerance:

      |tcnn - a|  must be the same order of magnitude as  |b - a|

|b - a| is the *legitimate* floating-point error floor implied by tcnn running in
half precision.  A wave32 lane/shuffle bug does not produce errors near that
floor -- it produces errors on the order of the output magnitude itself,
typically 1e2..1e4 times the floor.  We flag PASS when
|tcnn - a| <= FLOOR_FACTOR * max(|b - a|, tiny) and FAIL otherwise.

Also checked, because they are the specific signature of a wave-size bug:
  * batch-size sweep across / around the `batch_size_granularity` (256 on this
    build) and the 32/64 lane boundaries -- tail-handling bugs only appear at
    some sizes,
  * bitwise run-to-run determinism (a wave-size race shows up as nondeterminism
    that a single reference comparison can pass by luck),
  * error *structure* -- which rows fail and whether they cluster mod 16/32/64.

Run:
  podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    -e PYTHONPATH=/workspace/nerfstudio-rocm \
    -v <worktree>:/workspace/nerfstudio-rocm \
    --entrypoint python <image-with-tinycudann> \
    /workspace/nerfstudio-rocm/tests/rocm/tcnn_wave32_correctness.py
"""

import math
from typing import Optional

import numpy as np
import torch
from torch import nn

from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.utils.external import TCNN_EXISTS

assert TCNN_EXISTS, "tinycudann must be importable for this check to be meaningful"

import tinycudann as tcnn  # noqa: E402  (import after the TCNN_EXISTS assert on purpose)

device = torch.device("cuda")

# How many times the half-precision error floor we tolerate before calling it a
# real numerical error.  8x is deliberately generous: legitimate fp16 rounding
# differs from our emulation by small constant factors (fma vs mul+add, MMA
# accumulate width, operation order), but a wave32 correctness bug overshoots by
# orders of magnitude, so the verdict is insensitive to the exact value here.
FLOOR_FACTOR = 8.0

RESULTS = {}


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def compare(
    label: str,
    got: torch.Tensor,
    ref_fp32: torch.Tensor,
    ref_emul: Optional[torch.Tensor] = None,
    component: Optional[str] = None,
) -> bool:
    """Compare `got` against the fp32 reference, calibrated by the emulated one."""
    got = got.detach().float()
    a = ref_fp32.detach().float()
    err = (got - a).abs()
    max_err = err.max().item()
    scale = a.abs().max().item()

    if ref_emul is not None:
        b = ref_emul.detach().float()
        floor = (b - a).abs().max().item()
        floor_note = f"floor|b-a|={floor:.3e}"
        # emulation-vs-tcnn agreement is informative but not the verdict
        emul_diff = (got - b).abs().max().item()
        floor_note += f"  |tcnn-b|={emul_diff:.3e}"
    else:
        floor = 0.0
        floor_note = "floor|b-a|=n/a"

    bar = FLOOR_FACTOR * max(floor, 1e-7 * max(scale, 1e-3))
    ok = max_err <= bar
    rel = max_err / scale if scale > 0 else float("nan")

    print(
        f"  [{'PASS' if ok else 'FAIL'}] {label:<34} "
        f"|tcnn-a|={max_err:.3e}  rel={rel:.2e}  {floor_note}  bar={bar:.3e}  "
        f"(|a|max={scale:.3e})"
    )

    if not ok and got.ndim == 2:
        # error structure: which rows are bad, and do they cluster mod 16/32/64?
        row_err = err.max(dim=-1).values
        bad = (row_err > bar).nonzero().flatten()
        frac = bad.numel() / row_err.numel()
        print(f"         bad rows: {bad.numel()}/{row_err.numel()} ({frac:.1%})")
        for m in (16, 32, 64):
            hist = torch.bincount(bad % m, minlength=m)
            occupied = int((hist > 0).sum().item())
            print(f"         mod {m:>3}: {occupied}/{m} residues occupied")
        print(f"         first bad row indices: {bad[:16].tolist()}")

    if component is not None:
        RESULTS.setdefault(component, []).append((label, ok, max_err, floor))
    return ok


def determinism(label: str, fn, n: int = 3) -> bool:
    """Run `fn` n times and require bitwise-identical results."""
    ref = fn()
    ok = True
    for _ in range(n - 1):
        cur = fn()
        if not torch.equal(ref, cur):
            ok = False
            d = (cur.float() - ref.float()).abs().max().item()
            print(f"  [FAIL] {label:<34} NONDETERMINISTIC, max delta {d:.3e}")
            break
    if ok:
        print(f"  [PASS] {label:<34} bitwise identical over {n} runs")
    return ok


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
banner("Environment")
print(f"  torch            : {torch.__version__}")
print(f"  device           : {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"  gcnArchName      : {getattr(props, 'gcnArchName', '?')}")
print(f"  warp/wave size   : {getattr(props, 'warp_size', 'unknown')}")
print(f"  batch granularity: {tcnn.modules._C.batch_size_granularity()}")
print(f"  preferred prec.  : {tcnn.modules._C.preferred_precision()}")


# =========================================================================== #
# 1. MLP / FullyFusedMLP   -- highest-risk component (warp-shuffle reductions)
# =========================================================================== #
IN_DIM = 32  # NOTE: tcnn pads the network input up to minimum_alignment(network)
#              == n_neurons == 64 for FullyFusedMLP (network_with_input_encoding.h:69),
#              and the IdentityEncoding fills those padded columns with 1.0, NOT 0.0
#              (encodings/identity.h:64).  We therefore zero the corresponding
#              columns of tcnn's first weight matrix so the padding contributes
#              nothing, making the torch MLP an exact reference.
LAYER_WIDTH = 64
OUT_DIM = 15  # deliberately unaligned: exercises padded_output_width=16
NUM_LAYERS = 3  # -> tcnn n_hidden_layers = 2, i.e. 3 weight matrices
PADDED_OUT = ((OUT_DIM + 15) // 16) * 16

HALF = None  # filled in on first use from the native param precision


def run_mlp_test(activation: nn.Module, tag: str, component: str) -> None:
    global HALF
    banner(f"1{tag}. MLP (tcnn FullyFusedMLP, activation={activation.__class__.__name__}) "
           f"vs nerfstudio torch MLP -- exact parameter copy")

    torch.manual_seed(0)
    mlp_tcnn = MLP(
        in_dim=IN_DIM, num_layers=NUM_LAYERS, layer_width=LAYER_WIDTH, out_dim=OUT_DIM,
        activation=activation, out_activation=None, implementation="tcnn",
    ).to(device)
    mlp_torch = MLP(
        in_dim=IN_DIM, num_layers=NUM_LAYERS, layer_width=LAYER_WIDTH, out_dim=OUT_DIM,
        activation=activation, out_activation=None, implementation="torch",
    ).to(device)

    assert mlp_tcnn.tcnn_encoding is not None, "tcnn path was not taken"
    hp = mlp_tcnn.tcnn_encoding.native_tcnn_module.hyperparams()
    otype = hp.get("network", hp).get("otype", hp.get("otype"))
    assert otype == "FullyFusedMLP", f"expected FullyFusedMLP, got {otype} -- test would be meaningless"
    net_prec = mlp_tcnn.tcnn_encoding.native_tcnn_module.param_precision()
    # The python-side `params` Parameter is fp32, but modules.py casts it to the
    # native precision on every forward, so the *effective* precision is this one.
    HALF = tcnn.modules._torch_precision(net_prec)
    print(f"  hyperparams      : {hp}")
    print(f"  params storage   : {mlp_tcnn.tcnn_encoding.params.dtype}  native: {net_prec} -> {HALF}")

    n_params = mlp_tcnn.tcnn_encoding.params.numel()
    padded_in = (n_params - (NUM_LAYERS - 2) * LAYER_WIDTH**2 - PADDED_OUT * LAYER_WIDTH) // LAYER_WIDTH
    assert padded_in >= IN_DIM, f"derived padded input width {padded_in} < in_dim {IN_DIM}"
    print(f"  n params (tcnn)  : {n_params};  m_input_width={padded_in} (from {IN_DIM}), "
          f"padded_output_width={PADDED_OUT}")

    with torch.no_grad():
        for layer in mlp_torch.layers:
            layer.bias.zero_()
            # round the torch weights onto the half grid tcnn will use, so the only
            # remaining difference is arithmetic, not the weights themselves
            layer.weight.copy_(layer.weight.to(HALF).float())
        blocks = []
        for i, layer in enumerate(mlp_torch.layers):
            w = layer.weight  # [out_features, in_features], row-major
            if i == 0:
                b = torch.zeros(LAYER_WIDTH, padded_in, device=device)
                b[:, :IN_DIM] = w  # cols >= IN_DIM stay 0 so the 1.0 padding is inert
            elif i == len(mlp_torch.layers) - 1:
                b = torch.zeros(PADDED_OUT, LAYER_WIDTH, device=device)
                b[:OUT_DIM] = w
            else:
                b = w.clone()
            blocks.append(b)
        flat = torch.cat([b.reshape(-1) for b in blocks])
        assert flat.numel() == n_params, f"param layout mismatch: built {flat.numel()} vs tcnn {n_params}"
        mlp_tcnn.tcnn_encoding.params.copy_(flat.to(mlp_tcnn.tcnn_encoding.params.dtype))
    print(f"  parameter copy   : OK ({flat.numel()} values, biases zeroed, weights rounded to {HALF})")

    def split_params(flat_vec):
        """Slice a flat tcnn parameter/gradient vector into the *meaningful*
        sub-blocks, dropping the input-padding columns and output-padding rows
        (tcnn's gradients there are non-zero but have no torch analogue)."""
        parts, pos = [], 0
        w0 = flat_vec[pos : pos + LAYER_WIDTH * padded_in].reshape(LAYER_WIDTH, padded_in)
        parts.append(w0[:, :IN_DIM])
        pos += LAYER_WIDTH * padded_in
        for _ in range(NUM_LAYERS - 2):
            parts.append(flat_vec[pos : pos + LAYER_WIDTH**2].reshape(LAYER_WIDTH, LAYER_WIDTH))
            pos += LAYER_WIDTH**2
        parts.append(flat_vec[pos : pos + PADDED_OUT * LAYER_WIDTH].reshape(PADDED_OUT, LAYER_WIDTH)[:OUT_DIM])
        pos += PADDED_OUT * LAYER_WIDTH
        assert pos == flat_vec.numel()
        return torch.cat([p.reshape(-1) for p in parts])

    def ref_fp32(x):
        return mlp_torch(x)

    def ref_emul(x):
        """Same MLP with tcnn's half-precision rounding: operands in half, matmul
        accumulated in fp32 (MMA), result stored back to half between layers."""
        h, n = x, len(mlp_torch.layers)
        for i, layer in enumerate(mlp_torch.layers):
            h = (h.float() @ layer.weight.to(HALF).float().t()).to(HALF).float()
            if i < n - 1:
                h = activation(h).to(HALF).float()
        return h

    def min_abs_preactivation(x):
        """Smallest |pre-activation| anywhere in the hidden layers, per row.  With
        ReLU, a value near 0 means an fp16 rounding difference can flip the
        activation mask and legitimately change that row's gradient a lot."""
        h, mins = x, []
        for i, layer in enumerate(mlp_torch.layers[:-1]):
            z = h @ layer.weight.t()
            mins.append(z.abs().min(dim=-1).values)
            h = activation(z)
        return torch.stack(mins, -1).min(dim=-1).values

    print("\n  -- forward, batch-size sweep --")
    for bs in (1, 31, 33, 127, 128, 129, 255, 256, 257, 4096):
        torch.manual_seed(100 + bs)
        x = torch.rand(bs, IN_DIM, device=device) * 2 - 1
        with torch.no_grad():
            compare(f"forward bs={bs}", mlp_tcnn(x), ref_fp32(x), ref_emul(x), component=component)

    print("\n  -- determinism --")
    torch.manual_seed(7)
    x_det = torch.rand(4096, IN_DIM, device=device) * 2 - 1
    det_ok = determinism("forward x3 (bs=4096)", lambda: mlp_tcnn(x_det).detach().clone())
    RESULTS.setdefault(component, []).append(("forward determinism", det_ok, 0.0, 0.0))

    def do_backward(x0, dout, label, comp):
        xt = x0.clone().requires_grad_(True)
        out_t = mlp_tcnn(xt)
        mlp_tcnn.zero_grad(set_to_none=True)
        out_t.backward(dout.to(out_t.dtype))
        g_in_tcnn = xt.grad.clone()
        g_par_tcnn = split_params(mlp_tcnn.tcnn_encoding.params.grad.detach().float().clone())

        def weight_grads():
            return torch.cat([layer.weight.grad.reshape(-1) for layer in mlp_torch.layers])

        xa = x0.clone().requires_grad_(True)
        mlp_torch.zero_grad(set_to_none=True)
        ref_fp32(xa).backward(dout)
        g_in_a, g_par_a = xa.grad.clone(), weight_grads()

        xb = x0.clone().requires_grad_(True)
        mlp_torch.zero_grad(set_to_none=True)
        ref_emul(xb).backward(dout)
        g_in_b, g_par_b = xb.grad.clone(), weight_grads()
        mlp_torch.zero_grad(set_to_none=True)

        ok_in = compare(f"backward d/dinput  {label}", g_in_tcnn, g_in_a, g_in_b, component=comp)
        compare(f"backward d/dweight {label}", g_par_tcnn, g_par_a, g_par_b, component=comp)

        if not ok_in and isinstance(activation, nn.ReLU):
            # Diagnose: is the disagreement concentrated on rows whose
            # pre-activations sit on the ReLU kink?
            with torch.no_grad():
                mn = min_abs_preactivation(x0)
                row_err = (g_in_tcnn - g_in_a).abs().max(dim=-1).values
                bad = row_err > 8 * max((g_in_b - g_in_a).abs().max().item(), 1e-7)
                print("         ReLU-kink diagnosis: min|pre-activation| over hidden units")
                print(f"           disagreeing rows: median {mn[bad].median().item():.3e} (n={int(bad.sum())})")
                print(f"           agreeing rows   : median {mn[~bad].median().item():.3e}")
                print("           tcnn's own forward error is ~3e-4 at these magnitudes, so any")
                print("           |pre-activation| below that can legitimately flip the ReLU mask.")

    # A ReLU mask flip is a *discontinuity*, not an arithmetic error: if a hidden
    # pre-activation lies closer to 0 than tcnn's fp16 forward error (~3e-4 here),
    # tcnn and the fp32 reference can legitimately disagree about whether that
    # unit is on, and the row's gradient then differs by O(1) for reasons that
    # have nothing to do with wave32.  KINK_MARGIN is set >10x that error.
    KINK_MARGIN = 5e-3

    print("\n  -- backward: d(loss)/d(input) and d(loss)/d(weights) --")
    for bs in (128, 4096):
        torch.manual_seed(200 + bs)
        x0 = torch.rand(bs, IN_DIM, device=device) * 2 - 1
        torch.manual_seed(300 + bs)
        dout = torch.rand(bs, OUT_DIM, device=device) * 2 - 1  # O(1): keeps fp16 loss scaling in range

        if isinstance(activation, nn.ReLU):
            # all rows: informational only for ReLU (kink-contaminated)
            do_backward(x0, dout, f"bs={bs} ALL rows", f"{component}-ReLU-allrows")
            with torch.no_grad():
                safe = min_abs_preactivation(x0) > KINK_MARGIN
            n = int(safe.sum())
            print(f"         -> {n}/{bs} rows have min|pre-activation| > {KINK_MARGIN:g} (kink-safe)")
            do_backward(x0[safe], dout[safe], f"bs={n} kink-safe", component)
        else:
            do_backward(x0, dout, f"bs={bs}", component)

    if isinstance(activation, nn.ReLU):
        # Decisive discriminator between "ReLU kink" and "lane/wave32 bug":
        # re-run the same bs=128 backward under several input seeds.
        #   kink   -> the disagreeing row INDICES move freely with the seed, and
        #             their count tracks how many rows sit near the kink;
        #   lane bug -> the disagreeing rows stay pinned to the same residues
        #             mod 32 (the wave width) regardless of the input.
        print("\n  -- ReLU disagreement: seed sweep (kink vs lane-bug discriminator) --")
        for seed in (11, 12, 13, 14):
            torch.manual_seed(seed)
            x0 = torch.rand(128, IN_DIM, device=device) * 2 - 1
            torch.manual_seed(seed + 1000)
            dout = torch.rand(128, OUT_DIM, device=device) * 2 - 1

            xt = x0.clone().requires_grad_(True)
            o = mlp_tcnn(xt)
            mlp_tcnn.zero_grad(set_to_none=True)
            o.backward(dout.to(o.dtype))
            g_t = xt.grad.clone()

            xa = x0.clone().requires_grad_(True)
            mlp_torch.zero_grad(set_to_none=True)
            ref_fp32(xa).backward(dout)
            g_a = xa.grad.clone()

            xb = x0.clone().requires_grad_(True)
            mlp_torch.zero_grad(set_to_none=True)
            ref_emul(xb).backward(dout)
            g_b = xb.grad.clone()
            mlp_torch.zero_grad(set_to_none=True)

            with torch.no_grad():
                bar = 8 * max((g_b - g_a).abs().max().item(), 1e-7)
                bad = ((g_t - g_a).abs().max(dim=-1).values > bar).nonzero().flatten()
                mn = min_abs_preactivation(x0)
                near_kink = int((mn < 3e-4).sum())
                res32 = sorted({int(i) % 32 for i in bad})
                print(
                    f"    seed {seed}: {bad.numel()} disagreeing rows {bad[:8].tolist()}  "
                    f"residues mod 32 {res32}  |  rows with min|pre-act| < 3e-4: {near_kink}"
                )


run_mlp_test(nn.ReLU(), "a", "MLP")
# Control experiment: ReLU has a non-differentiable kink, so an fp16 rounding
# difference on a pre-activation near 0 can flip the activation mask and change
# that row's gradient by O(1) for reasons that are precision, not a kernel bug.
# Sigmoid is smooth everywhere, so it isolates the kernel arithmetic.
run_mlp_test(nn.Sigmoid(), "b", "MLP-Sigmoid")

# =========================================================================== #
# 2. SHEncoding
# =========================================================================== #
banner("2. SHEncoding (tcnn SphericalHarmonics) vs transcribed tcnn reference")

SH_LEVELS = 4  # degree=4 -> 16 outputs, already 16-aligned so tcnn's front-pad
#                (data_out(j)=1.0f for j<num_to_pad) is not exercised.

sh_tcnn = SHEncoding(levels=SH_LEVELS, implementation="tcnn").to(device)
sh_torch_ns = SHEncoding(levels=SH_LEVELS, implementation="torch").to(device)
assert sh_tcnn.tcnn_encoding is not None
print(f"  hyperparams      : {sh_tcnn.tcnn_encoding.native_tcnn_module.hyperparams()}")
print(f"  out dim          : {sh_tcnn.get_out_dim()}")


def sh_ref(u: torch.Tensor) -> torch.Tensor:
    """Verbatim transcription of tcnn's `sh_enc` (common_device.h:549-...) up to
    degree 4, including the [0,1] -> [-1,1] input remap done in `kernel_sh`
    (encodings/spherical_harmonics.h:66-72).  Differentiable, so autograd gives
    the factor-of-2 chain rule for free (tcnn does it by hand at :100)."""
    x = u[..., 0] * 2.0 - 1.0
    y = u[..., 1] * 2.0 - 1.0
    z = u[..., 2] * 2.0 - 1.0
    xy, xz, yz = x * y, x * z, y * z
    x2, y2, z2 = x * x, y * y, z * z
    z4 = z2 * z2
    c = [
        torch.full_like(x, 0.28209479177387814),
        -0.48860251190291987 * y,
        0.48860251190291987 * z,
        -0.48860251190291987 * x,
        1.0925484305920792 * xy,
        -1.0925484305920792 * yz,
        0.94617469575755997 * z2 - 0.31539156525251999,
        -1.0925484305920792 * xz,
        0.54627421529603959 * x2 - 0.54627421529603959 * y2,
        0.59004358992664352 * y * (-3.0 * x2 + y2),
        2.8906114426405538 * xy * z,
        0.45704579946446572 * y * (1.0 - 5.0 * z2),
        0.3731763325901154 * z * (5.0 * z2 - 3.0),
        0.45704579946446572 * x * (1.0 - 5.0 * z2),
        1.4453057213202769 * z * (x2 - y2),
        0.59004358992664352 * x * (-x2 + 3.0 * y2),
    ]
    return torch.stack(c, dim=-1)


print("\n  -- forward, batch-size sweep --")
for bs in (1, 31, 33, 127, 128, 129, 255, 256, 257, 4096):
    torch.manual_seed(400 + bs)
    u = torch.rand(bs, 3, device=device)
    with torch.no_grad():
        got = sh_tcnn(u)
        a = sh_ref(u)
        compare(f"forward bs={bs}", got, a, a.to(HALF).float(), component="SHEncoding")

print("\n  -- cross-check vs nerfstudio's own torch SH (sign-convention aware) --")
torch.manual_seed(11)
u = torch.rand(4096, 3, device=device)
with torch.no_grad():
    ns = sh_torch_ns(u * 2 - 1)  # nerfstudio takes directions, tcnn takes [0,1]
    ref = sh_ref(u)
    # tcnn follows the Condon-Shortley convention (negative sign on odd m);
    # nerfstudio's components_from_spherical_harmonics omits it and orders the
    # m<0 / m>0 pairs the other way round.  Map nerfstudio -> tcnn ordering:
    perm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sign = torch.tensor(
        [1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1], device=device, dtype=torch.float32
    )
    # nerfstudio index -> tcnn index for l=3 differs in the (9,15) and (11,13) pairing
    ns_mapped = ns[:, perm] * sign
    ns_mapped[:, 9] = -ns[:, 9]
    ns_mapped[:, 11] = -ns[:, 11]
    ns_mapped[:, 13] = -ns[:, 13]
    ns_mapped[:, 15] = -ns[:, 15]
    d = (ns_mapped - ref).abs().max().item()
    print(f"  nerfstudio-torch SH vs transcribed tcnn SH (after sign map): max abs diff {d:.3e}")
    print("  (informational only -- the transcribed reference is the authority here)")

print("\n  -- determinism --")
torch.manual_seed(12)
u_det = torch.rand(4096, 3, device=device)
det_ok = determinism("forward x3 (bs=4096)", lambda: sh_tcnn(u_det).detach().clone())
RESULTS.setdefault("SHEncoding", []).append(("forward determinism", det_ok, 0.0, 0.0))

print("\n  -- backward: d(loss)/d(input) --")
for bs in (128, 4096):
    torch.manual_seed(500 + bs)
    u0 = torch.rand(bs, 3, device=device)
    torch.manual_seed(600 + bs)
    dout = (torch.rand(bs, SH_LEVELS**2, device=device) * 2 - 1)

    ut = u0.clone().requires_grad_(True)
    o = sh_tcnn(ut)
    o.backward(dout.to(o.dtype))
    g_tcnn = ut.grad.clone()

    ua = u0.clone().requires_grad_(True)
    sh_ref(ua).backward(dout)
    g_a = ua.grad.clone()

    ub = u0.clone().requires_grad_(True)
    sh_ref(ub).backward(dout.to(HALF).float())
    g_b = ub.grad.clone()

    compare(f"backward d/dinput bs={bs}", g_tcnn, g_a, g_b, component="SHEncoding")


# =========================================================================== #
# 3. HashEncoding
# =========================================================================== #
banner("3. HashEncoding (tcnn HashGrid) vs transcribed tcnn reference")

NUM_LEVELS = 16
FEATURES_PER_LEVEL = 2
LOG2_HASHMAP = 19
MIN_RES = 16
MAX_RES = 1024

hg_tcnn = HashEncoding(
    num_levels=NUM_LEVELS,
    min_res=MIN_RES,
    max_res=MAX_RES,
    log2_hashmap_size=LOG2_HASHMAP,
    features_per_level=FEATURES_PER_LEVEL,
    implementation="tcnn",
).to(device)
hg_torch_ns = HashEncoding(
    num_levels=NUM_LEVELS,
    min_res=MIN_RES,
    max_res=MAX_RES,
    log2_hashmap_size=LOG2_HASHMAP,
    features_per_level=FEATURES_PER_LEVEL,
    implementation="torch",
).to(device)
assert hg_tcnn.tcnn_encoding is not None
print(f"  hyperparams      : {hg_tcnn.tcnn_encoding.native_tcnn_module.hyperparams()}")
grid_prec = hg_tcnn.tcnn_encoding.native_tcnn_module.param_precision()
GRID_HALF = tcnn.modules._torch_precision(grid_prec)
print(
    f"  params storage   : {hg_tcnn.tcnn_encoding.params.dtype}  native precision: {grid_prec} -> {GRID_HALF}"
    f"  n={hg_tcnn.tcnn_encoding.params.numel()}"
)

# IMPORTANT: read per_level_scale back from tcnn rather than using nerfstudio's
# float64 growth_factor.  The value crosses the JSON/C++ boundary as float32, and
# log2(float32(g)) differs from log2(g) by ~5e-8 -- enough to change
# ceil(grid_scale) at level 5 and thus the whole offset table.
PER_LEVEL_SCALE = float(hg_tcnn.tcnn_encoding.native_tcnn_module.hyperparams()["per_level_scale"])
print(f"  per_level_scale  : {PER_LEVEL_SCALE!r} (nerfstudio float64: {float(hg_tcnn.growth_factor)!r})")


def build_offset_table():
    """Transcription of GridEncodingTemplated's ctor (encodings/grid.h:693-724).
    Note the ordering: next_multiple(res^3, 8) FIRST, then min(..., 2^log2)."""
    log2s = np.float32(math.log2(PER_LEVEL_SCALE))
    offsets, resolutions, sizes = [], [], []
    offset = 0
    for lvl in range(NUM_LEVELS):
        # grid_scale / grid_resolution, common_device.h:918-927 (float math)
        scale = np.float32(np.exp2(np.float32(lvl) * log2s) * np.float32(MIN_RES)) - np.float32(1.0)
        resolution = int(math.ceil(float(scale))) + 1
        params_in_level = resolution**3
        params_in_level = ((params_in_level + 7) // 8) * 8
        params_in_level = min(params_in_level, 1 << LOG2_HASHMAP)
        offsets.append(offset)
        resolutions.append(resolution)
        sizes.append(params_in_level)
        offset += params_in_level
    offsets.append(offset)
    return offsets, resolutions, sizes, float(np.float32(log2s))


OFFSETS, RESOLUTIONS, SIZES, LOG2S = build_offset_table()
expected_n = OFFSETS[-1] * FEATURES_PER_LEVEL
print(f"  offset table     : total {OFFSETS[-1]} entries x {FEATURES_PER_LEVEL} features = {expected_n}")
assert expected_n == hg_tcnn.tcnn_encoding.params.numel(), (
    f"offset-table transcription wrong: {expected_n} != {hg_tcnn.tcnn_encoding.params.numel()}"
)
print("  offset table     : MATCHES tcnn's own n_params -- transcription validated")
for lvl in range(NUM_LEVELS):
    dense = RESOLUTIONS[lvl] ** 3 <= SIZES[lvl]
    print(
        f"    level {lvl:>2}: resolution={RESOLUTIONS[lvl]:>4}  size={SIZES[lvl]:>7}  "
        f"{'dense' if dense else 'hashed'}"
    )

M32 = (1 << 32) - 1
COHERENT_PRIME = [1, 2654435761, 805459861]


def grid_indices(pos_grid: torch.Tensor, lvl: int) -> torch.Tensor:
    """grid_index(), common_device.h:900-916, for one level.  pos_grid int64
    holding uint32 values, shape [..., 3]."""
    hashmap_size = SIZES[lvl]
    resolution = RESOLUTIONS[lvl]
    # replicate the `dim < 3 && stride <= hashmap_size` loop exactly
    stride = 1
    index = torch.zeros(pos_grid.shape[:-1], dtype=torch.int64, device=pos_grid.device)
    for dim in range(3):
        if stride > hashmap_size:
            break
        index = (index + pos_grid[..., dim] * stride) & M32
        stride = (stride * resolution) & M32
    if hashmap_size < stride:
        h = torch.zeros_like(index)
        for dim in range(3):
            h = torch.bitwise_xor(h, (pos_grid[..., dim] * COHERENT_PRIME[dim]) & M32)
        index = h
    return index % hashmap_size


def hashgrid_ref(x: torch.Tensor, params: torch.Tensor, emulate_half: bool = False) -> torch.Tensor:
    """kernel_grid (encodings/grid.h:64-170), Linear interpolation, N_POS_DIMS=3.

    pos_fract (common_device.h:1051): pos = fma(scale, input, 0.5);
    pos_grid = (uint32)(int)floor(pos); pos -= floor(pos).
    Accumulation in the kernel is `result = fma((T)weight, grid_val, result)`
    i.e. in T (half) -- `emulate_half=True` reproduces that.
    """
    outs = []
    for lvl in range(NUM_LEVELS):
        scale = float(np.float32(np.exp2(np.float32(lvl) * np.float32(LOG2S)) * np.float32(MIN_RES)) - np.float32(1.0))
        p = x * scale + 0.5
        fl = torch.floor(p)
        frac = p - fl
        pos_grid = fl.to(torch.int64)

        table = params[OFFSETS[lvl] * FEATURES_PER_LEVEL : OFFSETS[lvl + 1] * FEATURES_PER_LEVEL].reshape(
            SIZES[lvl], FEATURES_PER_LEVEL
        )
        acc = torch.zeros(x.shape[0], FEATURES_PER_LEVEL, device=x.device, dtype=torch.float32)
        for idx in range(8):
            weight = torch.ones(x.shape[0], device=x.device, dtype=torch.float32)
            local = pos_grid.clone()
            for dim in range(3):
                if (idx & (1 << dim)) == 0:
                    weight = weight * (1.0 - frac[..., dim])
                else:
                    weight = weight * frac[..., dim]
                    local[..., dim] = local[..., dim] + 1
            vals = table[grid_indices(local, lvl)]
            if emulate_half:
                acc = (acc + weight.to(GRID_HALF).float().unsqueeze(-1) * vals).to(GRID_HALF).float()
            else:
                acc = acc + weight.unsqueeze(-1) * vals
        outs.append(acc)
    return torch.cat(outs, dim=-1)


# Overwrite tcnn's ~1e-4 default init: otherwise everything sits near the fp16
# denormal floor and the comparison carries almost no information.
torch.manual_seed(21)
with torch.no_grad():
    hg_tcnn.tcnn_encoding.params.copy_(
        ((torch.rand_like(hg_tcnn.tcnn_encoding.params.float()) * 2 - 1)).to(GRID_HALF).float()
    )
params_f32 = hg_tcnn.tcnn_encoding.params.detach().to(GRID_HALF).float()
print(f"  params reset to U(-1,1), rounded to {GRID_HALF}")

print("\n  -- partition-of-unity invariant (layout-free) --")
with torch.no_grad():
    saved = hg_tcnn.tcnn_encoding.params.detach().clone()
    hg_tcnn.tcnn_encoding.params.fill_(0.5)
    torch.manual_seed(22)
    u = torch.rand(4096, 3, device=device)
    o = hg_tcnn(u).float()
    dev = (o - 0.5).abs().max().item()
    ok = dev <= 1e-3
    print(
        f"  [{'PASS' if ok else 'FAIL'}] constant grid (all=0.5) -> output must be 0.5 everywhere; "
        f"max deviation {dev:.3e}"
    )
    RESULTS.setdefault("HashEncoding", []).append(("partition of unity", ok, dev, 0.0))
    hg_tcnn.tcnn_encoding.params.copy_(saved)

print("\n  -- forward, batch-size sweep --")
for bs in (1, 31, 33, 127, 128, 129, 255, 256, 257, 4096):
    torch.manual_seed(700 + bs)
    u = torch.rand(bs, 3, device=device)
    with torch.no_grad():
        got = hg_tcnn(u)
        a = hashgrid_ref(u, params_f32, emulate_half=False)
        b = hashgrid_ref(u, params_f32, emulate_half=True)
        compare(f"forward bs={bs}", got, a, b, component="HashEncoding")

print("\n  -- comparison against nerfstudio's own torch HashEncoding (expected to disagree) --")
torch.manual_seed(31)
u = torch.rand(4096, 3, device=device)
with torch.no_grad():
    hg_torch_ns.hash_table.copy_(torch.rand_like(hg_torch_ns.hash_table) * 2 - 1)
    ns_out = hg_torch_ns(u)
    t_out = hg_tcnn(u).float()
    print(f"  nerfstudio-torch output std {ns_out.std().item():.3e}, tcnn output std {t_out.std().item():.3e}")
    print(
        "  NOTE: nerfstudio's torch hash encoding is NOT a valid reference for tcnn's HashGrid.\n"
        "        It scales by floor(min_res*growth^l) with no +0.5 stagger and always hashes\n"
        "        the full 2^log2_hashmap_size table, whereas tcnn uses\n"
        "        base_res*2^(l*log2 s) - 1 with the 0.5 offset and a dense/hashed split per\n"
        "        level.  The two therefore disagree on CORRECT hardware; the transcribed\n"
        "        reference above is the authority."
    )

print("\n  -- determinism --")
torch.manual_seed(32)
u_det = torch.rand(4096, 3, device=device)
det_ok = determinism("forward x3 (bs=4096)", lambda: hg_tcnn(u_det).detach().clone())
RESULTS.setdefault("HashEncoding", []).append(("forward determinism", det_ok, 0.0, 0.0))

print("\n  -- backward: d/dinput and d/dparams --")
for bs in (128, 4096):
    torch.manual_seed(800 + bs)
    u0 = torch.rand(bs, 3, device=device)
    torch.manual_seed(900 + bs)
    dout = (torch.rand(bs, NUM_LEVELS * FEATURES_PER_LEVEL, device=device) * 2 - 1)

    ut = u0.clone().requires_grad_(True)
    o = hg_tcnn(ut)
    hg_tcnn.zero_grad(set_to_none=True)
    o.backward(dout.to(o.dtype))
    g_in_tcnn = ut.grad.clone()
    g_par_tcnn = hg_tcnn.tcnn_encoding.params.grad.detach().float().clone()

    ua = u0.clone().requires_grad_(True)
    pa = params_f32.clone().requires_grad_(True)
    hashgrid_ref(ua, pa, emulate_half=False).backward(dout)
    g_in_a, g_par_a = ua.grad.clone(), pa.grad.clone()

    ub = u0.clone().requires_grad_(True)
    pb = params_f32.clone().requires_grad_(True)
    hashgrid_ref(ub, pb, emulate_half=True).backward(dout)
    g_in_b, g_par_b = ub.grad.clone(), pb.grad.clone()

    compare(f"backward d/dinput  bs={bs}", g_in_tcnn, g_in_a, g_in_b, component="HashEncoding")
    compare(f"backward d/dparams bs={bs}", g_par_tcnn, g_par_a, g_par_b, component="HashEncoding")


# =========================================================================== #
# verdict
# =========================================================================== #
banner("VERDICT")
overall = True
for comp in ("HashEncoding", "SHEncoding", "MLP", "MLP-Sigmoid", "MLP-ReLU-allrows"):
    checks = RESULTS.get(comp, [])
    failed = [c for c in checks if not c[1]]
    worst = max((c[2] for c in checks), default=0.0)
    ok = not failed
    if comp != "MLP-ReLU-allrows":  # informational: contaminated by ReLU-kink flips
        overall &= ok
    print(
        f"  {comp:<14}: {'CORRECT' if ok else 'NOT CORRECT'}  "
        f"({len(checks) - len(failed)}/{len(checks)} checks passed, worst |tcnn-a| = {worst:.3e})"
    )
    for label, passed, err, floor in failed:
        print(f"      FAILED: {label}  |tcnn-a|={err:.3e} floor={floor:.3e}")

print()
print(f"  OVERALL: tinycudann on this GPU is {'NUMERICALLY CORRECT' if overall else 'NOT NUMERICALLY CORRECT'}")
raise SystemExit(0 if overall else 1)
