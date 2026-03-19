import torch
import triton
import triton.language as tl
import math
import sys

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def attn_fwd_inner(
    Q, M, L, O,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    block_index_QO,
    scale,
    stride_K_N, stride_V_N,
    offsets_QO_N,
    offsets_KV_N,
    N,
    BLOCK_SIZE_QO: tl.constexpr, 
    BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    d: tl.constexpr,
):
    if DIAGONAL:
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
    else:
        lo = 0
        hi = block_index_QO * BLOCK_SIZE_QO

    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo

    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)

        mask_KV_N = offsets_KV_N < N
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.0)
        S = tl.dot(Q, K_T) * scale

        if DIAGONAL:
            causal_mask = offsets_QO_N[:, None] >= offsets_KV_N[None, :]
            S = tl.where(causal_mask, S, float('-inf'))
        
        M_new = tl.maximum(M, tl.max(S, axis=1))
        S -= M_new[:, None]

        P = tl.exp2(S)
        L_new = tl.sum(P, axis=1)
        alpha = tl.exp2(M - M_new)
        L = L * alpha + L_new

        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.0)
        O *= alpha[:, None]
        O = tl.dot(P,V, acc=O)
        M = M_new
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV
    return O, L, M

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_QO": BLOCK_SIZE_QO, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages, num_warps=num_warps,
        )
        for BLOCK_SIZE_QO in [16, 32, 64, 128]
        for BLOCK_SIZE_KV in [16, 32, 64, 128]
        for num_stages in [3, 5, 7]
        for num_warps in [4, 8, 16]
        if BLOCK_SIZE_QO >= BLOCK_SIZE_KV and BLOCK_SIZE_QO % BLOCK_SIZE_KV == 0
    ],
    key=["d"]
)
@triton.jit
def attn_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    LSE_ptr,
    scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_d,
    stride_K_B, stride_K_H, stride_K_N, stride_K_d,
    stride_V_B, stride_V_H, stride_V_N, stride_V_d,
    stride_O_B, stride_O_H, stride_O_N, stride_O_d,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    B, H, N,
    d: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    # 1 / log_e(2)
    rln2: tl.constexpr = 1.4426950408889634
    # e^x = 2^(log_2(e^x))
    # e^x = 2^(x * log_2(e))
    # e^x = 2^(x * rln2)
    scale *= rln2

    tl.static_assert(BLOCK_SIZE_KV <= d)
    
    block_index_QO = tl.program_id(axis=0)

    index_BH = tl.program_id(axis=1)
    index_B = index_BH // H
    index_H = index_BH % H

    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H

    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_d = tl.arange(0, d)

    # (BLOCK_SIZE_QO, d)
    Q_offsets = offsets_QO_N[:, None] * stride_Q_N + offsets_d[None, :] * stride_Q_d
    # (d, BLOCK_SIZE_KV)
    K_T_offsets = offsets_d[:, None] * stride_K_d + offsets_KV_N[None, :] * stride_K_N
    # (BLOCK_SIZE_KV, d)
    V_offsets = offsets_KV_N[:, None] * stride_V_N + offsets_d[None, :] * stride_V_d

    mask_QO_N = offsets_QO_N < N
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.0)

    M = tl.full(shape=[BLOCK_SIZE_QO], value=float("-inf"), dtype=tl.float32)
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32)
    O = tl.zeros([BLOCK_SIZE_QO, d], dtype=tl.float32)

    O, L, M = attn_fwd_inner(
        Q, M, L, O,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        offsets_QO_N, offsets_KV_N,
        N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        False,
        d
    )

    O, L, M = attn_fwd_inner(
        Q, M, L, O,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        offsets_QO_N, offsets_KV_N,
        N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        True,
        d
    )

    O = O / L[:, None]

    LSE = M + tl.math.log2(L)
    LSE_offsets = index_B * stride_LSE_B + index_H * stride_LSE_H + offsets_QO_N * stride_LSE_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask)

    O_offsets = offsets_QO_N[:, None] * stride_O_N + offsets_d[None, :] * stride_O_d
    tl.store(O_ptr + O_offsets, O, mask=mask_QO_N[:, None])


class flashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        B, H, N, d = q.shape
        O = torch.empty_like(q)
        # log sum exp
        LSE = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

        grid = lambda args: (
            triton.cdiv(N, args["BLOCK_SIZE_QO"]),
            B * H
        )

        attn_fwd[grid](
            q, k, v, O, LSE,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, N, d
        )

        # for bwd
        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.grid = grid
        ctx.B, ctx.H, ctx.N, ctx.d = B, H, N, d
        ctx.scale = scale
        return O

triton_attention = flashattention.apply

configs = []
for mode in ["fwd"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["SEQ_LEN"],
            x_vals=[512 * i for i in range(1, 20)],
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=[
                "torch.attn",
                "triton.attn"
            ],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name=f"attention-fwd",
            args={"mode": mode}
        )
    )

@triton.testing.perf_report(configs)
def benchmark(SEQ_LEN, mode, provider, device=DEVICE):
    BATCH = 32
    N_HEADS = 4
    HEAD_DIM = 128
    v = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float32, device=device, requires_grad=True)
    q = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float32, device=device, requires_grad=True)
    k = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float32, device=device, requires_grad=True)
    sm_scale = 1 / math.sqrt(HEAD_DIM)
    if provider == "torch":
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    if provider == "triton":
        fn = lambda: triton_attention(q, k, v, sm_scale)
    ms = triton.testing.do_bench(fn)
    total_flops = 4.0 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM * 0.5
    return total_flops * 1e-12 / (ms * 1e-3)

def test_flashattention_kernel(B, H, N, d, device=DEVICE, atol=5e-3):
    q = torch.randn((B, H, N, d), dtype=torch.float32, device=device, requires_grad=True)
    k = torch.randn((B, H, N, d), dtype=torch.float32, device=device, requires_grad=True)
    v = torch.randn((B, H, N, d), dtype=torch.float32, device=device, requires_grad=True)
    scale = 1 / math.sqrt(d)
    tri_attn = triton_attention(q, k, v, scale)
    torch_attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(tri_attn, torch_attn, atol=atol, rtol=0)
    print("PASSED")


if __name__ == "__main__":
    # test_flashattention_kernel(1, 1, 128, 32)
    # test_flashattention_kernel(1, 1, 128, 64)
    # test_flashattention_kernel(1, 1, 128, 128)
    # test_flashattention_kernel(32, 4, 128, 32)
    # test_flashattention_kernel(32, 4, 125, 32)
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)
