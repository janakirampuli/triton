import torch
import triton
import triton.language as tl
import math
import sys

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

'''
shared memory per SM in A10 gpu: 100.00 KB = 102400 B
we're loading Q(BLOCK_SIZE_M, d), O(BLOCK_SIZE_M, d), K(BLOCK_SIZE_N, d), V(BLOCK_SIZE_N, d)
triton buffers K, V by num_stages (S) and we're using fp16
bytes = (2 * B_M * d + 2 * S * B_N * d) * (2)
let's take d = 128 => 512 * B_M + 512 * B_N * S <= 102400
'''
@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32}, num_stages=5, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16}, num_stages=5, num_warps=4),
    ],
    key=["N"]
)
@triton.jit
def flash_attn_v2_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, scale,
    O_ptr,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_d,
    stride_K_B, stride_K_H, stride_K_N, stride_K_d,
    stride_V_B, stride_V_H, stride_V_N, stride_V_d,
    stride_O_B, stride_O_H, stride_O_N, stride_O_d,
    LSE_ptr,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    d: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    index_B = tl.program_id(axis=0)
    index_H = tl.program_id(axis=1)
    index_Q = tl.program_id(axis=2)

    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H

    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H

    offsets_QO_M = index_Q * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    offsets_d = tl.arange(0, d)

    # (BLOCK_SIZE_M, d)
    Q_offsets = offsets_QO_M[:, None] * stride_Q_N + offsets_d[None, :] * stride_Q_d
    # (BLOCK_SIZE_M, d)
    O_offsets = offsets_QO_M[:, None] * stride_O_N + offsets_d[None, :] * stride_O_d

    mask_QO = (offsets_QO_M[:, None] < N) & (offsets_d[None, :] < d)

    M = tl.full([BLOCK_SIZE_M], value=float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    O = tl.zeros([BLOCK_SIZE_M, d], dtype=tl.float32)

    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO, other = 0.0)

    num_steps = tl.cdiv(N, BLOCK_SIZE_N)

    # causal split: full KV blocks (no causal mask) + boundary blocks (causal mask)
    if IS_CAUSAL:
        causal_steps = tl.cdiv((index_Q + 1) * BLOCK_SIZE_M, BLOCK_SIZE_N)
        num_steps = tl.minimum(causal_steps, num_steps)
        num_full_steps = (index_Q * BLOCK_SIZE_M) // BLOCK_SIZE_N
        num_full_steps = tl.minimum(num_full_steps, num_steps)
    else:
        num_full_steps = num_steps

    # Stage 1: full blocks, no causal mask
    for start in range(0, num_full_steps):
        offsets_KV_N = start * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        K_offsets = offsets_KV_N[:, None] * stride_K_N + offsets_d[None, :] * stride_K_d
        V_offsets = offsets_KV_N[:, None] * stride_V_N + offsets_d[None, :] * stride_V_d

        mask_KV = (offsets_KV_N[:, None] < N) & (offsets_d[None, :] < d)

        K = tl.load(K_ptr + K_offsets, mask=mask_KV, other=0.0)
        V = tl.load(V_ptr + V_offsets, mask=mask_KV, other=0.0)

        QKT = tl.dot(Q, tl.trans(K)) * scale

        # boundary masking
        QKT = tl.where(offsets_KV_N[None, :] < N, QKT, float("-inf"))

        M_new = tl.max(QKT, axis=1)
        M_new = tl.maximum(M, M_new)

        P = tl.exp(QKT - M_new[:, None])
        L_new = tl.sum(P, axis=1)

        alpha = tl.exp(M - M_new)
        L_new = alpha * L + L_new

        O = O * alpha[:, None]
        O = O + tl.dot(P.to(V.dtype), V)

        M = M_new
        L = L_new

    # Stage 2: causal-masked blocks (contains diagonal/tail)
    for start in range(num_full_steps, num_steps):
        offsets_KV_N = start * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        K_offsets = offsets_KV_N[:, None] * stride_K_N + offsets_d[None, :] * stride_K_d
        V_offsets = offsets_KV_N[:, None] * stride_V_N + offsets_d[None, :] * stride_V_d

        mask_KV = (offsets_KV_N[:, None] < N) & (offsets_d[None, :] < d)

        K = tl.load(K_ptr + K_offsets, mask=mask_KV, other=0.0)
        V = tl.load(V_ptr + V_offsets, mask=mask_KV, other=0.0)

        QKT = tl.dot(Q, tl.trans(K)) * scale

        # boundary masking
        QKT = tl.where(offsets_KV_N[None, :] < N, QKT, float("-inf"))
        QKT = tl.where(offsets_QO_M[:, None] >= offsets_KV_N[None, :], QKT, float("-inf"))

        M_new = tl.max(QKT, axis=1)
        M_new = tl.maximum(M, M_new)

        P = tl.exp(QKT - M_new[:, None])
        L_new = tl.sum(P, axis=1)

        alpha = tl.exp(M - M_new)
        L_new = alpha * L + L_new

        O = O * alpha[:, None]
        O = O + tl.dot(P.to(V.dtype), V)

        M = M_new
        L = L_new
    O = O / L[:, None]
    tl.store(O_ptr + O_offsets, O.to(O_ptr.dtype.element_ty), mask=mask_QO)

    # store LSE = M + log(L)
    LSE_ptr += index_B * stride_LSE_B + index_H * stride_LSE_H
    LSE = M + tl.log(L)
    tl.store(LSE_ptr + offsets_QO_M * stride_LSE_N, LSE, mask=offsets_QO_M < N)


@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32}, num_stages=4, num_warps=4),
    ],
    key=["N"],
    # Backward uses atomic adds for dQ. During autotune, Triton runs multiple
    # candidate configs on the same buffers, so we must reset outputs between
    # trials to avoid accumulating gradients across benchmark runs. 
    #                                                       -- gpt-5.3-codex
    reset_to_zero=["dQ_ptr", "dK_ptr", "dV_ptr"]
)
@triton.jit
def flash_attn_v2_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, scale,
    O_ptr, dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    LSE_ptr, D_ptr,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_d,
    stride_K_B, stride_K_H, stride_K_N, stride_K_d,
    stride_V_B, stride_V_H, stride_V_N, stride_V_d,
    stride_O_B, stride_O_H, stride_O_N, stride_O_d,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    stride_D_B, stride_D_H, stride_D_N,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    d: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    index_B = tl.program_id(axis=0)
    index_H = tl.program_id(axis=1)
    index_K = tl.program_id(axis=2)

    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H

    dQ_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    dK_ptr += index_B * stride_K_B + index_H * stride_K_H
    dV_ptr += index_B * stride_V_B + index_H * stride_V_H

    O_ptr += index_B * stride_O_B + index_H * stride_O_H
    dO_ptr += index_B * stride_O_B + index_H * stride_O_H

    LSE_ptr += index_B * stride_LSE_B + index_H * stride_LSE_H
    D_ptr += index_B * stride_D_B + index_H * stride_D_H 

    offsets_d = tl.arange(0, d)
    offsets_K = index_K * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    K_offsets = offsets_K[:, None] * stride_K_N + offsets_d[None, :] * stride_K_d
    V_offsets = offsets_K[:, None] * stride_V_N + offsets_d[None, :] * stride_V_d

    mask_K = offsets_K < N
    mask_KV = mask_K[:, None] & (offsets_d[None, :] < d)

    K = tl.load(K_ptr + K_offsets, mask=mask_KV, other=0.0)
    V = tl.load(V_ptr + V_offsets, mask=mask_KV, other=0.0)

    dK = tl.zeros([BLOCK_SIZE_N, d], dtype=tl.float32)
    dV = tl.zeros([BLOCK_SIZE_N, d], dtype=tl.float32)

    start_q = 0
    if IS_CAUSAL:
        start_q = (index_K * BLOCK_SIZE_N) // BLOCK_SIZE_M

    num_q_blocks = tl.cdiv(N, BLOCK_SIZE_M)

    # causal split for backward:
    # 1) potentially partial q-blocks that need causal masking
    # 2) fully valid q-blocks where causal mask is unnecessary
    if IS_CAUSAL:
        partial_q_end = tl.cdiv((index_K + 1) * BLOCK_SIZE_N, BLOCK_SIZE_M)
        partial_q_end = tl.minimum(partial_q_end, num_q_blocks)
    else:
        partial_q_end = start_q

    # Stage 1: causal-masked q-blocks
    for index_Q in range(start_q, partial_q_end):
        offsets_Q = index_Q * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_Q = offsets_Q < N
        mask_QO = mask_Q[:, None] & (offsets_d[None, :] < d)

        Q_offsets = offsets_Q[:, None] * stride_Q_N + offsets_d[None, :] * stride_Q_d
        O_offsets = offsets_Q[:, None] * stride_O_N + offsets_d[None, :] * stride_O_d

        Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO, other=0.0)
        dO = tl.load(dO_ptr + O_offsets, mask=mask_QO, other=0.0)

        LSE = tl.load(LSE_ptr + offsets_Q * stride_LSE_N, mask=mask_Q, other=0.0)
        D = tl.load(D_ptr + offsets_Q * stride_D_N, mask=mask_Q, other=0.0)

        S = tl.dot(Q, tl.trans(K)) * scale

        valid_mask = (mask_Q[:, None] & mask_K[None, :]) & (offsets_Q[:, None] >= offsets_K[None, :])

        S = tl.where(valid_mask, S, float("-inf"))

        # reconstruct P using LSE: P = exp(S - LSE)
        P = tl.exp(S - LSE[:, None])
        P = tl.where(valid_mask, P, 0.0)

        # dV += P^T @ DO
        dV += tl.dot(tl.trans(P).to(V.dtype), dO)
        # dS = P * (dP - D)
        # => dS = P * (dO @ V^T - D)
        dS = P * (tl.dot(dO, tl.trans(V)) - D[:, None])
        dS = tl.where(valid_mask, dS, 0.0)

        # dQ += dS @ K * scale
        dQ_block = tl.dot(dS.to(K.dtype), K) * scale

        tl.atomic_add(dQ_ptr + Q_offsets, dQ_block.to(tl.float32), mask=mask_QO)

        # dK += dS^T @ Q * scale
        dK += tl.dot(tl.trans(dS).to(Q.dtype), Q) * scale

    # Stage 2: fully valid q-blocks (no causal mask needed)
    for index_Q in range(partial_q_end, num_q_blocks):
        offsets_Q = index_Q * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_Q = offsets_Q < N
        mask_QO = mask_Q[:, None] & (offsets_d[None, :] < d)

        Q_offsets = offsets_Q[:, None] * stride_Q_N + offsets_d[None, :] * stride_Q_d
        O_offsets = offsets_Q[:, None] * stride_O_N + offsets_d[None, :] * stride_O_d

        Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO, other=0.0)
        dO = tl.load(dO_ptr + O_offsets, mask=mask_QO, other=0.0)

        LSE = tl.load(LSE_ptr + offsets_Q * stride_LSE_N, mask=mask_Q, other=0.0)
        D = tl.load(D_ptr + offsets_Q * stride_D_N, mask=mask_Q, other=0.0)

        S = tl.dot(Q, tl.trans(K)) * scale

        valid_mask = mask_Q[:, None] & mask_K[None, :]
        S = tl.where(valid_mask, S, float("-inf"))

        # reconstruct P using LSE: P = exp(S - LSE)
        P = tl.exp(S - LSE[:, None])
        P = tl.where(valid_mask, P, 0.0)

        # dV += P^T @ DO
        dV += tl.dot(tl.trans(P).to(V.dtype), dO)
        # dS = P * (dP - D)
        # => dS = P * (dO @ V^T - D)
        dS = P * (tl.dot(dO, tl.trans(V)) - D[:, None])
        dS = tl.where(valid_mask, dS, 0.0)

        # dQ += dS @ K * scale
        dQ_block = tl.dot(dS.to(K.dtype), K) * scale

        tl.atomic_add(dQ_ptr + Q_offsets, dQ_block.to(tl.float32), mask=mask_QO)

        # dK += dS^T @ Q * scale
        dK += tl.dot(tl.trans(dS).to(Q.dtype), Q) * scale
    
    tl.store(dK_ptr + K_offsets, dK.to(dK_ptr.dtype.element_ty), mask=mask_KV)
    tl.store(dV_ptr + V_offsets, dV.to(dV_ptr.dtype.element_ty), mask=mask_KV)


class flashattention_v2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, is_causal=False):
        # batch_size, n_heads, seq_len, head_dim
        B, H, N, d = q.shape
        O = torch.zeros_like(q)

        LSE = torch.zeros((B, H, N), device=q.device, dtype=torch.float32)

        grid = lambda args: (
            B,
            H,
            triton.cdiv(N, args["BLOCK_SIZE_M"]),
        )

        flash_attn_v2_fwd_kernel[grid](
            q, k, v, scale,
            O,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE,
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            N, d=d, IS_CAUSAL=is_causal
        )

        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.scale = scale
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, O, LSE = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal

        B, H, N, d = q.shape

        # We keep initialization as fp32 for neat float32 Triton atomic adds
        dQ = torch.zeros_like(q, dtype=torch.float32)
        dK = torch.zeros_like(k, dtype=torch.float32)
        dV = torch.zeros_like(v, dtype=torch.float32)

        D = (dO.float() * O.float()).sum(dim=-1)

        grid = lambda args: (
            B,
            H,
            triton.cdiv(N, args["BLOCK_SIZE_N"])
        )

        flash_attn_v2_bwd_kernel[grid](
            q, k, v, scale,
            O, dO, dQ, dK, dV,
            LSE, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            N, d=d, IS_CAUSAL=is_causal
        )

        return dQ.to(q.dtype), dK.to(k.dtype), dV.to(v.dtype), None, None


triton_attention_v2 = flashattention_v2.apply

configs = []
for mode in ["fwd", "bwd", "fwd_bwd"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["SEQ_LEN"],
            x_vals=[512 * i for i in range(1, 15)],
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=[
                "torch.attn",
                "triton.attn_v2"
            ],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name=f"causal-attention-v2-{mode}",
            args={"mode": mode}
        )
    )
@triton.testing.perf_report(configs)
def benchmark(SEQ_LEN, mode, provider, device=DEVICE):
    BATCH = 32
    N_HEADS = 4
    HEAD_DIM = 128
    v = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float16, device=device, requires_grad=True)
    q = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float16, device=device, requires_grad=True)
    k = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float16, device=device, requires_grad=True)
    sm_scale = 1 / math.sqrt(HEAD_DIM)
    if provider == "torch":
        fwd_fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    if provider == "triton":
        fwd_fn = lambda: triton_attention_v2(q, k, v, sm_scale, True)

    out = fwd_fn()
    dO = torch.randn_like(out)

    if mode == "fwd":
        fn = fwd_fn
    elif mode == "bwd":
        fn = lambda: out.backward(dO, retain_graph=True)
    else:
        def fn():
            out = fwd_fn()
            out.backward(dO)

    ms = triton.testing.do_bench(fn)

    flops_per_fwd = 4.0 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM * 0.5 
    if mode == "fwd":
        total_flops = flops_per_fwd
    elif mode == "bwd":
        # bwd pass is 2x the FLOPs of fwd pass
        total_flops = 2 * flops_per_fwd
    else: # fwd_bwd
        total_flops = 3 * flops_per_fwd
        
    return total_flops * 1e-12 / (ms * 1e-3)


def test_flashattention_kernel(B, H, N, d, device=DEVICE, atol=5e-3):
    q = torch.randn((B, H, N, d), dtype=torch.float16, device=device, requires_grad=True)
    k = torch.randn((B, H, N, d), dtype=torch.float16, device=device, requires_grad=True)
    v = torch.randn((B, H, N, d), dtype=torch.float16, device=device, requires_grad=True)
    scale = 1 / math.sqrt(d)

    tri_attn = triton_attention_v2(q, k, v, scale, True)

    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)

    torch_attn = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    torch.testing.assert_close(tri_attn, torch_attn, atol=atol, rtol=0)
    print("FWD PASSED")

    dO = torch.randn_like(tri_attn)
    tri_attn.backward(dO)
    torch_attn.backward(dO)

    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=0)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=0)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=0)

    print("BWD PASSED")


if __name__ == "__main__":
    # test_flashattention_kernel(1, 1, 128, 32)
    # test_flashattention_kernel(1, 1, 128, 64)
    # test_flashattention_kernel(1, 1, 128, 128)
    # test_flashattention_kernel(32, 4, 128, 32)
    # test_flashattention_kernel(32, 4, 125, 32)
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)
