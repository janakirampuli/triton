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
def flash_attn_v1_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, scale,
    O_ptr,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_d,
    stride_K_B, stride_K_H, stride_K_N, stride_K_d,
    stride_V_B, stride_V_H, stride_V_N, stride_V_d,
    stride_O_B, stride_O_H, stride_O_N, stride_O_d,
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

    # we don't need to load KV blocks that crossed the current Q block 
    if IS_CAUSAL:
        causal_steps = tl.cdiv((index_Q + 1) * BLOCK_SIZE_M, BLOCK_SIZE_N)
        num_steps = tl.minimum(causal_steps, num_steps)

    for start in range(0, num_steps):
        offsets_KV_N = start * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        K_offsets = offsets_KV_N[:, None] * stride_K_N + offsets_d[None, :] * stride_K_d
        V_offsets = offsets_KV_N[:, None] * stride_V_N + offsets_d[None, :] * stride_V_d

        mask_KV = (offsets_KV_N[:, None] < N) & (offsets_d[None, :] < d)

        K = tl.load(K_ptr + K_offsets, mask=mask_KV, other=0.0)
        V = tl.load(V_ptr + V_offsets, mask=mask_KV, other=0.0)

        QKT = tl.dot(Q, tl.trans(K)) * scale

        # boundary masking
        QKT = tl.where(offsets_KV_N[None, :] < N, QKT, float("-inf"))

        if IS_CAUSAL:
            QKT = tl.where(offsets_QO_M[:, None] >= offsets_KV_N[None, :], QKT, float("-inf"))

        M_new = tl.max(QKT, axis=1)
        M_new = tl.maximum(M, M_new)

        P = tl.exp(QKT - M_new[:, None])
        L_new = tl.sum(P, axis=1)

        alpha = tl.exp(M - M_new)
        L_new = alpha * L + L_new

        O = O * (L * alpha)[:, None]
        O = O + tl.dot(P.to(V.dtype), V)
        O = O / L_new[:, None]

        M = M_new
        L = L_new
    
    tl.store(O_ptr + O_offsets, O, mask=mask_QO)


class flashattention_v1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, is_causal=False):
        # batch_size, n_heads, seq_len, head_dim
        B, H, N, d = q.shape
        O = torch.empty_like(q)

        grid = lambda args: (
            B,
            H,
            triton.cdiv(N, args["BLOCK_SIZE_M"]),
        )

        flash_attn_v1_fwd_kernel[grid](
            q, k, v, scale,
            O,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            N, d=d, IS_CAUSAL=is_causal
        )

        return O

triton_attention_v1 = flashattention_v1.apply

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
                "triton.attn_v1"
            ],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name=f"causal-attention-v1-fwd",
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
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    if provider == "triton":
        fn = lambda: triton_attention_v1(q, k, v, sm_scale, True)
    ms = triton.testing.do_bench(fn)
    total_flops = 4.0 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM * 0.5 # (causal=True)
    return total_flops * 1e-12 / (ms * 1e-3)


def test_flashattention_kernel(B, H, N, d, device=DEVICE, atol=5e-3):
    q = torch.randn((B, H, N, d), dtype=torch.float16, device=device, requires_grad=True)
    k = torch.randn((B, H, N, d), dtype=torch.float16, device=device, requires_grad=True)
    v = torch.randn((B, H, N, d), dtype=torch.float16, device=device, requires_grad=True)
    scale = 1 / math.sqrt(d)
    tri_attn = triton_attention_v1(q, k, v, scale, True)
    torch_attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(tri_attn, torch_attn, atol=atol, rtol=0)
    print("PASSED")


if __name__ == "__main__":
    test_flashattention_kernel(1, 1, 128, 32)
    test_flashattention_kernel(1, 1, 128, 64)
    test_flashattention_kernel(1, 1, 128, 128)
    test_flashattention_kernel(32, 4, 128, 32)
    test_flashattention_kernel(32, 4, 125, 32)
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)
