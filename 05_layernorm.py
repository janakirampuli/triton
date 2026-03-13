import torch
import triton
import triton.language as tl
import sys

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def layernorm_fwd_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    stride_M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(axis=0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M

    sum_accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask = cols < N, other=0.0).to(tl.float32)
        sum_accumulator += x
    mean = tl.sum(sum_accumulator, axis=0) / N

    var_accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask = cols < N, other=0.0).to(tl.float32)
        diff = tl.where(cols < N, x - mean, 0.0)
        var_accumulator += diff * diff
    var = tl.sum(var_accumulator, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(w_ptr + cols, mask=mask)
        b = tl.load(b_ptr + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask)

        x_norm = (x - mean) * rstd
        y = x_norm * w + b

        tl.store(y_ptr + cols, y, mask=mask)

@triton.jit
def layernorm_bwd_dldw_dldb_kernel(
    dldw_inter_ptr,
    dldb_inter_ptr,
    dldw_ptr,
    dldb_ptr,
    GROUP_SIZE,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    col_offsets = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dldw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dldb_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_offsets = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_offsets[:, None] < GROUP_SIZE) & (col_offsets[None, :] < N)
        offsets = row_offsets[:, None] * N + col_offsets[None, :]

        dldw_acc += tl.load(dldw_inter_ptr + offsets, mask=mask, other=0.0)
        dldb_acc += tl.load(dldb_inter_ptr + offsets, mask=mask, other=0.0)

    dldw_chunk = tl.sum(dldw_acc, axis=0)
    dldb_chunk = tl.sum(dldb_acc, axis=0)

    tl.store(dldw_ptr + col_offsets, dldw_chunk, mask=col_offsets < N)
    tl.store(dldb_ptr + col_offsets, dldb_chunk, mask=col_offsets < N)


@triton.jit
def layernorm_bwd_dldx_kernel(
    x_ptr,
    dldx_ptr,
    dldy_ptr,
    w_ptr,
    dldw_inter_ptr,
    dldb_inter_ptr,
    mean_ptr,
    rstd_ptr,
    locks_ptr,
    stride,
    N,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptr += pid * stride
    dldx_ptr += pid * stride
    dldy_ptr += pid * stride
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dldy = tl.load(dldy_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)

    x_normed = tl.where(mask, (x - mean) * rstd, 0.0)
    dydx_normed = tl.where(mask, w * dldy, 0.0)
    c1 = tl.sum(x_normed * dydx_normed, axis=0) / N
    c2 = tl.sum(dydx_normed, axis=0) / N
    dldx = (dydx_normed - (x_normed * c1 + c2)) * rstd

    tl.store(dldx_ptr + cols, dldx, mask=mask)

    dldw_cont = (dldy * x_normed).to(w.dtype)
    dldb_cont = dldy.to(w.dtype)

    lock_id = pid % GROUP_SIZE
    locks_ptr += lock_id
    counts_ptr = locks_ptr + GROUP_SIZE

    dldw_inter_ptrs = dldw_inter_ptr + lock_id * N + cols
    dldb_inter_ptrs = dldb_inter_ptr + lock_id * N + cols

    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass


    count = tl.load(counts_ptr)
    if count == 0:
        tl.atomic_xchg(counts_ptr, 1)
    else:
        dldw_cont += tl.load(dldw_inter_ptrs, mask=mask)
        dldb_cont += tl.load(dldb_inter_ptrs, mask=mask)

    tl.store(dldw_inter_ptrs, dldw_cont, mask=mask)
    tl.store(dldb_inter_ptrs, dldb_cont, mask=mask)

    tl.atomic_xchg(locks_ptr, 0)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        normalised_shaped,
        weight,
        bias,
        eps
    ):
        M, N = x.reshape(-1, x.shape[-1]).shape
        y = torch.empty_like(x)
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)

        # Less than 64KB per feature: enqueue fused kernel
        # how many entries can we fit in SRAM if SRAM is 64KB?
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        layernorm_fwd_kernel[(M,)](
            x, y, weight, bias, mean, rstd,
            x.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

        return y

    @staticmethod
    def backward(
        ctx,
        dldy
    ):
        x, w, b, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape

        dldx = torch.empty_like(x)
        dldw = torch.empty_like(w)
        dldb = torch.empty_like(b)

        GROUP_SIZE = 64
        if N <= 8192: GROUP_SIZE = 96
        if N <= 4096:  GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256

        dldw_inter = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)
        dldb_inter = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=b.device)

        locks = torch.zeros(2 * GROUP_SIZE, dtype=torch.int32, device=w.device)

        layernorm_bwd_dldx_kernel[(M, )](
            x,
            dldx,
            dldy,
            w,
            dldw_inter,
            dldb_inter,
            mean,
            rstd,
            locks,
            x.stride(0),
            N,
            GROUP_SIZE=GROUP_SIZE,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )

        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)
        layernorm_bwd_dldw_dldb_kernel[grid](
            dldw_inter,
            dldb_inter,
            dldw,
            dldb,
            min(GROUP_SIZE, M),
            N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128,
        )

        return dldx, None, dldw, dldb, None

layernorm = LayerNorm.apply

def test_layernorm(size: tuple, dtype, eps=1e-5, device=DEVICE):
    M, N = size[0], size[1]
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    x.requires_grad_(True)
    weight = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    y_triton = layernorm(x, (N, ), weight, bias, eps)
    y_torch = torch.layer_norm(x, (N, ), weight, bias, eps)
    torch.testing.assert_close(y_triton, y_torch, atol=1e-2, rtol=0)
    print("FWD PASSED")

    dldy = 0.1 * torch.randn_like(x)
    y_triton.backward(dldy, retain_graph=True)
    dldx_triton, dldw_triton, dldb_triton = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    y_torch.backward(dldy, retain_graph=True)
    dldx_torch, dldw_torch, dldb_torch = [_.grad.clone() for _ in [x, weight, bias]]

    torch.testing.assert_close(dldx_triton, dldx_torch, atol=1e-2, rtol=0)
    torch.testing.assert_close(dldw_triton, dldw_torch, atol=1e-2, rtol=0)
    torch.testing.assert_close(dldb_triton, dldb_torch, atol=1e-2, rtol=0)
    print("BWD PASSED")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-bwd',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    )
)
def benchmark(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    x_shape = (M, N)
    w_shape = (N, )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)

    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dldy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.05, 0.95]

    def y_fwd():
        if provider == 'triton':
            return layernorm(x, w_shape, weight, bias, eps)
        if provider == 'torch':
            return torch.layer_norm(x, w_shape, weight, bias, eps)

    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=100)
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dldy, retain_graph=True), quantiles=quantiles, grad_to_none=[x, weight, bias], rep=100)
        
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_layernorm((1234, 4321), dtype=torch.float16)
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)
