import torch
import triton
import triton.language as tl
import sys

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
# shared memory per SM
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

'''
some visualisation

        GPU
+-------------------+
| SM 0              |
| block block block |
+-------------------+
| SM 1              |
| block block block |
+-------------------+
| SM 2              |
| block block block |
+-------------------+

'''

def naive_softmax(x):
    # x.shape = (M, N)
    # reads += M*N, writes += M
    x_max = x.max(dim=1)[0] # (M)
    # read += M*N + M, writes += M*N
    z = x - x_max[:, None]
    # read += M*N, write += M*N
    num = torch.exp(z)
    # read += M*N, write += M
    den = num.sum(dim=1)
    # read += M*N + M, write += M*N
    out = num/den[:, None]
    # total mem ops = 8*M*N + 4*M
    return out

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # no of progs/blocks can be less than n_rows so we need to skip by num_progs and finish that row calculation
    row_step = tl.num_programs(0)

    for row_idx in tl.range(pid, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)

        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols
        # (BLOCK_SIZE)
        row = tl.load(input_ptrs, mask=mask, other=float('-inf')) 
        # (BLOCK_SIZE)
        row_minus_max = row - tl.max(row, axis=0)
        num = tl.exp(row_minus_max)
        # (1)
        den = tl.sum(num, axis=0)
        # (BLOCK_SIZE)
        output = num/den

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, output ,mask=mask)
        


def softmax(x):
    n_rows, n_cols = x.shape

    # each block gets a row(assuming a row fits in smem)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # warps per block
    num_warps = 8
    # 
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    y = torch.empty_like(x)

    kernel = softmax_kernel.warmup(
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)
    )

    kernel._init_handles()
    # regs per thread
    n_regs = kernel.n_regs
    # total shared memory per block
    size_smem = kernel.metadata.shared

    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

    smem_occupancy = SIZE_SMEM // size_smem
    # prog = block here in triton
    # so calculating no of blocks/progs
    progs_per_sm = min(reg_occupancy, smem_occupancy)

    num_progs = min(NUM_SM * progs_per_sm, n_rows)

    grid = (num_progs, 1, 1)

    # either make x, y contiguous or use strides
    kernel[grid](
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols
    )

    return y



def test_softmax(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    x = torch.randn(size[0], size[1], device=DEVICE)
    z_triton = softmax(x)
    z_pytorch = torch.softmax(x, dim=1)

    torch.testing.assert_close(z_triton, z_pytorch, atol=atol, rtol=rtol)
    print('PASSED')


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch', 'naive_softmax'],
        line_names=["Triton", "Torch", "Naive Softmax"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={'M': 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    test_softmax(size=(1800, 500))
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark.run(save_path='.', print_data=False)