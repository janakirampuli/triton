import torch
import triton
import triton.language as tl
import sys

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def dropout_kernel(
    x_ptr, output_ptr,
    n_elements,
    p, seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1-p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, p, seed):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    dropout_kernel[grid](
        x, output,
        n_elements,
        p, seed,
        BLOCK_SIZE=1024
    )
    return output

def test_dropout(size: tuple, device=DEVICE):
    torch.manual_seed(0)
    n = size[0]
    x = torch.randn((n,), device=device)

    triton_res = dropout(x, p=0.5, seed=7)

    keep_ratio = (triton_res != 0).float().mean()

    assert abs(keep_ratio - 0.5) < 0.05
    print("PASSED")
    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 30)], 
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('green', '-'), ('red', '--')],
        ylabel='GB/s',
        plot_name='dropout-performance',
        args={}, 
    )
)
def benchmark(N, provider):
    x = torch.randn((N, ), device=DEVICE)
    
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.dropout(x, p=0.5, train=True))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: dropout(x, p=0.5, seed=7))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    N = 1024
    test_dropout(size=(N,))
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark.run(save_path='.', print_data=False)
 