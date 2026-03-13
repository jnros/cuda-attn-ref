# inf-cuda

Attention mechanism implementations in CUDA, built incrementally from scratch.
Paged Attention impl matches how vLLM manages KV cache memory.

## Files

| File | Description |
|------|-------------|
| `attn_bench_cpu.cu` | CPU-only baseline, no kernels |
| `attn_bench_naive.cu` | Single-threaded GPU kernel, everything in one pass |
| `attn_bench_paralogits.cu` | Parallelized Q·K logits computation, sequential softmax/output |
| `attn_bench.cu` | **Paged Attention** - KV cache scattered across physical frames via page table |

## Paged Attention

Final implementation (`attn_bench.cu`) demonstrates the core PagedAttention memory pattern:

- Token KV pairs stored in fixed-size pages (16 tokens/page)
- Page table maps logical pages → physical frames
- On-demand page allocation during token writes
- GPU receives flattened page table snapshot for lookups
- Kernels use `pa_get_token()` to transparently resolve token locations

