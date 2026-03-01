all: attn_bench attn_bench_naive attn_bench_cpu

attn_bench: attn_bench.cu
	nvcc -o $@ $< -g

attn_bench_naive: attn_bench_naive.cu
	nvcc -o $@ $< -g

attn_bench_cpu: attn_bench_cpu.cu
	nvcc -o $@ $< -g

clean:
	rm -f attn_bench attn_bench_naive attn_bench_cpu

.PHONY: clean
