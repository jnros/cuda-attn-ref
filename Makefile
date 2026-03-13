all: attn_bench attn_bench_paralogits attn_bench_naive attn_bench_cpu

attn_bench: attn_bench.cu
	nvcc -o $@ $< -g -G -O0

attn_bench_paralogits: attn_bench_paralogits.cu
	nvcc -o $@ $< -g -G -O0

attn_bench_naive: attn_bench_naive.cu
	nvcc -o $@ $< -g -G -O0

attn_bench_cpu: attn_bench_cpu.cu
	nvcc -o $@ $< -g -G -O0

clean:
	rm -f attn_bench attn_bench_naive attn_bench_cpu

.PHONY: clean
