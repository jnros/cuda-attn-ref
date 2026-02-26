attn_bench: attn_bench.cu
	nvcc -o $@ $< -g

clean:
	rm -f attn_bench

.PHONY: clean
