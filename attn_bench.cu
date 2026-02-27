#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

int main(void)
{
	__half *ptr;
	__half *k,*kv_hbm;
	__half *q_hbm;
	__half *kvk,*kvv;
	__half *q;
	float qq;
	float kk;

	float *attention_scores;
	int i, j, d, sz, num_tokens;
	cudaError_t err;

	srand((unsigned int)time(NULL));

	num_tokens = 256;
	d = 64;
	sz = num_tokens * 2 * d * sizeof(__half);

	q = (__half *) malloc(d * sizeof(__half));
	if (q == NULL) {
		fprintf(stderr, "malloc failed\n");
		return 1;
	}

	for (j = 0; j < d; j++) {
		q[j] = (__half) (((float) rand()/ (float) RAND_MAX) - 0.5) * (__half) 1.0;
	}

	printf("Q initialized\n");
	k = (__half *) malloc(sz);
	if (k == NULL) {
		fprintf(stderr, "malloc failed\n");
		return 1;
	}

	for (i = 0; i < num_tokens; i++) {
		kvk = (__half *) (k + i * d * 2);
		for (j = 0; j < d; j++)
			kvk[j] = (__half) (((float) rand()/ (float) RAND_MAX) - 0.5) * (__half) 1.0;
		
		kvv = kvk + d;
		for (j = 0; j < d; j++)
			kvv[j] = (__half) (((float) rand()/ (float) RAND_MAX) - 0.5) * (__half) 1.0;
		
	}
	printf("K & V initialized\n");

	attention_scores = (float *) malloc(num_tokens * sizeof(float));
	if (attention_scores == NULL) {
		fprintf(stderr, "malloc failed\n");
		return 1;
	}
		
	printf("Attention scores \n");
	for (i = 0; i < num_tokens; i++) {
		attention_scores[i] = 0.0;

		kvk = (__half *) (k + i * d * 2);
		for (j = 0; j < d; j++) {
			qq = (float) q[j];
			kk = (float) kvk[j];
			attention_scores[i] = attention_scores[i] + (qq * kk);
		}
		printf("[%d]:%6.2f  ", i, attention_scores[i]);
	}
	
	printf("Success!\n");
#if 0
	err = cudaMalloc(&q_hbm, d * sizeof(__half));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	err = cudaMemcpy(q_hbm, q, d * sizeof(__half), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc(&ptr, sz);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}
	kv_hbm = ptr;

	err = cudaMemcpy(kv_hbm, k, sz, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}
	printf("Success!\n");	

	err = cudaFree(q_hbm);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaFree failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	err = cudaFree(ptr);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaFree failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}
#endif

	return 0;
}
