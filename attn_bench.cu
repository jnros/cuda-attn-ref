#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
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
	float vv;
	float max, total, total2;

	float *attn_logits_raw;
	float *attn_logits;
	float *attn_weights;
	float *attn_output;
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

	attn_logits_raw = (float *) malloc(num_tokens * sizeof(float));
	if (attn_logits_raw == NULL) {
		fprintf(stderr, "malloc failed\n");
		return 1;
	}
		
	printf("Attention scores \n");
	for (i = 0; i < num_tokens; i++) {
		attn_logits_raw[i] = 0.0;

		kvk = (__half *) (k + i * d * 2);
		for (j = 0; j < d; j++) {
			qq = (float) q[j];
			kk = (float) kvk[j];
			attn_logits_raw[i] = attn_logits_raw[i] + (qq * kk);
		}
		if ((i % 32) == 0)
			printf("[%3d]:%6.2f  ", i, attn_logits_raw[i]);
	}
	printf("\n");

	attn_logits = (float *) malloc(num_tokens * sizeof(float));
	if (attn_logits == NULL) {
		fprintf(stderr, "malloc failed\n");
		return 1;
	}

	printf("Scaled logits \n");
	for (i = 0; i < num_tokens; i++) {
		attn_logits[i] = attn_logits_raw[i] / sqrt(d);
		if ((i % 32) == 0)
			printf("[%3d]:%6.2f  ", i, attn_logits[i]);
	}
	printf("\n");
	
	attn_weights = (float *) malloc(num_tokens * sizeof(float));
	if (attn_weights == NULL) {
		fprintf(stderr, "malloc failed\n");
		return 1;
	}
	max = 0;
	for (i = 0; i < num_tokens; i++) {
		if (attn_logits[i] > max) 
			max = attn_logits[i];
	}

	total = 0;
	for (i = 0; i < num_tokens; i++) {
		total = total + exp(attn_logits[i] - max);
	}

	total2 = 0;
	for (i = 0; i < num_tokens; i++) {
		attn_weights[i] = exp(attn_logits[i] - max) / total;
		total2 = total2 + attn_weights[i];
		if ((i % 32) == 0)
			printf("[%3d]:%13.10f/%13.10f \n", i, attn_weights[i], total2);
	}

	attn_output = (float *) malloc(num_tokens * sizeof(float));
	if (attn_output == NULL) {
		fprintf(stderr, "malloc failed\n");
		return 1;
	}

	printf("Attention outputs\n");
	for (i = 0; i < num_tokens; i++) {
		kvv = (__half *) (k + i * d * 2 + d);
		attn_output[i] = 0;
		for (j = 0; j < d; j++) {
			kk = (float) kvv[j];
			attn_output[i] = attn_output[i] + (attn_weights[i] * kk);
		}
		if ((i % 32) == 0)
			printf("[%3d]:%13.10f/%13.10f \n", i, attn_weights[i], attn_output[i]);
	}
	printf("\n");

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
