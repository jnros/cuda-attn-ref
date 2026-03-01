#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__
void attn_logits(__half *KV, 
		__half *Q, 
		float *scratch, //logits_raw, logits, weights
		int num_tokens, 
		int d,
		float *output)
{
	int i, j;
	float kk;
	float qq;
	float sum;
	float *attn_logits_raw;
	float *attn_logits;
	__half *kvk,*kvv;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > num_tokens) 
		return;

	kvk = (__half *) (KV + i * d * 2);
	sum = 0;	
	for (j = 0; j < d; j++) {
		qq = (float) Q[j];
		kk = (float) kvk[j];
		sum = sum + (qq * kk);
	}

	attn_logits_raw = (float *) scratch;
	attn_logits_raw = (float *) scratch + num_tokens;
	attn_logits_raw[i] = sum;
	attn_logits[i] = sum / sqrtf((float) d);
}

__global__
void attn(__half *KV, 
		__half *Q, 
		float *scratch, //logits_raw, logits, weights
		int num_tokens, 
		int d,
		float *output)
{
	__half *kvk,*kvv;
	float qq;
	float kk;
	float vv;
	float max, total, total2;

	float *attn_logits_raw;
	float *attn_logits;
	float *attn_weights;
	int i, j;
	
	attn_logits_raw = (float *) scratch;
	attn_logits = (float *) scratch + num_tokens;
	attn_weights = (float *) scratch + 2*num_tokens;
	for (i = 0; i < num_tokens; i++) {
		attn_logits_raw[i] = 0.0;
		attn_logits[i] = 0.0;
		attn_weights[i] = 0.0;
	}
	for (i = 0; i < d; i++)
		output[i] = 0.0;

	for (i = 0; i < num_tokens; i++) {
		kvk = (__half *) (KV + i * d * 2);
		for (j = 0; j < d; j++) {
			qq = (float) Q[j];
			kk = (float) kvk[j];
			attn_logits_raw[i] = attn_logits_raw[i] + (qq * kk);
		}
		if ((i % 32) == 0)
			printf("[%3d]:%6.2f  ", i, attn_logits_raw[i]);
	}
	printf("\n");

	printf("Scaled logits \n");
	for (i = 0; i < num_tokens; i++) {
		attn_logits[i] = attn_logits_raw[i] / sqrtf((float) d);
		if ((i % 32) == 0)
			printf("[%3d]:%6.2f  ", i, attn_logits[i]);
	}
	printf("\n");
	
	max = attn_logits[0];
	for (i = 1; i < num_tokens; i++)
		if (attn_logits[i] > max) 
			max = attn_logits[i];

	total = 0;
	for (i = 0; i < num_tokens; i++) {
		total = total + expf(attn_logits[i] - max);
	}

	total2 = 0;
	for (i = 0; i < num_tokens; i++) {
		attn_weights[i] = expf(attn_logits[i] - max) / total;
		total2 = total2 + attn_weights[i];
		if ((i % 32) == 0)
			printf("[%3d]:%13.10f/%13.10f \n", i, attn_weights[i], total2);
	}

	printf("Attention outputs\n");
	for (j = 0; j < d; j++) {
		output[j] = 0;
	}
	for (i = 0; i < num_tokens; i++) {
		kvv = (__half *) (KV + i * d * 2 + d);
		for (j = 0; j < d; j++) {
			vv = (float) kvv[j];
			output[j] = output[j] + (attn_weights[i] * vv);
		}
	}
	for (j = 0; j < d; j++) {
		printf("[%03d] %13.10f ",j,output[j]);
		if ((j % 4) == 3)
			printf("\n");
	}

	printf("Success!\n");
}

int main(void)
{
	__half *k;
	__half *kvk,*kvv;
	__half *q;
	__half *q_hbm;
	__half *k_hbm;
	float *scr_hbm;
	float *out_hbm;
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
	err = cudaMalloc(&q_hbm, d * sizeof(__half));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	for (j = 0; j < d; j++) {
		q[j] = (__half) (((float) rand()/ (float) RAND_MAX) - 0.5) * (__half) 1.0;
	}

	k = (__half *) malloc(sz);
	if (k == NULL) {
		fprintf(stderr, "malloc failed\n");
		return 1;
	}
	err = cudaMalloc(&k_hbm, sz);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	// KV dimens [num_tokens][2][d]
	// Packed as K then V per token, all 16 bit
	// Output layout 32 bit float [d]
	for (i = 0; i < num_tokens; i++) {
		kvk = (__half *) (k + i * d * 2);
		for (j = 0; j < d; j++)
			kvk[j] = (__half) (((float) rand()/ (float) RAND_MAX) - 0.5) * (__half) 1.0;
		
		kvv = kvk + d;
		for (j = 0; j < d; j++)
			kvv[j] = (__half) (((float) rand()/ (float) RAND_MAX) - 0.5) * (__half) 1.0;
		
	}
	printf("K & V initialized\n");

	// scratch pad [raw logits, logits, attn weights], each float[num_tokens]
	err = cudaMalloc(&scr_hbm, 3 * num_tokens * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc(&out_hbm, d * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	err = cudaMemcpy(k_hbm, k, sz, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	err = cudaMemcpy(q_hbm, q, d * sizeof(__half), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

   	attn<<<1,1>>>(k_hbm, q_hbm, scr_hbm, num_tokens, d, out_hbm);
	
	err = cudaGetLastError();
	if (err != cudaSuccess) {
    		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
    		return 1;
	}

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "kernel sync failed: %s\n", cudaGetErrorString(err));
		return 1;
	}

	cudaFree(q_hbm);
	cudaFree(k_hbm);
	cudaFree(scr_hbm);
	cudaFree(out_hbm);

	free(q);
	free(k);
	return 0;
}
