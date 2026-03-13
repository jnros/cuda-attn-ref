#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define PA_BLOCK_SIZE 16 //tokens per page
#define PA_NUM_PAGES 256

typedef struct {
	int physical_frame; //-1 if not mapped
} pa_page_t;

typedef struct {
	pa_page_t ptable[PA_NUM_PAGES];
	int free_list[PA_NUM_PAGES];
	int num_free;
} pa_pool_t;

pa_pool_t pa_pool;
__half *kvpool_dev;

int pa_pt_init(int d)
{
	cudaError_t err;
	int i;

	err = cudaMalloc(&kvpool_dev, PA_NUM_PAGES * PA_BLOCK_SIZE * 2 * d * sizeof(__half));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	pa_pool.num_free = PA_NUM_PAGES;
	for (i = 0; i < PA_NUM_PAGES; i++) {
		pa_pool.ptable[i].physical_frame = -1;
		pa_pool.free_list[i] = i;
	}

	return 0;
}

int pa_alloc_frame(void) {
	pa_pool_t *pool;
	int n;

	pool = &pa_pool;
	if (pool->num_free == 0) {
		fprintf(stderr, "out of frame memory\n");
		return -1;
	}
	
	pool->num_free = pool->num_free - 1;

	return pool->free_list[pool->num_free];
}

int pa_map_page(int logical)
{
	pa_pool_t *pool;
	pa_page_t *pg;
	int pframe;

	pframe = pa_alloc_frame();
	if (pframe == -1)
		return -1;

	pool = &pa_pool;
	pool->ptable[logical].physical_frame = pframe;

	return 0;
}

int pa_free_page(int logical)
{
    // TODO: return frame to free_list
    // pa_pool.free_list[pa_pool.num_free++] = phys;
    // pa_pool.ptable[logical].physical_frame = -1;
}

int pa_write_token(int token_idx, __half *kv_data, int d)
{
	__half *dst;
	int logical_page = token_idx / PA_BLOCK_SIZE;
	int slot = token_idx % PA_BLOCK_SIZE;
	int phys = pa_pool.ptable[logical_page].physical_frame;
	int rc;

	if (phys == -1) {
		rc = pa_map_page(logical_page);
		if (rc)
			return -1;
		phys = pa_pool.ptable[logical_page].physical_frame;
	}

	dst = kvpool_dev + (phys * PA_BLOCK_SIZE + slot) * 2 * d;

	cudaMemcpy(dst, kv_data, 2*d*sizeof(__half), cudaMemcpyHostToDevice);
	return 0;
}

__device__
__half* pa_get_token(int token_idx, __half *kvpool_dev, int *page_table, int d) {
    int logical_page = token_idx / PA_BLOCK_SIZE;
    int slot = token_idx % PA_BLOCK_SIZE;

    int phys = page_table[logical_page];
    return kvpool_dev + (phys * PA_BLOCK_SIZE + slot) * 2 * d;
}

__global__
void attn_logits(__half *pool_dev, //k and v  
		__half *Q, 
		float *scratch, //page_table, logits_raw, logits, weights
		int num_tokens, 
		int d)
{
	int i, j;
	float kk;
	float qq;
	float sum;
	int *pt_flat;
	float *attn_logits_raw;
	float *attn_logits;
	__half *kvk;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_tokens) 
		return;
	if ((i % 4) == 1)
		printf("attn_logits thread i = %3d ", i);

	pt_flat = (int *) scratch;
	// kvk = (__half *) (KV + i * d * 2);
	kvk = pa_get_token(i, pool_dev, pt_flat, d);
	if (kvk == NULL)
		return;

	sum = 0;	
	for (j = 0; j < d; j++) {
		qq = __half2float(Q[j]);
		kk = __half2float(kvk[j]);
		sum = sum + (qq * kk);
	}

	attn_logits_raw = (float *) scratch  + (PA_NUM_PAGES);
	attn_logits = attn_logits_raw + num_tokens;
	attn_logits_raw[i] = sum;
	attn_logits[i] = sum / sqrtf((float) d);
}

__global__
void attn(__half *pool_dev, 
		float *scratch, //page_table, logits_raw, logits, weights
		int num_tokens, 
		int d,
		float *output)
{
	__half *kvv;
	float vv;
	float max, total, total2;
	float *attn_logits_raw;
	float *attn_logits;
	float *attn_weights;
	int *pt_flat, i, j;
	
	pt_flat = (int *) scratch;
	attn_logits_raw = (float *) scratch + (PA_NUM_PAGES);
	attn_logits = (float *) attn_logits_raw + num_tokens;
	attn_weights = (float *) attn_logits_raw + 2 * num_tokens;

	printf("Scaled logits \n");
	for (i = 0; i < num_tokens; i++) 
		if ((i % 32) == 0)
			printf("[%3d]:%6.2f  ", i, attn_logits[i]);
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
		/* kvv = (__half *) (KV + i * d * 2 + d); */
		kvv = pa_get_token(i, pool_dev, pt_flat, d);
		kvv = kvv + d;
		for (j = 0; j < d; j++) {
			vv = __half2float(kvv[j]);
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
	__half *q_dev;
	float *scr_dev;
	float *out_dev;
	int i, j, d, sz, num_tokens;
	int block, grid;
	int pt_flat[PA_NUM_PAGES];
	int rc;
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

	// Init page table and alloc dev mem for KV cache
	pa_pt_init(d);

	err = cudaMalloc(&q_dev, d * sizeof(__half));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	// Init to random 
	for (j = 0; j < d; j++) 
		q[j] = __float2half(((float) rand()/ RAND_MAX) - 0.5);

	k = (__half *) malloc(sz);
	if (k == NULL) {
		fprintf(stderr, "malloc failed\n");
		return 1;
	}
	// KV dimens [num_tokens][2][d]
	// Packed as K then V per token, all 16 bit
	// Output layout 32 bit float [d]
	for (i = 0; i < num_tokens; i++) {
		kvk = (__half *) (k + i * d * 2);
		for (j = 0; j < d; j++)
			kvk[j] = __float2half(((float) rand()/ RAND_MAX) - 0.5);
		
		kvv = kvk + d;
		for (j = 0; j < d; j++)
			kvv[j] = __float2half(((float) rand()/ RAND_MAX) - 0.5);
		rc = pa_write_token(i, kvk, d);
		if (rc != 0) {
			fprintf(stderr, "failed to write token\n");
			return 1;
		}
	}

#if 0	
	err = cudaMalloc(&k_dev, sz);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	err = cudaMemcpy(k_dev, k, sz, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}
#endif
	printf("K & V initialized\n");

	// scratch pad [pt_flat, // int[PA_NUM_PAGES]
	// raw logits, logits, attn weights], each float[num_tokens]
	err = cudaMalloc(&scr_dev, (PA_NUM_PAGES * sizeof(int)) + (3 * num_tokens * sizeof(float)));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc(&out_dev, d * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}


	err = cudaMemcpy(q_dev, q, d * sizeof(__half), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	for (i = 0; i < PA_NUM_PAGES; i++)
		pt_flat[i] = pa_pool.ptable[i].physical_frame;
	err = cudaMemcpy(scr_dev, pt_flat, PA_NUM_PAGES * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}
	
	
	block = 256;
	grid = (num_tokens + block - 1) / block;

	attn_logits<<<grid, block>>>(kvpool_dev, q_dev, scr_dev, num_tokens, d);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
    		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
    		return 1;
	}

   	attn<<<1,1>>>(kvpool_dev, scr_dev, num_tokens, d, out_dev);
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

	cudaFree(kvpool_dev);
	cudaFree(q_dev);
	cudaFree(scr_dev);
	cudaFree(out_dev);

	free(q);
	free(k);
	return 0;
}
