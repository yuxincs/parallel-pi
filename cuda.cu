#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

const float PI = 3.1415926535897932;
const long STEP_NUM = 1000000000;
const float STEP_LENGTH = 1.0 / 1000000000;
const int THREAD_NUM = 500;
const int BLOCK_NUM = 50;

__global__ void integrateSimple(float *sum, int stepNum, float stepLength, int threadNum, int blockNum)
{
	int globalThreadId = threadIdx.x + blockIdx.x *  blockDim.x;
	int start = (stepNum / (blockNum * threadNum)) * globalThreadId;
	int end = (stepNum / (blockNum * threadNum)) * (globalThreadId + 1);
	float x;
	for (int i = start; i < end; i ++)
	{
		x = (i + 0.5f) * stepLength;
		sum[globalThreadId] += 1.0f / (1.0f + x * x);
	}
}

__global__ void integrateOptimised(float *globalSum, int stepNum, float stepLength, int threadNum, int blockNum)
{
	int globalThreadId = threadIdx.x + blockIdx.x * blockDim.x;
	int start = (stepNum / (blockNum * threadNum)) * globalThreadId;
	int end = (stepNum / (blockNum * threadNum)) * (globalThreadId + 1);
	int localThreadId = threadIdx.x;
	int blockId = blockIdx.x;

	// shared memory to hold the sum for each block
	__shared__ float blockSum[THREAD_NUM];

	memset(blockSum, 0, threadNum * sizeof(float));

	float x;
	for (int i = start; i < end; i ++) 
	{
		x = (i + 0.5f) * stepLength;
		blockSum[localThreadId] += 1.0f / (1.0f+ x * x);
	}
	blockSum[localThreadId] *= stepLength * 4;

	// wait for all threads to catch up
	__syncthreads();

	// for each block, do sum using shared memory
	/*for (int i = blockDim.x / 2; i > 0; i >>= 1)
	{ 
		if (tx < i)
			s_sum[tx] += s_sum[tx + i];

		__syncthreads();
	}*/

	// sum up the summation of the block
	if(localThreadId == 0)
	{
		float sum = 0.0;
		for(int i = 0;i < threadNum; i++)
			sum += blockSum[i];

		// write results to global memory
		globalSum[blockId] = sum;
	}
}

// TODO: Check with this function to provide parallel reduction
// parallel reduction to speedup summation
__global__ static void sumReduce(long *n, float *globalSum)
{
	int tx = threadIdx.x;
    __shared__ float s_sum[THREAD_NUM];
    
    if (tx < BLOCK_NUM)
      s_sum[tx] = globalSum[tx * THREAD_NUM];
    else
      s_sum[tx] = 0.0f;

	// for each block
    for (int i = blockDim.x / 2; i > 0; i >>= 1) 
	{ 
        if (tx < i)
           s_sum[tx] += s_sum[tx + i];
        __syncthreads();
    }

    globalSum[tx] = s_sum[tx];
}

int main()
{
	int deviceCount = 0;

	printf("\nConfiguring device...\n");
    
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return 1;
    }

	if(deviceCount == 0)
	{
		printf("There are no available CUDA device(s)\n");
		return 1;
	}
	else
		printf("%d CUDA Capable device(s) detected\n", deviceCount);

	/* Simple Calculation (without optimization) */
	dim3 block(THREAD_NUM);
	dim3 grid(BLOCK_NUM);
	float *hostSum = NULL, *deviceSum = NULL;
	float piSimple = 0;
	
	// allocate host memory
	hostSum = (float *)malloc(BLOCK_NUM * THREAD_NUM * sizeof(float));	

	// allocate device memory
	cudaMalloc((void **) &deviceSum, BLOCK_NUM * THREAD_NUM * sizeof(float));

	// clear device memory
	cudaMemset(deviceSum, 0, BLOCK_NUM * THREAD_NUM * sizeof(float));

	// CUDA events needed to measure execution time
	cudaEvent_t startTime, stopTime;
	float simpleGpuTime, optimizedGpuTime;

	// start timer
	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);
	cudaEventRecord(startTime, 0);

	printf("Start calculating in simple kernel function...\n");
	integrateSimple<<<grid, block>>>(deviceSum, STEP_NUM, STEP_LENGTH, THREAD_NUM, BLOCK_NUM);	

	// stop timer
	cudaEventRecord(stopTime, 0);
	cudaEventSynchronize(stopTime);
	cudaEventElapsedTime(&simpleGpuTime, startTime, stopTime);

	// retrieve result from device
	cudaMemcpy(hostSum, deviceSum, BLOCK_NUM * THREAD_NUM * sizeof(float), cudaMemcpyDeviceToHost);
	
	// sum result on host
	for (int i = 0;i < BLOCK_NUM * THREAD_NUM; i++)
		piSimple += hostSum[i];	
	
	piSimple *= STEP_LENGTH * 4;
	printf("PI = %.16lf with error %.16lf\nTime elapsed : %f seconds.\n\n", piSimple, fabs(piSimple - PI), simpleGpuTime / 1000);

	free(hostSum);
	cudaFree(deviceSum);

	/* Optimized Calculation */
    float pi = 0.0;
    float *deviceBlockSum;
	float *hostBlockSum;
 
	// allocate memory on host
	hostBlockSum = (float *)malloc(BLOCK_NUM * sizeof(float));

    // allocate memory on GPU
    cudaMalloc( (void **) &deviceBlockSum, sizeof(float) * BLOCK_NUM);

	// Start timer
	cudaEventRecord(startTime, 0);
	printf("Start calculating in optimized kernel function...\n");
	integrateOptimised<<<BLOCK_NUM, THREAD_NUM>>>(deviceBlockSum, STEP_NUM, STEP_LENGTH, THREAD_NUM, BLOCK_NUM);

	// retrieve result from device
	cudaMemcpy(hostBlockSum, deviceBlockSum, BLOCK_NUM * sizeof(float), cudaMemcpyDeviceToHost);
	
	// sum result on host
	for (int i = 0;i < BLOCK_NUM; i++)
		pi += hostBlockSum[i];	

	cudaEventRecord(stopTime, 0);
	cudaEventSynchronize(stopTime);
	cudaEventElapsedTime(&optimizedGpuTime, startTime, stopTime);

	printf("PI = %.16lf with error %.16lf\nTime elapsed : %f seconds.\n\n", pi, fabs(pi - PI), optimizedGpuTime / 1000);

	// free memory
	cudaFree(deviceBlockSum);

	// reset Device
	cudaDeviceReset();
	return 0;
}