#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <omp.h>

#define INTERVALS 1000000000

// Max number of threads per block
#define THREADS 512
#define BLOCKS 64

double calculatePiCPU();

// Synchronous error checking call. Enable with nvcc -DDEBUG
inline void checkCUDAError(const char *fileName, const int line)
{ 
	#ifdef DEBUG 
		cudaThreadSynchronize();
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess) 
		{
			printf("Error at %s: line %i: %s\n", fileName, line, cudaGetErrorString(error));
			exit(-1); 
		}
	#endif
}


__global__ void integrateSimple(float *sum, float step, int threads, int blocks)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (int i = idx; i < INTERVALS; i+=threads*blocks)
	{
		float x = (i+0.5f) * step;
		sum[idx] += 4.0f / (1.0f+ x*x);
	}
}

__global__ void integrateOptimised(int *n, float *g_sum)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;

	// Shared memory to hold the sum for each block
	__shared__ float s_sum[THREADS];

	float sum = 0.0f;
	float step  = 1.0f / (float)*n;

	for (int i = idx + 1; i <= *n; i += blockDim.x * BLOCKS) 
	{
		float x = step * ((float)i - 0.5f);
		sum += 4.0f / (1.0f+ x*x);
	}
	s_sum[tx] = sum * step;

	// Wait for all threads to catch up
	__syncthreads();

	// For each block, do sum using shared memory
	for (int i = blockDim.x / 2; i > 0; i >>= 1)
	{ 
		if (tx < i)
		{
			s_sum[tx] += s_sum[tx + i];
		}

		__syncthreads();
	}

	// Write results to global memory
	g_sum[idx] = s_sum[tx];
}

// Parallel reduction to speedup summation
__global__ static void sumReduce(int *n, float *g_sum)
{
	int tx = threadIdx.x;
    __shared__ float s_sum[THREADS];
    
    if (tx < BLOCKS)
      s_sum[tx] = g_sum[tx * THREADS];
    else
	{
      s_sum[tx] = 0.0f;
	}

	// For each block
    for (int i = blockDim.x / 2; i > 0; i >>= 1) 
	{ 
        if (tx < i)
		{
           s_sum[tx] += s_sum[tx + i];
		}
        __syncthreads();
    }

    g_sum[tx] = s_sum[tx];
}

int main()
{
	const float PI25DT = 3.141592653589793238462643;
	int deviceCount = 0;

	printf("Starting...");
    
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return 1;
    }

	deviceCount == 0 ? printf("There are no available CUDA device(s)\n") : printf("%d CUDA Capable device(s) detected\n", deviceCount);

	float cpuStart = omp_get_wtime();
	printf("\nCalculating Pi using CPU over %i intervals...\n", (int)INTERVALS);
	double piCpu = calculatePiCPU();
	printf("Pi is approximately %.16f, Error: %.16f\n", piCpu, fabs(piCpu - PI25DT));
	
	float cpuEnd = omp_get_wtime();
	float cpuTime = (cpuEnd - cpuStart) * 1000;

	/*--------- Simple Kernel ---------*/

	int threads = 8, blocks = 30;
	dim3 block(threads);
	dim3 grid(blocks);
	float *sum_h, *sum_d;
	float step = 1.0f / INTERVALS;
	float piSimple = 0;
	
	// Allocate host memory
	sum_h = (float *)malloc(blocks*threads*sizeof(float));	

	// Allocate device memory
	cudaMalloc((void **) &sum_d, blocks*threads*sizeof(float));

	// CUDA events needed to measure execution time
	cudaEvent_t start, stop;
	float gpuTime, optimizedGpuTime;

	// Start timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	printf("\nCalculating Pi using simple GPU kernel over %i intervals...\n", (int)INTERVALS);
	integrateSimple<<<grid, block>>>(sum_d, step, threads, blocks);	
	checkCUDAError(__FILE__, __LINE__);

	// Stop timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	// Retrieve result from device
	cudaMemcpy(sum_h, sum_d, blocks*threads*sizeof(float), cudaMemcpyDeviceToHost);
	
	// Sum result on host
	for (int i=0;i < threads*blocks; i++)
	{
		piSimple += sum_h[i];	
	}
	
	piSimple *= step;
	printf("Pi is approximately %.16f, Error: %.16f\n", piSimple, fabs(piSimple - PI25DT));

	free(sum_h);
	cudaFree(sum_d);

	/*--------- Optimized Kernel ---------*/

    int N = 0; 
    int *n_d; 
    float pi;
    float *pi_d;
 
    // Allocate memory on GPU
    cudaMalloc( (void **) &n_d, sizeof(int) * 1 );
    cudaMalloc( (void **) &pi_d, sizeof(float) * BLOCKS * THREADS );

	while (1)
	{
		printf("\nEnter no. of intervals for optimised kernel (Recommended 1000000): ");
		fflush(stdout);
		scanf("%d",&N);

		if (N > 0) break;
		else {
			printf("Please enter a valid number");
		}
	}
	// Copy N to GPU
	cudaMemcpy( n_d, &N, sizeof(int) * 1, cudaMemcpyHostToDevice );	

	// Start timer
	cudaEventRecord(start, 0);
	printf("Launching optimised kernel...\n");
	integrateOptimised<<<BLOCKS,THREADS>>>(n_d, pi_d);
	checkCUDAError(__FILE__, __LINE__);
	sumReduce<<< 1, THREADS >>>(n_d, pi_d);
	checkCUDAError(__FILE__, __LINE__);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&optimizedGpuTime, start, stop);
	
	// copy back from GPU to CPU
	cudaMemcpy(&pi, pi_d, sizeof(float) * 1, cudaMemcpyDeviceToHost);

	printf("Pi is approximately %.16f, Error: %.16f\n", pi, fabs(pi - PI25DT));

	// Print execution times
	printf("\n======================================\n\n");
	printf("CPU implementation time: %f ms\n", cpuTime);
	printf("Simple GPU implementation time: %f ms\n", gpuTime);
	printf("Optimised GPU implementation time: %f ms\n", optimizedGpuTime);

	// Free memory
	cudaFree(n_d);
	cudaFree(pi_d);

	// Reset Device
	cudaDeviceReset();
	return 0;
}

double calculatePiCPU()
{
	double pi, sum = 0.0;
	double step = 1.0 / INTERVALS;

	for (int i = 0; i < INTERVALS; i++)
	{
		double x = (i - 0.5) * step;
		sum += 4.0 / (1.0 + x*x);
	}
	
	pi = step * sum;
	return pi;
}