/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

// Includes
#include <stdio.h>

// includes, project
// #include "../include/sdkHelper.h"  // helper for shared functions common to CUDA SDK samples
//#include <shrQATest.h>
//#include <shrUtils.h>

// includes CUDA
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define FIRST_PHASE_TIME 0xFFF
#define SECOND_PHASE_TIME 0xFFF

// Variables
float* h_A;
float* h_B;
float* h_C;
float* d_A;
float* d_B;
float* d_C;
bool noprompt = false;

// Functions
void CleanupResources(void);
void RandomInit(float*, int);
void ParseArguments(int, char**);

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

// end of CUDA Helper Functions


// Device code
__global__ void VecAdd_seperate(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
		int thread_id = threadIdx.x;
    int k = thread_id;
		__shared__ long counter[THREADS_PER_BLOCK];
		__shared__ double result[THREADS_PER_BLOCK];

		counter[k] = 0;
		result[k] = k;

		__syncthreads();

		while (counter[0] < FIRST_PHASE_TIME) {
			counter[k] ++;
			result[k] *= result[k];
		}
		__syncthreads();

		counter[k] = 0;

		__syncthreads();

		if (thread_id < 32) {
			while (counter[0] < SECOND_PHASE_TIME) {
				counter[k] ++;
				result[k] *= result[k];
			}
		}

		__syncthreads();

		//for(k=0;k<100;k++){
			//if (i < N)
       // C[i] = A[i] + B[i];
		//}
}

//modified
__global__ void VecAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
		int thread_id = threadIdx.x;
    int k = thread_id;
		__shared__ long counter[THREADS_PER_BLOCK];
		__shared__ double result[THREADS_PER_BLOCK];

		counter[k] = 0;
		result[k] = k;

		__syncthreads();

		while (counter[0] < FIRST_PHASE_TIME) {
            long tc = counter[k] + 1;
            double tr = result[k] * result[k];
            // float tr = result[k];
            // tr *= tr;
			counter[k] = tc;
			result[k] = tr;
		}
		__syncthreads();

		counter[k] = 0;

		__syncthreads();

		if (thread_id < 32) {
			while (counter[0] < SECOND_PHASE_TIME) {
				long tc = counter[k] + 1;
                double tr = result[k] * result[k];
                counter[k] = tc;
                result[k] = tr;
			}
		}

		__syncthreads();

		//for(k=0;k<100;k++){
			//if (i < N)
       // C[i] = A[i] + B[i];
		//}
}

#ifndef TEST_ROUND
#define TEST_ROUND 9
#endif

// Host code
int main(int argc, char** argv)
{

    printf("Vector Addition\n");
    int N = 8192;
    size_t size = N * sizeof(float);
    //ParseArguments(argc, argv);

    // Allocate input vectors h_A and h_B in host memory

    cudaEvent_t begin, end;
    float elapsed_time, min_time = -1;
    
    for(int r = 0; r < TEST_ROUND; r++){
        h_A = (float*)malloc(size);
        if (h_A == 0) CleanupResources();
        h_B = (float*)malloc(size);
        if (h_B == 0) CleanupResources();
        h_C = (float*)malloc(size);
        if (h_C == 0) CleanupResources();

        RandomInit(h_A, N);
        RandomInit(h_B, N);

        cudaEventCreate(&begin);
        cudaEventCreate(&end);

        // Allocate vectors in device memory
        checkCudaErrors( cudaMalloc((void**)&d_A, size) );
        checkCudaErrors( cudaMalloc((void**)&d_B, size) );
        checkCudaErrors( cudaMalloc((void**)&d_C, size) );

        // Copy vectors from host memory to device memory
        checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );

        // Invoke kernel
        int threadsPerBlock = THREADS_PER_BLOCK;
        //int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGrid = 30;

        cudaEventRecord(begin);

#ifdef SEPERATE
        VecAdd_seperate<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
#else
        VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
#endif
        cudaEventRecord(end);
        cudaEventSynchronize(begin);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, begin, end);

#ifdef SEPERATE
        printf("Seperate version. ");
#endif

        if(min_time < 0 || elapsed_time < min_time){
            min_time = elapsed_time;
        }

        printf("elapsed time: %f ms, min time: %f ms\n", elapsed_time, min_time);

        getLastCudaError("kernel launch failure");


        // Copy result from device memory to host memory
        // h_C contains the result in host memory
        checkCudaErrors( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );
        
        // Verify result
        int i;
        /*for (i = 0; i < N; ++i) {
            float sum = h_A[i] + h_B[i];
            if (fabs(h_C[i] - sum) > 1e-5)
                break;
        }*/

        CleanupResources();
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }

    // Initialize input vectors
}

void CleanupResources(void)
{
    // Free device memory
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
        
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--noprompt") == 0 ||
            strcmp(argv[i], "-noprompt") == 0) 
        {
            noprompt = true;
            break;
        }
    }
}
