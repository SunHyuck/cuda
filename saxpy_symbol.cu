#include "./common.cpp"
#include <stdio.h>
#include <stdlib.h>

const unsigned vecSize = 256*1024*1024;
const float host_a = 1.234f;

__constant__ float saxpy_a = 1.234f;

float host_x[vecSize];
float host_y[vecSize];
float host_z[vecSize];

__device__ float dev_vecX[vecSize];
__device__ float dev_vecY[vecSize];
__device__ float dev_vecZ[vecSize];

__global__ void kernelSaxpy(unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dev_vecZ[i] = saxpy_a * dev_vecX[i] + dev_vecY[i];
    }
}


int main(const int argc, const char* argv[]) {
    // switch(argc) {
    // case 1:
    //     break;
    // case 2:
    //     vecSize = procArg(argv[0], argv[1], 1);
    //     break;
    // case 3:
    //     vecSize = procArg(argv[0], argv[1], 1);
    //     saxpy_a = procArg<float>(argv[0], argv[2]);
    //     break;
    // default:
    //     printf("usage: %s[num] [a]\n", argv[0]);
    //     exit(EXIT_FAILURE);
    //     break;
    // }

    srand(0);
    setNormalizedRandomData(host_x, vecSize);
    setNormalizedRandomData(host_y, vecSize);

    ELAPSED_TIME_BEGIN(3);
    cudaMemcpyToSymbol(dev_vecX, host_x, sizeof(host_x));
    cudaMemcpyToSymbol(dev_vecY, host_y, sizeof(host_y));
    CUDA_CHECK_ERROR();

    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid((vecSize+dimBlock.x-1)/dimBlock.x, 1, 1);
    ELAPSED_TIME_BEGIN(1);
    kernelSaxpy<<<dimGrid,dimBlock>>>(vecSize);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(1);
    CUDA_CHECK_ERROR();

    cudaMemcpyFromSymbol(host_z, dev_vecZ, sizeof(host_z));
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(3);
}