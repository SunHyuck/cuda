#include "./common.cpp"
#include <stdio.h>
#include <stdlib.h>

unsigned nrow = 10000;
unsigned ncol = 10000;

__global__ void kernel_matadd(float* c, const float* a, const float* b, unsigned nrow, unsigned ncol, size_t dev_pitch) {
    register unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ncol) {
        register unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
        if (row < nrow) {
            register unsigned offset = row * dev_pitch + col * sizeof(float);
            *((float*)((char*)c + offset)) = *((const float*)((const char*)a + offset)) + *((const float*)((const char*)b + offset));
        }
    }
}


int main(const int argc, const char* argv[]) {
    float* matA = nullptr;
    float* matB = nullptr;
    float* matC = nullptr;

    matA = new float[nrow*ncol];
    matB = new float[nrow*ncol];
    matC = new float[nrow*ncol];

    srand(0);
    setNormalizedRandomData(matA, nrow*ncol);
    setNormalizedRandomData(matB, nrow*ncol);

    float* dev_matA = nullptr;
    float* dev_matB = nullptr;
    float* dev_matC = nullptr;

    size_t host_pitch = ncol * sizeof(float);
    size_t dev_pitch = 0;

    cudaMallocPitch((void**)&dev_matA, &dev_pitch, ncol * sizeof(float), nrow);
    cudaMallocPitch((void**)&dev_matB, &dev_pitch, ncol * sizeof(float), nrow);
    cudaMallocPitch((void**)&dev_matC, &dev_pitch, ncol * sizeof(float), nrow);

    cudaMemcpy2D(dev_matA, dev_pitch, matA, host_pitch, ncol*sizeof(float), nrow, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dev_matB, dev_pitch, matB, host_pitch, ncol*sizeof(float), nrow, cudaMemcpyHostToDevice);

    ELAPSED_TIME_BEGIN(1);
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid((ncol+dimBlock.x-1)/dimBlock.x, (nrow+dimBlock.y-1)/dimBlock.y, 1);

    kernel_matadd<<<dimGrid,dimBlock>>>(dev_matC, dev_matA, dev_matB, nrow, ncol, dev_pitch);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(1);
    
    cudaMemcpy(matC, dev_matC, nrow*ncol*sizeof(float), cudaMemcpyDeviceToHost);

}