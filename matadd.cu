#include "./common.cpp"
#include <stdio.h>
#include <stdlib.h>

unsigned nrow = 10000;
unsigned ncol = 10000;

__global__ void kernel_matadd(float* c, const float* a, const float* b, unsigned nrow, unsigned ncol) {
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < nrow && col < ncol) {
        unsigned i = row * ncol + col;
        c[i] = a[i] + b[i];
    }
}

int main(const int argc, const char* argv[]) {
    float* matA = nullptr;
    float* matB = nullptr;
    float* matC = nullptr;

    matA = new float[nrow*ncol];
    matB = new float[nrow*ncol];
    matC = new float[nrow*ncol];

    ELAPSED_TIME_BEGIN(0);
    for (register unsigned r = 0; r < nrow; ++r) {
        for (register unsigned c = 0; c < ncol; ++c) {
            unsigned i = r * ncol + c;
            matC[i] = matA[i] + matB[i];
        }
    }
    ELAPSED_TIME_END(0);

    float* dev_matA = nullptr;
    float* dev_matB = nullptr;
    float* dev_matC = nullptr;

    cudaMalloc((void**)&dev_matA, nrow*ncol*sizeof(float));
    cudaMalloc((void**)&dev_matB, nrow*ncol*sizeof(float));
    cudaMalloc((void**)&dev_matC, nrow*ncol*sizeof(float));

    cudaMemcpy(dev_matA, matA, nrow*ncol*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matB, matB, nrow*ncol*sizeof(float), cudaMemcpyHostToDevice);

    ELAPSED_TIME_BEGIN(1);
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid((ncol+dimBlock.x-1)/dimBlock.x, (nrow+dimBlock.y-1)/dimBlock.y, 1);

    kernel_matadd<<<dimGrid,dimBlock>>>(dev_matC, dev_matA, dev_matB, nrow, ncol);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(1);
    
    cudaMemcpy(matC, dev_matC, nrow*ncol*sizeof(float), cudaMemcpyDeviceToHost);

}