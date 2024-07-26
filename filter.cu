#include "./common.cpp"
#include <stdio.h>
#include <stdlib.h>

unsigned nrow = 10000;
unsigned ncol = 10000;
dim3 dimImage(300, 300, 256);

__global__ void kernel_filter(float* c, const float* a, const float* b, unsigned ndim_z, unsigned ndim_y, unsigned ndim_x) {
    unsigned idx_z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_x < ndim_x && idx_y < ndim_y && idx_z < ndim_z) {
        unsigned i = (idx_z * ndim_y + idx_y) * ndim_x + idx_x;
        c[i] = a[i] * b[i];
    }
}

__global__ void kernel_filter_pitched(void* matC, const void* matA, const void* matB, size_t pitch, unsigned ndim_z, unsigned ndim_y, unsigned ndim_x) {
    register unsigned idx_z = blockIdx.z * blockDim.z + threadIdx.z;
    register unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    register unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_x < ndim_x && idx_y < ndim_y && idx_z < ndim_z) {
        register unsigned offset_in_byte = (idx_z * ndim_y + idx_y) * pitch + idx_x * sizeof(float);
        *((float*)((char*)matC + offset_in_byte)) = *((const float*)((const char*)matA + offset_in_byte)) * *((const float*)((const char*)matB + offset_in_byte));
    }
}

int main() {
    float* matA = nullptr;
    float* matB = nullptr;
    float* matC = nullptr;

    matA = new float[dimImage.z * dimImage.y * dimImage.x];
    matB = new float[dimImage.z * dimImage.y * dimImage.x];
    matC = new float[dimImage.z * dimImage.y * dimImage.x];

    ELAPSED_TIME_BEGIN(0);
    for (register unsigned z = 0; z < dimImage.z; ++z) {
        for (register unsigned y = 0; y < dimImage.y; ++y) {
            for (register unsigned x = 0; x < dimImage.x; ++x) {
                int i = (z * dimImage.y + y) * dimImage.x + x;
                matC[i] = matA[i] * matB[i];
            }
        }
    }
    ELAPSED_TIME_END(0);

    float* dev_matA = nullptr;
    float* dev_matB = nullptr;
    float* dev_matC = nullptr;

    cudaMalloc((void**)&dev_matA, dimImage.z * dimImage.y * dimImage.x * sizeof(float));
    cudaMalloc((void**)&dev_matB, dimImage.z * dimImage.y * dimImage.x * sizeof(float));
    cudaMalloc((void**)&dev_matC, dimImage.z * dimImage.y * dimImage.x * sizeof(float));
    
    cudaMemcpy(dev_matA, matA, dimImage.z * dimImage.y * dimImage.x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matB, matB, dimImage.z * dimImage.y * dimImage.x * sizeof(float), cudaMemcpyHostToDevice);

    ELAPSED_TIME_BEGIN(1);
    dim3 dimBlock(8, 8, 8);
    dim3 dimGrid(div_up(dimImage.x, dimBlock.x), div_up(dimImage.y, dimBlock.y), div_up(dimImage.z, dimBlock.z));
    printf("dimGrid = %d * %d * %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("dimBlock = %d * %d * %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    kernel_filter<<<dimGrid, dimBlock>>>(dev_matC, dev_matA, dev_matB, dimImage.z, dimImage.y, dimImage.x);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(1);
    cudaMemcpy(matC, dev_matC, dimImage.z * dimImage.y * dimImage.x * sizeof(float), cudaMemcpyDeviceToHost);
}