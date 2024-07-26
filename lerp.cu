#include "./common.cpp"
#include <stdio.h>
#include <stdlib.h>

const unsigned vecSize = 256*1024*1024;
float lerp_t = 0.234F;

__global__ void kernel_lerp(float* z, const float t, const float* x, const float* y, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = (1.0F-t)*x[i] + t*y[i];
    }
}

__global__ void kernel_lerp_fma(float* z, const float t, const float* x, const float* y, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = fmaf(t, y[i], fmaf(-t, x[i], x[i]));
    }
}

int main() {
    float* vecX = new float[vecSize];
    float* vecY = new float[vecSize];
    float* vecZ = new float[vecSize];

    float* dev_vecX = nullptr;
    float* dev_vecY = nullptr;
    float* dev_vecZ = nullptr;

    cudaMalloc((void**)&dev_vecX, vecSize*sizeof(float));
    cudaMalloc((void**)&dev_vecY, vecSize*sizeof(float));
    cudaMalloc((void**)&dev_vecZ, vecSize*sizeof(float));
    CUDA_CHECK_ERROR();

    srand(0);
    setNormalizedRandomData(vecX, vecSize);
    setNormalizedRandomData(vecY, vecSize);

    ELAPSED_TIME_BEGIN(3);

    cudaMemcpy(dev_vecX, vecX, vecSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vecY, vecY, vecSize*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid((vecSize+dimBlock.x-1)/dimBlock.x, 1, 1);
    ELAPSED_TIME_BEGIN(1);
    kernel_lerp<<<dimGrid,dimBlock>>>(dev_vecZ, lerp_t, dev_vecX, dev_vecY, vecSize);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(1);
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(2);
    kernel_lerp_fma<<<dimGrid,dimBlock>>>(dev_vecZ, lerp_t, dev_vecX, dev_vecY, vecSize);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(2);
    CUDA_CHECK_ERROR();

    cudaMemcpy(vecZ, dev_vecZ, vecSize*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(3);

    cudaFree(dev_vecX);
    cudaFree(dev_vecY);
    cudaFree(dev_vecZ);

    float sumX = getSum(vecX, vecSize);
    float sumY = getSum(vecY, vecSize);
    float sumZ = getSum(vecZ, vecSize);
    float diff = fabsf(sumZ - ((1-lerp_t)*sumX+lerp_t*sumY));

    printf("vecSize = %d\n", vecSize);
    printf("t    = %f\n", lerp_t);
    printf("sumX = %f\n", sumX);
    printf("sumY = %f\n", sumY);
    printf("sumZ = %f\n", sumZ);
    printf("diff(sumZ, (1-t)*sumX+t*sumY) = %f\n", diff);
    printf("diff(sumZ, (1-t)*sumX+t*sumY)/vecSize = %f\n", diff/vecSize);

    printf("vecX = [%8f %8f %8f %8f ... %8f %8f %8f %8f]\n", vecX[0], vecX[1], vecX[2], vecX[3], vecX[vecSize-4], vecX[vecSize-3], vecX[vecSize-2], vecX[vecSize-1]);
    printf("vecY = [%8f %8f %8f %8f ... %8f %8f %8f %8f]\n", vecY[0], vecY[1], vecY[2], vecY[3], vecY[vecSize-4], vecY[vecSize-3], vecY[vecSize-2], vecY[vecSize-1]);
    printf("vecZ = [%8f %8f %8f %8f ... %8f %8f %8f %8f]\n", vecZ[0], vecZ[1], vecZ[2], vecZ[3], vecZ[vecSize-4], vecZ[vecSize-3], vecZ[vecSize-2], vecZ[vecSize-1]);


    delete[] vecX;
    delete[] vecY;
    delete[] vecZ;
}