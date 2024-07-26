#include "./common.cpp"
#include <stdio.h>
#include <stdlib.h>

const unsigned SIZE = 1024*1024;

void kerenelVecAdd(unsigned i, float* c, const float* a, const float* b) {
    c[i] = a[i] + b[i];
}

__global__ void singleKernelVecAdd(float* c, const float* a, const float* b) {
    for (register unsigned i = 0; i < SIZE; ++i) {
        c[i] = a[i] + b[i];
    }
}

__global__ void kernelVecAdd(float*c, const float* a, const float* b, unsigned n) {
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    float* vecA = new float[SIZE];
    float* vecB = new float[SIZE];
    float* vecC = new float[SIZE];

    srand(0);
    setNormalizedRandomData(vecA, SIZE);
    setNormalizedRandomData(vecB, SIZE);

    float* dev_vecA = nullptr;
    float* dev_vecB = nullptr;
    float* dev_vecC = nullptr;

    cudaMalloc((void**)&dev_vecA, SIZE*sizeof(float));
    cudaMalloc((void**)&dev_vecB, SIZE*sizeof(float));
    cudaMalloc((void**)&dev_vecC, SIZE*sizeof(float));


    // printf("============HOST============");
    ELAPSED_TIME_BEGIN(0);
    for (register unsigned i = 0; i < SIZE; i++) {
        kerenelVecAdd(i, vecC, vecA, vecB);
    }
    ELAPSED_TIME_END(0);

    ELAPSED_TIME_BEGIN(3);
    cudaMemcpy(dev_vecA, vecA, SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vecB, vecB, SIZE*sizeof(float), cudaMemcpyHostToDevice);

    ELAPSED_TIME_BEGIN(1);
    singleKernelVecAdd<<<1,1>>>(dev_vecC, dev_vecA, dev_vecB);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(1);

    ELAPSED_TIME_BEGIN(2);
    kernelVecAdd<<<SIZE/1024, 1024>>>(dev_vecC, dev_vecA, dev_vecB, SIZE);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(2);

    cudaMemcpy(vecC, dev_vecC, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    ELAPSED_TIME_END(3);

    float sumA = getSum(vecA, SIZE);
    float sumB = getSum(vecB, SIZE);
    float sumC = getSum(vecC, SIZE);
    float diff = fabsf(sumC - (sumA + sumB));
    printf("SIZE = %d\n", SIZE);
    printf("sumA = %f\n", sumA);
    printf("sumB = %f\n", sumB);
    printf("sumC = %f\n", sumC);
    printf("diff(sumC, sumA+sumB) = %f\n", diff);
    printf("diff(sumC, sumA+sumB)/SIZE = %f\n", diff/SIZE);

    printf("vecA = [%8f %8f %8f %8f ... %8f %8f %8f %8f]\n", vecA[0], vecA[1], vecA[2], vecA[3], vecA[SIZE-4], vecA[SIZE-3], vecA[SIZE-2], vecA[SIZE-1]);
    printf("vecB = [%8f %8f %8f %8f ... %8f %8f %8f %8f]\n", vecB[0], vecB[1], vecB[2], vecB[3], vecB[SIZE-4], vecB[SIZE-3], vecB[SIZE-2], vecB[SIZE-1]);
    printf("vecC = [%8f %8f %8f %8f ... %8f %8f %8f %8f]\n", vecC[0], vecC[1], vecC[2], vecC[3], vecC[SIZE-4], vecC[SIZE-3], vecC[SIZE-2], vecC[SIZE-1]);

    delete[] vecA;
    delete[] vecB;
    delete[] vecC;

    return 0;
}