#include "./common.cpp"
#include <stdio.h>
#include <stdlib.h>

const unsigned SIZE = 256*1024*1024;

void kerenelVecAdd(unsigned i, float* c, const float* a, const float* b) {
    c[i] = a[i] + b[i];
}

__global__ void singleKernelVecAdd(float* c, const float* a, const float* b) {
    for (register unsigned i = 0; i < SIZE; ++i) {
        c[i] = a[i] + b[i];
    }
}

__global__ void kernelVecAdd(float*c, const float* a, const float* b, unsigned n, long long* times) {
    clock_t start = clock();
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
    clock_t end = clock();
    if (i==0) {
        times[0] = (long long)(end-start);
    }
}

int main(const int argc, const char* argv[]) {
    // SIZE = procArg(argc, argv[]);
    float* vecA = new float[SIZE];
    float* vecB = new float[SIZE];
    float* vecC = new float[SIZE];

    srand(0);
    setNormalizedRandomData(vecA, SIZE);
    setNormalizedRandomData(vecB, SIZE);

    float* dev_vecA = nullptr;
    float* dev_vecB = nullptr;
    float* dev_vecC = nullptr;
    long long* host_times = new long long[1];
    long long* dev_times = nullptr;

    cudaMalloc((void**)&dev_vecA, SIZE*sizeof(float));
    cudaMalloc((void**)&dev_vecB, SIZE*sizeof(float));
    cudaMalloc((void**)&dev_vecC, SIZE*sizeof(float));
    cudaMalloc((void**)&dev_times, 1*sizeof(long long));
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(0);
    for (register unsigned i = 0; i < SIZE; i++) {
        kerenelVecAdd(i, vecC, vecA, vecB);
    }
    ELAPSED_TIME_END(0);
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(3);
    cudaMemcpy(dev_vecA, vecA, SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vecB, vecB, SIZE*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(1);
    singleKernelVecAdd<<<1,1>>>(dev_vecC, dev_vecA, dev_vecB);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(1);
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(2);
    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid((SIZE+dimBlock.x-1)/dimBlock.x,1,1);
    kernelVecAdd<<<dimGrid,dimBlock>>>(dev_vecC, dev_vecA, dev_vecB, SIZE, dev_times);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(2);
    CUDA_CHECK_ERROR();
    cudaMemcpy(vecC, dev_vecC, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_times, dev_times, 1*sizeof(long long), cudaMemcpyDeviceToHost);
    ELAPSED_TIME_END(3);
    CUDA_CHECK_ERROR();
    
    cudaFree(dev_vecA);
    cudaFree(dev_vecB);
    cudaFree(dev_vecC);

    int peak_clk = 1;
    cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
    printf("num clock = %lld, peak clock rate = %dkHz, elapsed time: %f usec\n", host_times[0], peak_clk, host_times[0]*1000.0f/(float)peak_clk);

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
    delete[] host_times;

    return 0;
}