#include <stdio.h>

#if defined(NDEBUG)
#define CUDA_CHECK_ERROR() 0
#else
#define CUDA_CHECK_ERROR() do { \
    cudaError_t e = cudaGetLastError(); \
    if (cudaSuccess != e) { \
        printf("cuda failure\"%s\" at %s:%d\n", \
            cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)
#endif

// CUDA kernel function
__global__ void add_kernel(float* b, const float* a) {
    int i = threadIdx.x;
    b[i] = a[i] + 1.0f;
}

int main() {
    const int SIZE = 8;
    const float a[SIZE] = {0., 1., 2., 3., 4., 5., 6., 7.};
    float b[SIZE] = {0., 0., 0., 0., 0., 0., 0., 0.};

    printf("a = {%f, %f, %f, %f, %f, %f, %f, %f}\n", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);

    float* dev_a = nullptr;
    float* dev_b = nullptr;

    cudaMalloc((void**)&dev_a, SIZE*sizeof(float));
    cudaMalloc((void**)&dev_b, SIZE*sizeof(float));
    cudaMemcpy(dev_a, a, SIZE*sizeof(float), cudaMemcpyHostToDevice);
    
    add_kernel<<<1,SIZE>>>(dev_b, dev_a);
    cudaDeviceSynchronize();

    cudaMemcpy(b, dev_b, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    printf("b = {%f, %f, %f, %f, %f, %f, %f, %f}\n", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
    CUDA_CHECK_ERROR();

    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}