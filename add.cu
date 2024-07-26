#include <stdio.h>

__global__ void gpu_add_kernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void cpu_add_kernel(int idx, int* c, const int* a, const int* b) {
    int i = idx;
    c[i] = a[i] + b[i];
}

int main(void) {
    const int SIZE = 5;
    const int a[SIZE] = {1, 2, 3, 4, 5};
    const int b[SIZE] = {10, 20, 30, 40, 50};
    int c[SIZE] = {0};
    int d[SIZE] = {0};

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_d = 0;

    cudaMalloc((void**)&dev_a, SIZE*sizeof(float));
    cudaMalloc((void**)&dev_b, SIZE*sizeof(float));
    cudaMalloc((void**)&dev_d, SIZE*sizeof(float));

    cudaMemcpy(dev_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice);

    gpu_add_kernel<<<1,SIZE>>>(dev_d, dev_a, dev_b);
    cudaDeviceSynchronize();

    cudaError_t err = cudaPeekAtLastError();
    if (cudaSuccess != err) {
        printf("CUDA: ERROR: cuda failure \"%s\"\n", cudaGetErrorString(err));
    }
    else {
        printf("CUDA: success\n");
    }

    cudaMemcpy(d, dev_d, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    for (register int i = 0; i < SIZE; i++) {
       cpu_add_kernel(i, c, a, b);
    }

    printf("{%d, %d, %d, %d, %d} + {%d, %d, %d, %d, %d} = {%d, %d, %d, %d, %d}\n",
        a[0], a[1], a[2], a[3], a[4], b[0], b[1], b[2], b[3], b[4], c[0], c[1], c[2], c[3], c[4]);

    printf("{%d, %d, %d, %d, %d} + {%d, %d, %d, %d, %d} = {%d, %d, %d, %d, %d}\n",
        a[0], a[1], a[2], a[3], a[4], b[0], b[1], b[2], b[3], b[4], d[0], d[1], d[2], d[3], d[4]);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_d);

    return 0;
}