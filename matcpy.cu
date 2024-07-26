#include "./common.cpp"

unsigned matsize = 16000;

__global__ void kernelMatCpy(float* C, const float* A, int matsize, size_t pitch_in_elem) {
    register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gy < matsize) {
        register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
        if (gx < matsize) {
            register unsigned idx = gy * pitch_in_elem + gx;
            C[idx] = A[idx];
        }
    }
}

__global__ void kernelMatCpyShd(float* C, const float* A, int matsize, size_t pitch_in_elem) {
    __shared__ float s_mat[32][32];
    register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gy < matsize) {
        register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
        if (gx < matsize) {
            register unsigned idx = gy * pitch_in_elem + gx;
            s_mat[threadIdx.y][threadIdx.x] = A[idx];
            __syncthreads();
            C[idx] = s_mat[threadIdx.y][threadIdx.x];
        }
    }
}

__global__ void kernelMatTranspose(float* C, const float* A, unsigned matsize, size_t pitch_in_elem) {
    register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gy < matsize) {
        register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
        if (gx < matsize) {
            register unsigned idxA = gy * pitch_in_elem + gx;
            register unsigned idxC = gx * pitch_in_elem + gy;
            C[idxC] = A[idxA];
        }
    }
}

__global__ void kernelMatTransposeNVShd(float* C, const float* A, unsigned matsize, size_t pitch_in_elem) {
    __shared__ float mat[32][32];
    register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y;
    register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gy < matsize && gx < matsize) {
        register unsigned idxA = gy * pitch_in_elem + gx;
        mat[threadIdx.y][threadIdx.x] = A[idxA];
    }
    __syncthreads();
    if (gy < matsize && gx < matsize) {
        register unsigned idxC = gx * pitch_in_elem + gy;
        C[idxC] = mat[threadIdx.y][threadIdx.x];
    }
}

__global__ void kernelMatTransposeShd(float* C, const float* A, unsigned matsize, size_t pitch_in_elem) {
    __shared__ float mat[32][32];
    register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y;
    register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gy < matsize && gx < matsize) {
        register unsigned idxA = gy * pitch_in_elem + gx;
        mat[threadIdx.y][threadIdx.x] = A[idxA];
    }
    __syncthreads();
    gy = blockIdx.x * blockDim.x + threadIdx.y;
    gx = blockIdx.y * blockDim.y + threadIdx.x;
    if (gy < matsize && gx < matsize) {
        register unsigned idxC = gy * pitch_in_elem + gx;
        C[idxC] = mat[threadIdx.x][threadIdx.y];
    }
}

__global__ void kernelMatTransposeShdOptBK(float* C, const float* A, unsigned matsize, size_t pitch_in_elem) {
    __shared__ float mat[32][32+1];
    register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y;
    register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gy < matsize && gx < matsize) {
        register unsigned idxA = gy * pitch_in_elem + gx;
        mat[threadIdx.y][threadIdx.x] = A[idxA];
    }
    __syncthreads();
    gy = blockIdx.x * blockDim.x + threadIdx.y;
    gx = blockIdx.y * blockDim.y + threadIdx.x;
    if (gy < matsize && gx < matsize) {
        register unsigned idxC = gy * pitch_in_elem + gx;
        C[idxC] = mat[threadIdx.x][threadIdx.y];
    }
}

int main() {
    float* matA = new float[matsize * matsize];
    float* matC = new float[matsize * matsize];

    setNormalizedRandomData(matA, matsize * matsize);
    printf("==================MemCpy==================\n");
    printf("CPU Version Memory Copy\n");
    ELAPSED_TIME_BEGIN(0);
    for(register int y = 0; y < matsize; y++) {
        for(register int x = 0; x < matsize; x++) {
            register unsigned i = y * matsize + x;
            matC[i] = matA[i];
        }
    }
    ELAPSED_TIME_END(0);

    printf("Memcpy Version\n");
    ELAPSED_TIME_BEGIN(1);
    memcpy(matC, matA, matsize * matsize * sizeof(float));
    ELAPSED_TIME_END(1);

    printf("Memcpy CUDA Naive Version\n");
    float* dev_matA;
    float* dev_matC;
    size_t host_pitch = matsize * sizeof(float);
    size_t dev_pitch = 0;
    cudaMallocPitch((void**)&dev_matA, &dev_pitch, matsize * sizeof(float), matsize);
    cudaMallocPitch((void**)&dev_matC, &dev_pitch, matsize * sizeof(float), matsize);

    cudaMemcpy2D(dev_matA, dev_pitch, matA, host_pitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(div_up(matsize, dimBlock.x), div_up(matsize, dimBlock.y), 1);
    register unsigned pitch_in_elem = dev_pitch / sizeof(float);

    ELAPSED_TIME_BEGIN(2);
    kernelMatCpy<<<dimGrid,dimBlock>>>(dev_matC, dev_matA, matsize, pitch_in_elem);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(2);

    printf("Memcpy CUDA Shared Memory Version\n");
    ELAPSED_TIME_BEGIN(3);
    kernelMatCpyShd<<<dimGrid,dimBlock>>>(dev_matC, dev_matA, matsize, pitch_in_elem);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(3);

    printf("==================Transpose==================\n");
    printf("CPU Matrix Transpose\n");
    ELAPSED_TIME_BEGIN(4);
    for (register unsigned y = 0; y < matsize; y++) {
        for (register unsigned x = 0; x < matsize; x++) {
            unsigned indA = y * matsize + x;
            unsigned indC = x * matsize + y;
            matC[indC] = matA[indA];
        }
    }
    ELAPSED_TIME_END(4);

    printf("CUDA Global Memory Transpose\n");
    ELAPSED_TIME_BEGIN(5);
    kernelMatTranspose<<<dimGrid,dimBlock>>>(dev_matC, dev_matA, matsize, pitch_in_elem);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(5);

    printf("CUDA Shared Memory Naive Transpose\n");
    ELAPSED_TIME_BEGIN(6);
    kernelMatTransposeNVShd<<<dimGrid,dimBlock>>>(dev_matC, dev_matA, matsize, pitch_in_elem);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(6);

    printf("CUDA Global Memory Transpose\n");
    ELAPSED_TIME_BEGIN(7);
    kernelMatTransposeShd<<<dimGrid,dimBlock>>>(dev_matC, dev_matA, matsize, pitch_in_elem);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(7);

    printf("CUDA Global Memory Transpose Bank Optimization\n");
    ELAPSED_TIME_BEGIN(8);
    kernelMatTransposeShdOptBK<<<dimGrid,dimBlock>>>(dev_matC, dev_matA, matsize, pitch_in_elem);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(8);
}