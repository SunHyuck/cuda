#include "./common.cpp"

unsigned matsize = 4096;
const unsigned TILE_WIDTH = 32;
const unsigned MAX_TILE_WIDTH = 32;

__global__ void kernelMatMul(float* C, const float* A, const float* B, unsigned matsize, size_t pitch_in_elem) {
    register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y;
    register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gy < matsize && gx < matsize) {
        float sum = 0.0f;
        for (register unsigned k = 0; k < matsize; k++) {
            register unsigned idxA = gy * pitch_in_elem + k;
            register unsigned idxB = k * pitch_in_elem + gx;
            sum += A[idxA] * B[idxB];
        }
        register unsigned idxC = gy * pitch_in_elem + gx;
        C[idxC] = sum;
    }
}

__global__ void kernelMatMulAT(float* C, const float* A, const float* B, unsigned matsize, size_t pitch_in_elem) {
    __shared__ float s_A[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    __shared__ float s_B[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    register unsigned ntiles = matsize / TILE_WIDTH;
    register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y;
    register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
    register float sum = 0.0f;
    for (register unsigned tile = 0; tile < ntiles; tile++) {
        register unsigned idxA = gy * pitch_in_elem + (tile * TILE_WIDTH + threadIdx.x);
        s_A[threadIdx.y][threadIdx.x] = A[idxA];
        register unsigned idxB = (tile * TILE_WIDTH + threadIdx.y) * pitch_in_elem + gx;
        s_B[threadIdx.y][threadIdx.x] = B[idxB];
        __syncthreads();
        for (register unsigned k = 0; k < TILE_WIDTH; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    register unsigned idxC = gy * pitch_in_elem + gx;
    C[idxC] = sum;
}

__global__ void kernelMatMulTile(float* C, const float* A, const float* B, unsigned matsize, size_t pitch_in_elem) {
    __shared__ float s_A[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    __shared__ float s_B[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    register unsigned ntiles = (matsize + TILE_WIDTH -1) / TILE_WIDTH;
    register unsigned remaining = matsize;
    register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y;
    register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
    register float sum = 0.0f;
    for (register unsigned tile = 0; tile < ntiles; tile++) {
        register unsigned nelem = min(remaining, TILE_WIDTH);
        remaining -= TILE_WIDTH;
        if (gy < matsize && threadIdx.x < nelem) {
            register unsigned idxA = gy * pitch_in_elem + (tile * TILE_WIDTH + threadIdx.x);
            s_A[threadIdx.y][threadIdx.x] = A[idxA];
        }
        if (gx < matsize && threadIdx.y < nelem) {
            register unsigned idxB = (tile * TILE_WIDTH + threadIdx.y) * pitch_in_elem + gx;
            s_B[threadIdx.y][threadIdx.x] = B[idxB];
        }
        __syncthreads();
        for (register unsigned k = 0; k < TILE_WIDTH; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (gy < matsize && gx < matsize) {
        register unsigned idxC = gy * pitch_in_elem + gx;
        C[idxC] = sum;
    }
}

int main() {
    float* matA = nullptr;
    float* matB = nullptr;
    float* matC = nullptr;

    matA = new float[matsize * matsize];
    matB = new float[matsize * matsize];
    matC = new float[matsize * matsize];

    setNormalizedRandomData(matA, matsize * matsize);
    setNormalizedRandomData(matB, matsize * matsize);

    printf("CPU Naive Triple for-loop\n");
    ELAPSED_TIME_BEGIN(0);
    for (register unsigned y = 0; y < matsize; y++) {
        for (register unsigned x = 0; x < matsize; x++) {
            unsigned indC = y * matsize + x;
            register float ans = 0.0f;
            for (register unsigned k = 0; k < matsize; k ++) {
                unsigned indA = y * matsize + k;
                unsigned indB = k * matsize + x;
                ans += matA[indA] * matB[indB];
            }
            matC[indC] = ans;
        }
    }
    ELAPSED_TIME_END(0);

    printf("CPU outer-K Version\n");
    ELAPSED_TIME_BEGIN(1);
    for (register unsigned k = 0; k < matsize; k++) {
        for (register unsigned y = 0; y < matsize; y++) {
            for (register unsigned x = 0; x < matsize; x++) {
                unsigned indC = y * matsize + x;
                unsigned indA = y * matsize + k;
                unsigned indB = k * matsize + x;
                matC[indC] += matA[indA] * matB[indB];
            }
        }
    }
    ELAPSED_TIME_END(1);

    float* dev_matA = nullptr;
    float* dev_matB = nullptr;
    float* dev_matC = nullptr;

    size_t host_pitch = matsize * sizeof(float);
    size_t dev_pitch = 0;
    cudaMallocPitch((void**)&dev_matA, &dev_pitch, matsize * sizeof(float), matsize);
    cudaMallocPitch((void**)&dev_matB, &dev_pitch, matsize * sizeof(float), matsize);
    cudaMallocPitch((void**)&dev_matC, &dev_pitch, matsize * sizeof(float), matsize);

    cudaMemcpy2D(dev_matA, dev_pitch, matA, host_pitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dev_matA, dev_pitch, matB, host_pitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);

    printf("CUDA Version\n");
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(div_up(matsize, dimBlock.x), div_up(matsize, dimBlock.y), 1);
    register unsigned pitch_in_elem = dev_pitch / sizeof(float);
    ELAPSED_TIME_BEGIN(2);
    kernelMatMul<<<dimGrid,dimBlock>>>(dev_matC, dev_matA, dev_matB, matsize, pitch_in_elem);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(2);

    printf("CUDA Aligned Tile Version\n");
    ELAPSED_TIME_BEGIN(3);
    kernelMatMulAT<<<dimGrid, dimBlock>>>(dev_matC, dev_matA, dev_matB, matsize, pitch_in_elem);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(3);

    printf("CUDA Tile Version\n");
    ELAPSED_TIME_BEGIN(4);
    kernelMatMulTile<<<dimGrid, dimBlock>>>(dev_matC, dev_matA, dev_matB, matsize, pitch_in_elem);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(4);
}