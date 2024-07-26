#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace std::chrono;

chrono::system_clock::time_point __time_begin[10] = {chrono::system_clock::now(), };

template <typename TYPE>
__host__ __device__ inline TYPE div_up(TYPE lhs, TYPE rhs) {
    return (lhs + rhs -1 ) / rhs;
}

#define CUDA_CHECK_ERROR() do { \
    cudaError_t e = cudaGetLastError(); \
    if (cudaSuccess != e) { \
        printf("cuda failure \"%s\" at %s:%d\n", \
                cudaGetErrorString(e), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
}while(0)

#define ELAPSED_TIME_BEGIN(N) do {\
    __time_begin[(N)] = chrono::system_clock::now();\
    printf("elapsed wall-clock time[%d] started\n", (N));\
    fflush(stdout);\
} while(0)

#define ELAPSED_TIME_END(N) do {\
    chrono::system_clock::time_point time_end = chrono::system_clock::now();\
    chrono::microseconds elapsed_msec = chrono::duration_cast<chrono::microseconds>(time_end - __time_begin[(N)]);\
    printf("elapsed wall-clock time[%d] = %ld usec\n", (N), (long)elapsed_msec.count());\
    fflush(stdout);\
} while(0)

template <typename TYPE>
void setNormalizedRandomData(TYPE* pDst, long long num, TYPE bound=static_cast<TYPE>(1000)) {
    int32_t bnd = static_cast<int32_t>(bound);
    while(num--) {
        *pDst++ = (rand()%bnd)/static_cast<TYPE>(bnd);
    }
}

template <typename TYPE>
TYPE getSum(const TYPE* pSrc, int num) {
    register TYPE sum = static_cast<TYPE>(0);

    const int chunk = 128*1024;
    while (num > chunk) {
        register TYPE partial = static_cast<TYPE>(0);
        register int n = chunk;
        while(n--) {
            partial += *pSrc++;
        }
        sum += partial;
        num -= chunk;
    }

    register TYPE partial = static_cast<TYPE>(0);
    while (num--) {
        partial += *pSrc++;
    }
    sum += partial;
    return sum;
}

template <typename TYPE>
void printVec(const char* name, const TYPE* vec, int num) {
    std::streamsize ss = std::cout.precision();
    std::cout.precision(5);
    std::cout<<name<<"=[";
    std::cout<<fixed<<showpoint<<std::setw(8)<<vec[0]<<" ";
    std::cout<<fixed<<showpoint<<std::setw(8)<<vec[1]<<" ";
    std::cout<<fixed<<showpoint<<std::setw(8)<<vec[2]<<" ";
    std::cout<<fixed<<showpoint<<std::setw(8)<<vec[3]<<" ";
    std::cout<<fixed<<showpoint<<std::setw(8)<<vec[num-4]<<" ";
    std::cout<<fixed<<showpoint<<std::setw(8)<<vec[num-3]<<" ";
    std::cout<<fixed<<showpoint<<std::setw(8)<<vec[num-2]<<" ";
    std::cout<<fixed<<showpoint<<std::setw(8)<<vec[num-1]<<"]"<<std::endl;
    std::cout.precision(ss);
}

template <typename TYPE>
TYPE procArg(const char* progname, const char* str, TYPE lbound = -1, TYPE ubound = -1) {
    char* pEnd = nullptr;
    TYPE value = 0;
    if (typeid(TYPE) == typeid(float) && typeid(TYPE) == typeid(double)) {
        value = strtof(str, &pEnd);
    } else {
        value = strtol(str, &pEnd, 10);
    }
    if (typeid(TYPE) != typeid(float) && typeid(TYPE) != typeid(double)) {
        if (pEnd != nullptr && *pEnd != '\0') {
            switch(*pEnd) {
                case 'k':
                case 'K':
                    value *= 1024;
                    break;
                case 'm':
                case 'M':
                    value *= (1024*1024);
                    break;
                case 'g':
                case 'G':
                    value *= (1024*1024*1024);
                    break;
                default:
                    printf("%s: ERROR: illegal parameter '%s'\n", progname, str);
                    exit(EXIT_FAILURE);
                    break;
            }
        }
    }
    if (lbound != -1 && value < lbound) {
        printf("%s: ERROR: invalid value: '%s'\n", progname, str);
        exit(EXIT_FAILURE);
    }
    if (ubound != -1 && value > ubound) {
        printf("%s: ERROR: invalid value: '%s\n", progname, str);
        exit(EXIT_FAILURE);
    }
    return value;
}