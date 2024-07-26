#include <stdio.h>
#include "./common.cpp"

void bigJob(void) {
    int count = 0;
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 10000; j++) {
            count++;
        }
    }
    printf("We got %d counts.\n", count);
};

int main(void) {
    ELAPSED_TIME_BEGIN(0);
    bigJob();
    ELAPSED_TIME_END(0);
    return 0;
}