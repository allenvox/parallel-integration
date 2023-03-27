#define _POSIX_C_SOURCE 1
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

const double PI = 3.14159265358979323846;
const int n = 10000000; // 10 mil

// variant 1: y = x / (y ^ 2); x = [0, 1]; y = [2, 5];
double func(double x, double y) {
    return x / (y * y);
}

double getrand(unsigned int *seed) {
    return (double)rand_r(seed) / RAND_MAX;
}

int main () {
    printf("Numerical integration using Monte-Carlo method\nn = %d\n", n);
    int in = 0;
    double s = 0.0;
    double t = omp_get_wtime();
    #pragma omp parallel num_threads(1) // 1 - serial, 2... - multithreaded
    {
        double s_loc = 0;
        int in_loc = 0;
        unsigned int seed = omp_get_thread_num();
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            double x = getrand(&seed);         // x in [0, 1]
            double y = getrand(&seed) * 7 - 2; // y in [2, 5]
            if (y <= func(x, y)) {
                in_loc++;
                s_loc += func(x, y);
            }
        }
        #pragma omp atomic
        s += s_loc;
        #pragma omp atomic
        in += in_loc;
    }
    double v = 3; // b - a
    double res = v * s / in;
    t = omp_get_wtime() - t;
    printf("Result: %.12f\nElapsed time: %.12f\n", res, t);
    return 0;
}
