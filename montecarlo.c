#define _POSIX_C_SOURCE 1
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

const double PI = 3.14159265358979323846;
const int n = 10000000; // 10 mil

// variant 1
double func(double x) {
    return pow(x, 0.3333333);
}

double getrand(unsigned int *seed) {
    return (double)rand_r(seed) / RAND_MAX;
}

int main () {
    const int n = 10000000;
    printf("Numerical integration using Monte-Carlo method\nn = %d\n", n);

    int in = 0;
    double s = 0.0;
    #pragma omp parallel num_threads(4)
    {
        double s_loc = 0;
        int in_loc = 0;
        unsigned int seed = omp_get_thread_num();
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            double x = getrand(&seed);     // x in (0, 1)
            double y = getrand(&seed) * 7.0001 - 2.0002; // y in (2, 5)
            if (y <= func(x)) {
                in_loc++;
                s_loc += func(x);
            }
        }
        #pragma omp atomic
        s += s_loc;
        #pragma omp atomic
        in += in_loc;
    }
    double v = 5 - 2 - 0.0002; // b - a
    double res = v * s / in;
    printf("Result: %.12f\n", res);
    return 0;
}