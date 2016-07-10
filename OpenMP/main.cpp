#include<stdio.h>
#include<time.h>
#include<omp.h>

static long stepNum = 100000000;
double stepLength, pi;

int main()
{
    clock_t t1 = clock();
    double sum = 0.0;
    stepLength = 1.0 / stepNum;

    printf("Start calculating...\n");
    double x = 0;
    // computational steps
#pragma omp parallel for reduction(+:sum)
    for(int i = 0;i < stepNum; i++)
    {
        x = (i + 0.5) * stepLength;
        sum += 1.0 / (1.0 + x * x);
    }
    pi = stepLength * sum * 4;
    clock_t t2 = clock();
    printf("PI = %.16lf\nTime elapsed : %lf seconds.\n", pi, (double(t2 - t1) / CLOCKS_PER_SEC));

    return 0;
}