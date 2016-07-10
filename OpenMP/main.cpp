#include<stdio.h>
#include<time.h>
#include<omp.h>

const long STEP_NUM = 100000000;
const double STEP_LENGTH = 1.0 / STEP_NUM;

int main()
{
    clock_t startTime = clock();
    double sum = 0.0;
    double pi;

    printf("Start calculating...\n");
    double x = 0;
    // computational steps
#pragma omp parallel for reduction(+:sum)
    for(int i = 0;i < STEP_NUM; i++)
    {
        x = (i + 0.5) * STEP_LENGTH;
        sum += 1.0 / (1.0 + x * x);
    }
    pi = STEP_LENGTH * sum * 4;
    clock_t endTime = clock();
    printf("PI = %.16lf\nTime elapsed : %lf seconds.\n", pi, (double(endTime - startTime) / CLOCKS_PER_SEC));

    return 0;
}