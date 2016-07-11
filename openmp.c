#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

const double PI = 3.1415926535897932;
const long STEP_NUM = 1000000000;
const double STEP_LENGTH = 1.0 / 1000000000;

int main()
{
    struct timeval startTime;
    gettimeofday(&startTime, NULL);

    double sum = 0.0;
    double pi;

    printf("Start calculating...\n");
    // computational steps
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0;i < STEP_NUM; i++)
    {
        double x = (i + 0.5) * STEP_LENGTH;
        sum += 1.0 / (1.0 + x * x);
    }
    pi = STEP_LENGTH * sum * 4;

    struct timeval endTime;
    gettimeofday(&endTime, NULL);
    printf("PI = %.16lf with error %.16lf\nTime elapsed : %lf seconds.\n", pi, PI - pi, (endTime.tv_sec - startTime.tv_sec) + ((double)(endTime.tv_usec - startTime.tv_usec) / 10E6 ));

    return 0;
}