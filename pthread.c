#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

typedef struct _thread_param
{
    long start, end;
    double sum;
}ThreadParameter;

const double PI = 3.1415926535897932;
const long STEP_NUM = 1000000000;
const double STEP_LENGTH = 1.0 / 1000000000;
const int THREAD_NUM = 20;

void * calc(void * p)
{
    ThreadParameter * param = (ThreadParameter *)p;

    double x;
    double sum = 0.0;
    for(long i = param->start; i < param->end; i++)
    {
        x = (i + 0.5) * STEP_LENGTH;
        sum += 1.0 / (1.0 + x * x);
    }
    param->sum = sum;

    return NULL;
}

int main()
{
    struct timeval startTime;
    gettimeofday(&startTime, NULL);

    double pi = 0;

    pthread_t * threadHandles = (pthread_t *) malloc(sizeof(pthread_t) * THREAD_NUM);
    ThreadParameter * paramArray[THREAD_NUM];

    printf("\nStart calculating with %d threads...\n", THREAD_NUM);
    for (int i = 0; i < THREAD_NUM; i++)
    {
        ThreadParameter * param = (ThreadParameter *)malloc(sizeof(ThreadParameter));
        param->start = i * STEP_NUM / THREAD_NUM;
        param->end = param->start + STEP_NUM / THREAD_NUM;
        paramArray[i] = param;
        pthread_create(&threadHandles[i], NULL, calc, (void *)param);
    }

    for (int i = 0; i < THREAD_NUM; i++)
        pthread_join(threadHandles[i], NULL);

    for(int i = 0; i < THREAD_NUM; i++)
    {
        pi += paramArray[i]->sum;
        free(paramArray[i]);
    }

    pi = STEP_LENGTH * pi * 4;

    free(threadHandles);

    struct timeval endTime;
    gettimeofday(&endTime, NULL);

    printf("PI = %.16lf with error %.16lf\nTime elapsed : %lf seconds.\n\n", pi, PI - pi, (endTime.tv_sec - startTime.tv_sec) + ((double)(endTime.tv_usec - startTime.tv_usec) / 10E6 ));

    return 0;
}