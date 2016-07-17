#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

const double PI = 3.1415926535897932;
const long STEP_NUM = 10000000;

int main (int argc, char* argv[])
{
    long totalCount = 0; 

    MPI_Init (&argc, &argv);

    int rank, size;
    // get process ID
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    // get processes number
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    // synchronize all processes and get the begin time
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)
        printf("\n%d processes initialized.\nStart calculating...\n", size); 

    double startTime = MPI_Wtime();

    long count = 0;
    srand((int)time(NULL) ^ omp_get_thread_num());
    // each process will calculate a part of the sum
	#pragma omp parallel for reduction(+:count) num_threads(4)
    for (int i = 0; i < STEP_NUM / size; i ++)
    {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if ((x * x + y * y) <= 1)
            count += 1;
    }
    
    // sum up all results
    MPI_Reduce(&count, &totalCount, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // synchronize all processes and get the end time
    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    // caculate and print PI
    if (rank == 0)
    {
        double pi = ((double)totalCount / STEP_NUM) * 4;
        printf("PI = %.16lf with error %.16lf\nTime elapsed : %lf seconds.\n\n", pi, PI - pi, (endTime - startTime));
    }

    MPI_Finalize();

    return 0;
}