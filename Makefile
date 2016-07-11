OUTPUT = ./Output

all: OpenMP PThread MPI

OpenMP: openmp.c Output
	gcc-6 -std=c99 -fopenmp -o ${OUTPUT}/OpenMP openmp.c

PThread: pthread.c Output
	${CC} -std=c99 -lpthread -o ${OUTPUT}/PThread pthread.c

MPI: mpi.c
	mpicc -o ${OUTPUT}/MPI mpi.c

Output: ${OUTPUT}
	mkdir ${OUTPUT}

.PHONY: clean test
clean:
	rm -rf ${OUTPUT}

test: OpenMP PThread MPI
	${OUTPUT}/OpenMP
	${OUTPUT}/PThread
	mpiexec -np 4 ${OUTPUT}/MPI