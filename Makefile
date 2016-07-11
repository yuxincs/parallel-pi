OUTPUT = ./Output

all: OpenMP PThread MPI

OpenMP: openmp.c Output
	${CC} -std=c99 -fopenmp -o ${OUTPUT}/OpenMP openmp.c

PThread: pthread.c Output
	${CC} -std=c99 -lpthread -o ${OUTPUT}/PThread pthread.c

MPI: mpi.c
	mpicc -std=c99 -o ${OUTPUT}/MPI mpi.c

CUDA: cuda.cu
	nvcc -o ${OUTPUT}/CUDA cuda.cu

Output: ${OUTPUT}
	mkdir ${OUTPUT}

.PHONY: clean test
clean:
	rm -rf ${OUTPUT}

test: OpenMP PThread MPI
	${OUTPUT}/OpenMP
	${OUTPUT}/PThread
	mpiexec -np 20 ${OUTPUT}/MPI
	${OUTPUT}/CUDA
