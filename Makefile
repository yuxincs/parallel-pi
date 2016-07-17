OUTPUT = ./Output

all: OpenMP PThread MPI CUDA

OpenMP: openmp.c Output
	${CC} -std=c99 -fopenmp -o ${OUTPUT}/OpenMP openmp.c

PThread: pthread.c Output
	${CC} -std=c99 -lpthread -o ${OUTPUT}/PThread pthread.c

MPI: mpi.c
	mpicc -std=c99 -o ${OUTPUT}/MPI mpi.c

CUDA: cuda.cu
	nvcc -o ${OUTPUT}/CUDA cuda.cu

MPIOMP: mpiomp.c
	mpicc -fopenmp -std=c99 -o ${OUTPUT}/MPIOMP mpiomp.c

Output: ${OUTPUT}
	mkdir ${OUTPUT}

.PHONY: clean test
clean:
	rm -rf ${OUTPUT}

test: testOpenMP testPThread testMPI testCUDA
	
testOpenMP: OpenMP
	${OUTPUT}/OpenMP

testCUDA: CUDA
	${OUTPUT}/CUDA

testMPI: MPI
	mpiexec -np 16 ${OUTPUT}/MPI

testPThread: PThread
	${OUTPUT}/PThread

testMPIOMP: MPIOMP
	mpiexec -np 4 ${OUTPUT}/MPIOMP
