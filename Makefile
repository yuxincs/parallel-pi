OUTPUT = ./Output

all: OpenMP PThread MPI

OpenMP: openmp.c Output
	${CC} -fopenmp -o ${OUTPUT}/OpenMP openmp.c

PThread: pthread.c Output
	${CC} -lpthread -o ${OUTPUT}/PThread pthread.c

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
	mpirun -n 4 ${OUTPUT}/MPI