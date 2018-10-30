all: openmp pthread mpi cuda

openmp: openmp.c
	${CC} -std=c99 -fopenmp -o pi-openmp openmp.c

pthread: pthread.c
	${CC} -std=c99 -pthread -o pi-pthread pthread.c

mpi: mpi.c
	mpicc -std=c99 -o pi-mpi mpi.c

cuda: cuda.cu
	nvcc -o pi-cuda cuda.cu

mpiomp: mpiomp.c
	mpicc -fopenmp -std=c99 -o pi-mpiomp mpiomp.c

.PHONY: clean test
clean:
	rm pi-*

test: testopenmp testpthread testmpi testcuda

testopenmp: openmp
	export OMP_NUM_THREADS=16
	pi-openmp

testcuda: cuda
	pi-cuda

testmpi: mpi
	mpiexec -np 16 pi-mpi

testpthread: pthread
	pi-pthread

testmpiomp: mpiomp
	mpiexec -np 4 pi-mpiomp
