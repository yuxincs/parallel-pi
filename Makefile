all: openmp pthread mpi cuda

openmp: openmp.c
	${CC} -std=c99 -fopenmp -o openmp openmp.c

pthread: pthread.c
	${CC} -std=c99 -pthread -o pthread pthread.c

mpi: mpi.c
	mpicc -std=c99 -o mpi mpi.c

cuda: cuda.cu
	nvcc -o cuda cuda.cu

mpiomp: mpiomp.c
	mpicc -fopenmp -std=c99 -o mpiomp mpiomp.c

.PHONY: clean test
clean:
	rm pthread openmp mpi cuda mpiomp

test: testopenmp testpthread testmpi testcuda

testopenmp: openmp
	openmp

testcuda: cuda
	cuda

testmpi: mpi
	mpiexec -np 16 mpi

testpthread: pthread
	pthread

testmpiomp: mpiomp
	mpiexec -np 4 mpiomp
