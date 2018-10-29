OUTPUT = ./output

all: openmp pthread mpi cuda

openmp: openmp.c output
	${CC} -std=c99 -fopenmp -o ${OUTPUT}/openmp openmp.c

pthread: pthread.c output
	${CC} -std=c99 -lpthread -o ${OUTPUT}/pthread pthread.c

mpi: mpi.c output
	mpicc -std=c99 -o ${OUTPUT}/mpi mpi.c

cuda: cuda.cu output
	nvcc -o ${OUTPUT}/cuda cuda.cu

mpiomp: mpiomp.c output
	mpicc -fopenmp -std=c99 -o ${OUTPUT}/MPIOMP mpiomp.c

output: ${OUTPUT}
	mkdir ${OUTPUT}

.PHONY: clean test
clean:
	rm -rf ${OUTPUT}

test: testopenmp testpthread testmpi testcuda

testopenmp: openmp
	${OUTPUT}/openmp

testcuda: cuda
	${OUTPUT}/cuda

testmpi: mpi
	mpiexec -np 16 ${OUTPUT}/mpi

testpthread: pthread
	${OUTPUT}/pthread

testmpiomp: mpiomp
	mpiexec -np 4 ${OUTPUT}/mpiomp
