# PI-Calculation

Calculating PI Value In Different Parallel Framework.

## Basic Theory
Since<!-- $$ \int_0^1 \frac{1}{1+x^2}dx = arctanx\big|_0^1$$ -->
<img src='http://latex.codecogs.com/svg.latex?%5Cdpi%7B120%7D%20%5Clarge%20%5Cint_0%5E1%20%5Cfrac%7B1%7D%7B1&plus;x%5E2%7Ddx%20%3D%20arctanx%5Cbig%7C_0%5E1'></img>Thus we have<!-- $$\Pi = 4 \times \int_0^1\frac{1}{1+x^2}dx$$ -->
<img src='http://latex.codecogs.com/svg.latex?%5Cdpi%7B120%7D%20%5Clarge%20%5CPi%20%3D%204%20%5Ctimes%20%5Cint_0%5E1%5Cfrac%7B1%7D%7B1&plus;x%5E2%7Ddx'></img>


And we use mid-rectangle method to calculate the integration, which includes loops that may be optimized using parallel computing methods.

## Directories
#### PThread
Use `pthread` as the parallel framework.

#### OpenMP
Use `OpenMP` as the parallel framework to calculate, note that in macOS the default `clang` does not support `OpenMP`
, thus it needs to be built with `gcc` or `clang-omp`.

`gcc-6` could be directly installed by 
```
brew install gcc --without-multilib
```

and `clang-omp`  could be installed via
```
brew install clang-omp
```

#### MPI
Use `MPI` as the parallel framework.

#### CUDA
Use `CUDA` to optimize the parallel computing process, which must be running under CUDA environment i.e. you must have 
at least a nVidia card and `nvcc` installed to compile and run the code.

#### MPIOMP
Use mixed method of `MPI` and `OpenMP` to gain scalability and high parallel performance at the same time.

## Experiment
All experiments are carried out under `Linux` with `nvcc` and nVidia cards installed.

And I chose 2^30 as the STEP_NUM for all framework.

Parallel parameters are listed below:

| Framework     | Parameters              |
|:-------------:|:-----------------------:|
| OpenMP        | 16 Threads              |
| MPI           | 16 Processes            |
| PThread       | 16 Threads              |
| CUDA          | 512 Threads / 64 Blocks |
| MPIOMP        | 4 Processes / 4 Threads |


## License
[MIT](https://github.com/RyanWangGit/PI-Calculation/blob/master/LICENSE.md).
