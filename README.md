<!-- Enable MathJax engine to use Tex -->
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# PI-Calculation
###### By : Ryan Wang @ HUST
###### Email : wangyuxin@hust.edu.cn

Calculating PI Value In Different Parallel Framework.

## Basic Theory
We all know that 

$$ \int_0^1 \frac{1}{1+x^2}dx = arctanx\big|_0^1$$ 

Thus we have

$$\Pi = 4 \times \int_0^1\frac{1}{1+x^2}dx$$

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

## Experiment
All experiments are carried out under `Linux` with `nvcc` and nVidia cards installed.

And I chose 2^30 as the STEP_NUM for all framework except `OpenMP` which does not provide manual settings.

Parallel parameters are listed below:
|               |                        |
|:-------------:|:----------------------:|
| MPI           | 64 Processes           |
| PThread       | 64 Threads             |
| CUDA          | 512 Threads/64 Blocks  |


## License
[MIT](https://github.com/RyanWangGit/PI-Calculation/blob/master/LICENSE.md).
