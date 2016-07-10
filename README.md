# PI-Calculation
###### By : Ryan Wang @ HUST
###### Email : wangyuxin@hust.edu.cn

Calculating PI Value In Different Parallel Framework.

## Directories
#### PThread
Use `pthread` as the parallel framework.

#### OpenMP
Use `OpenMP` as the parallel framework to calculate, note that in macOS it needs to be built with `gcc` available,
`gcc-6` is recommended, which could be installed by 
```
brew install gcc --without-multilib
```
Otherwise `clang-omp` is needed to support `OpenMP`. 
## License
[MIT](https://github.com/RyanWangGit/PI-Calculation/blob/master/LICENSE.md).
