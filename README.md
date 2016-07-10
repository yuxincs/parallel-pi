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
Use `OpenMP` as the parallel framework to calculate, note that in macOS it needs to be built with `gcc` available,
`gcc-6` is recommended, which could be installed by 
```
brew install gcc --without-multilib
```
Otherwise `clang-omp` is needed to support `OpenMP`. 
## License
[MIT](https://github.com/RyanWangGit/PI-Calculation/blob/master/LICENSE.md).
