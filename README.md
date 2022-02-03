# BilevelHeuristics.jl


This package implements different heuristics and metaheuristics algorithms for 
bilevel optimization.

## Algorithms

The algorithms implemented are listed as follows:

### BLEMO: Bilevel Evolutionary Multiobjective Optimization

Use the following command to perform the experiments regarding BLEMO:

Open a Julia console (REPL) in the project's main folder and run:

```julia
julia> include("scripts/test-blemo.jl")
```

**First execution can take some minutes since dependencies are installed.**

Important files:
- BLEMO source code is in `src/algorithms/BLEMO/BLEMO.jl`.
- Experiments are in `scripts/test-blemo.jl`.

More information about the project structure is detailed [here](https://juliadynamics.github.io/DrWatson.jl/stable/project/#Default-Project-Setup-1).

### BCA: Bilevel Centers Algorithms

```julia
using Metaheuristics
using BilevelHeuristics


# objective functions
F(x, y) = sum(x.^2) - sum( ( x - y ).^2 ) # upper level
f(x, y) = sum(y.^2) + sum( ( x - y ).^2 ) # lower level

# bounds
bounds_ul = bounds_ll = [-5ones(5) 5ones(5)]'
D_ul = size(bounds_ul, 2)
D_ll = size(bounds_ll, 2)

# parameters
K = 7
η_max = 2.0

# global options for the upper level
options_ul = Options(f_calls_limit = 500*D_ul, iterations=1000, f_tol = 1e-2, debug = true)

# global options for the lower level
options_ll = Options(f_calls_limit = 500*D_ul, debug = false  , f_tol = 1e-3)


# information on the optimizaton problem
information_ul = Information(f_optimum = 0.0)
information_ll = Information(f_optimum = 0.0)

# the algorithm
bca = BCA(;N = K*D_ul, n = K*D_ll, K, η_max, options_ul, options_ll, information_ul, information_ll)

# approximate solutions
r = optimize( F, f, bounds_ul, bounds_ll, bca)

```
