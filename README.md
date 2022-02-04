# BilevelHeuristics.jl


This package implements different heuristic and metaheuristic algorithms for 
bilevel optimization.

**This package is under development (still an alpha version).**

## Installation


Open the Julia (Julia 1.2 or Later) REPL and press `]` to open the Pkg prompt. To add this package, use the add command:


Type `]`  
```julia
pkg> add https://github.com/jmejia8/BilevelHeuristics.jl.git
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("https://github.com/jmejia8/BilevelHeuristics.jl.git")
```

## Algorithms

The algorithms implemented are listed as follows:

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
