# BilevelHeuristics - Heuristics and Metaheuristics for Bilevel Optimization

**Author: Jesus Mejía (@jmejia8)**

 Approximate algorithms for bilevel optimization.

[![Source](https://img.shields.io/badge/GitHub-source-green.svg)](https://github.com/jmejia8/BilevelHeuristics.jl)
[![Build Status](https://travis-ci.com/jmejia8/BilevelHeuristics.jl.svg?branch=main)](https://app.travis-ci.com/jmejia8/BilevelHeuristics.jl)

## Introduction

Bilevel Optimization is a very challenging task that require high-performance algorithms
to optimize hierarchical problem. This package implements a variety of approximate 
algorithms for bilevel optimization.

BilevelHeuristics extends the [Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl)
API to implement bilevel optimization algorithms.

## Installation

Open the Julia (Julia 1.6 or later) REPL and press `]` to open the Pkg prompt. To add this package, use the add command:

```
pkg> add BilevelHeuristics
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("BilevelHeuristics")
```

## Example


```julia-repl
julia> F(x, y) = sum(x.^2) + sum(y.^2)
F (generic function with 1 method)

julia> f(x, y) = sum((x - y).^2) + y[1]^2
f (generic function with 1 method)

julia> bounds_ul = bounds_ll = [-ones(5)'; ones(5)']
2×5 Matrix{Float64}:
 -1.0  -1.0  -1.0  -1.0  -1.0
  1.0   1.0   1.0   1.0   1.0

julia> res = optimize(F, f, bounds_ul, bounds_ll)
+=========== RESULT ==========+
  iteration: 108
    minimum: 
          F: 7.68483e-08
          f: 3.96871e-09
  minimizer: 
          x: [1.0283390421119262e-5, -0.00017833559080058394, -1.612275010196171e-5, 0.00012064585960330227, 4.38964383738248e-5]
          y: [1.154609166391327e-5, -0.0001300400306798623, 1.1811981430188257e-6, 8.868498295184257e-5, 5.732849695863675e-5]
    F calls: 2503
    f calls: 5044647
    Message: Stopped due UL function evaluations limitations. 
 total time: 21.4550 s
+============================+
```
