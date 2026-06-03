# BilevelHeuristics - Heuristics and Metaheuristics for Bilevel Optimization

**Author: Jesus Mejía (@jmejia8)**

Approximate algorithms for bilevel optimization.

[![Source](https://img.shields.io/badge/GitHub-source-green.svg)](https://github.com/jmejia8/BilevelHeuristics.jl)
[![Build Status](https://img.shields.io/travis/com/jmejia8/BilevelHeuristics.jl/main)](https://app.travis-ci.com/jmejia8/BilevelHeuristics.jl)

## Introduction

Bilevel Optimization is a very challenging task that requires high-performance algorithms
to solve hierarchical problems involving two coupled optimization tasks (for two decision-makers leader and a
follower). This package implements a variety of approximate algorithms for single-
objective, multi-objective, and semi-vectorial bilevel optimization.

BilevelHeuristics extends the [Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl)
API to implement bilevel optimization algorithms.

Available algorithms include dedicated single-objective methods ([`BCA`](@ref),
[`QBCA`](@ref), [`QBCA2`](@ref), [`SABO`](@ref)), multi-objective methods
([`BLEMO`](@ref), [`SMS_MOBO`](@ref)), and flexible frameworks that accept any
Metaheuristics.jl solver ([`Nested`](@ref), [`DBMA`](@ref)).

## Installation

Open the Julia (Julia 1.6 or later) REPL and press `]` to open the Pkg prompt. To add this package, use the add command:

```
pkg> add BilevelHeuristics
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("BilevelHeuristics")
```

## Quick start

### Single-objective

```julia-repl
julia> F(x, y) = sum(x.^2) + sum(y.^2)
F (generic function with 1 method)

julia> f(x, y) = sum((x - y).^2) + y[1]^2
f (generic function with 1 method)

julia> bounds = [-ones(5)'; ones(5)'];

julia> res = optimize(F, f, bounds, bounds, BCA())
+=========== RESULT ==========+
  iteration: 108
    minimum: 
          F: 4.08164e-10
          f: 3.49457e-10
  minimizer: 
          x: [1.821e-6, 9.431e-6, 5.039e-6, 1.244e-5, -9.431e-6]
          y: [5.174e-6, -1.650e-6, -1.591e-6, 2.166e-6, -3.175e-6]
    F calls: 2503
    f calls: 5123090
    Message: Stopped due UL function evaluations limitations. 
 total time: 32.3749 s
+============================+
```

### Multi-objective

```julia-repl
julia> F(x, y) = [y[1] - x[1], y[2]], [-1.0 - sum(y)], [0.0];

julia> f(x, y) = y, [-x[1]^2 + sum(y .^ 2)], [0.0];

julia> bounds_ul = [0.0 1.0]';

julia> bounds_ll = [-1 -1; 1 1.0];

julia> res = optimize(F, f, bounds_ul, bounds_ll, SMS_MOBO());
```
