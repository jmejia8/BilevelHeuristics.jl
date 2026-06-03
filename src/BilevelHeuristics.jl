"""
    BilevelHeuristics

Heuristics and metaheuristics for solving **bilevel optimization** problems — a class of
hierarchical optimization where one problem (the *upper level* or *leader*) is constrained
by the optimal solution of another (the *lower level* or *follower*).

## Mathematical formulation

```math
\\begin{aligned}
\\min_{x \\in X,\\, y \\in Y}\\; & F(x, y)                    && \\text{(upper level / leader)}  \\\\
\\text{s.t.}\\;        & G_i(x, y) \\leq 0,\\; i = 1,\\dots,I \\\\
                      & H_j(x, y) = 0,\\;  j = 1,\\dots,J \\\\
                      & y \\in \\arg\\min_{z \\in Y}\\; \\{ f(x, z) : g_k(x, z) \\leq 0,\\; h_l(x, z) = 0 \\}
                        && \\text{(lower level / follower)}
\\end{aligned}
```

Here:
- `x` — upper-level decision vector,
- `y` — lower-level decision vector,
- `F`, `f` — upper- and lower-level objective functions,
- `G_i`, `H_j` — upper-level inequality and equality constraints,
- `g_k`, `h_l` — lower-level inequality and equality constraints.

This formulation models a **Stackelberg game** (von Stackelberg, 1934): the leader moves
first by choosing `x`, the follower reacts optimally by solving its own problem
parameterised by `x`, and the leader evaluates `F` at the resulting pair `(x, y)`.

## Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| [`BCA`](@ref) | Single-objective | Bilevel Centers Algorithm (physics-inspired, center-of-mass) |
| [`QBCA`](@ref) | Single-objective | Quasi-Newton BCA (BFGS-based lower-level solver) |
| [`QBCA2`](@ref) | Single-objective | Improved QBCA with pseudo-feasible solution handling |
| [`SABO`](@ref) | Single-objective | Surrogate-Assisted Bilevel Optimization |
| [`Nested`](@ref) | Single/Multi-objective | Flexible nested framework (any UL/LL optimizer) |
| [`DBMA`](@ref) | Multi-objective UL | Differential evolution with ε-constrained method |
| [`BLEMO`](@ref) | Multi-objective | Bilevel Evolutionary Multi-objective Optimization |
| [`SMS_MOBO`](@ref) | Multi-objective | S-metric-selection Multi-objective Bilevel Optimization |

All algorithms follow the common `Metaheuristics.jl` interface and are called via
[`optimize`](@ref).

## Quick example

```julia
using BilevelHeuristics

F(x, y) = sum(x.^2) + sum(y.^2)           # upper-level objective
f(x, y) = sum((x - y).^2) + y[1]^2        # lower-level objective
bounds = [-ones(5)'; ones(5)']             # bounds for both levels

res = optimize(F, f, bounds, bounds, BCA())
x, y = minimizer(res)
Fmin, fmin = minimum(res)
```

## References

- von Stackelberg, H. (1934). *Marktform und Gleichgewicht*. Springer.
- Dempe, S. (2002). *Foundations of Bilevel Programming*. Springer.
- Mejía-de-Dios, J. A., & Mezura-Montes, E. (2018–2022). See individual algorithm
  citations.
"""
module BilevelHeuristics

export BCA, QBCA, QBCA2, SABO, optimize, minimum, minimizer
export TestProblems, BLEMO, SMS_MOBO

export get_ll_population, get_ul_population,ulvector,llvector,ulfval,llfval,ulpositions
export ulpositions,llpositions,is_pseudo_feasible
export ulfvals, ulgvals, ulhvals, llfvals, llgvals, llhvals
export BLState, BLOptions, BLInformation
export Nested
export DBMA


# import dependencies
include("external.jl")

# core
include("core/core.jl")

include("common/common.jl")
include("optimize.jl")
include("TestProblems/TestProblems.jl")
include("BiApprox/BiApprox.jl")


# algorithms
include("algorithms/BCA/BCA.jl")
include("algorithms/QBCA/QBCA.jl")
include("algorithms/QBCA2/QBCA2.jl")
include("algorithms/SABO/SABO.jl")
include("algorithms/BLEMO/BLEMO.jl")
include("algorithms/SMS_MOBO/SMS_MOBO.jl")

# framework
include("algorithms/Nested/Nested.jl")
include("algorithms/DBMA/DBMA.jl")

# deprecations
@deprecate Heuristic Nested
@deprecate Heuristic(;kargs...) Nested(;kargs...)


end # module
