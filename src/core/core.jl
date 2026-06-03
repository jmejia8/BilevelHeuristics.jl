"""
    BLProblem(ul, ll)

A bilevel problem defined by two [`Metaheuristics.Problem`](@extref)s — one for the
upper (leader) level and one for the lower (follower) level.

- `ul` — upper-level problem (objective `F`, bounds, search space).
- `ll` — lower-level problem (objective `f`, bounds, search space).
"""
mutable struct BLProblem <: Metaheuristics.AbstractProblem
    ul::Metaheuristics.Problem
    ll::Metaheuristics.Problem
end

"""
    BLInformation(; ul = Information(), ll = Information())

Stores runtime information (best / worst function values, feasible ratios, etc.) for
both the upper and lower levels.  Used internally for termination checks and logging.

## Fields
- `ul` — [`Metaheuristics.Information`](@extref) for the upper level.
- `ll` — [`Metaheuristics.Information`](@extref) for the lower level.
"""
struct BLInformation
    ul::Metaheuristics.Information
    ll::Metaheuristics.Information
end

function Base.show(io::IO, blinfo::BLInformation)
    println(io, "Upper-level information:")
    Base.show(io, blinfo.ul)

    println(io, "\nLower-level information:")
    Base.show(io, blinfo.ll) 
end

"""
    BLOptions(; ul = Options(), ll = Options())

Configuration settings (population size, iteration budget, tolerances, verbosity, etc.)
for both the upper and lower levels.

## Fields
- `ul` — [`Metaheuristics.Options`](@extref) for the upper level (e.g.
  `f_calls_limit`, `iterations`, `f_tol`, `verbose`, `seed`, `store_convergence`).
- `ll` — [`Metaheuristics.Options`](@extref) for the lower level.

## Example
```julia
options = BLOptions(
    ul = Options(f_calls_limit = 10_000, verbose = true),
    ll = Options(f_calls_limit = 50_000),
)
```
"""
struct BLOptions
    ul::Metaheuristics.Options
    ll::Metaheuristics.Options
end


function Base.show(io::IO, bloptions::BLOptions)
    println(io, "Upper-level options:")
    Base.show(io, bloptions.ul)

    println(io, "\nLower-level options:")
    Base.show(io, bloptions.ll) 
end

include("BLIndividual.jl")
include("BLState.jl")
include("BLAlgorithm.jl")

