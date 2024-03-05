module TestProblems

export MMF, MTP, DS, TP

include("MMF/MMF.jl")
include("MTP/MTP.jl")
include("DS/DS.jl")
include("TP/TP.jl")
include("RealWorld/RealWorld.jl")

"""
    get_problem(problem)

Return `F, f, bounds_ul, bounds_ll, Ψ, o` where

- `F` is the upper-level objective function.
- `f` is the lower-level objective function.
- `bounds_ul` is the upper-level bounds.
- `bounds_ll` is the lower-level bounds. 
- `Ψ`-mapping that receives upper-level decision vector and return a Matrix containing lower-level optimal solutions.
- `o` contain a sampled optimal solutions.

# Example

```julia
using BilevelHeuristics
using DelimitedFiles

F, f, bounds_ul, bounds_ll, _ = TestProblems.get_problem("TP1")

algorithm = SMS_MOBO(;
        ul = SMS_EMOA(;N = 100), # upper level optimizer
        ll = NSGA2(;N = 50), # lower level optimizer
        ul_offsprings = 10,
        options_ul = Options(iterations = 100, f_calls_limit=Inf),
        options_ll = Options(iterations = 50, f_calls_limit=Inf)
    )

res = optimize(F, f, bounds_ul, bounds_ll, algorithm)

final_archive = algorithm.parameters.archive

final_archive[1][1] # upper level (solution #1)
final_archive[1][2] # lower level (solution #1)

ul_front = fvals([sol[1] for sol in final_archive])
writedlm("data.csv", ul_front, ',')
```
"""
function get_problem(problem::AbstractString)
    if problem[1:2] == "TP"
        return TP.get_problem(parse(Int, problem[3:end]))
    elseif problem[1:2] == "DS"
        return DS.get_problem(parse(Int, problem[3:end]))
    elseif problem == "GoldMining"
        return RealWorld.GoldMining()
    else
        error("Undefined problem $problem")
    end
    
end


get_problem(problem::Symbol) = get_problem(String(problem))


end
