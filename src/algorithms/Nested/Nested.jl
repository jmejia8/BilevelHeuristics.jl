abstract type AbstractNested <: AbstractParameters end

"""
    Nested(; ul, ll)

A flexible **nested framework** for bilevel optimisation that accepts any
`Metaheuristics.jl` algorithm as the upper- and lower-level solver.

Unlike the specialised algorithms ([`BCA`](@ref), [`QBCA`](@ref), etc.) which use
tightly-coupled strategies, `Nested` decouples the two levels entirely:
1. The upper-level optimizer proposes candidate `x` values.
2. For each `x`, the lower-level optimizer runs to completion, finding optimal `y`.
3. The joint solution `(x, y)` is evaluated and the upper-level population is updated.

This makes `Nested` suitable for **prototyping** — you can experiment with different
combinations of algorithms (e.g. `GA` + `DE`, `PSO` + `BFGS`) without implementing any
bilevel-specific logic.

## Parameters
- `ul` — a configured `Metaheuristics.AbstractAlgorithm` for the upper level.
- `ll` — a configured `Metaheuristics.AbstractAlgorithm` for the lower level.

## Example

```julia
using BilevelHeuristics, Metaheuristics

ul = GA(N = 50)          # Genetic Algorithm at upper level
ll = DE(N = 50)          # Differential Evolution at lower level
res = optimize(F, f, bounds_ul, bounds_ll, Nested(; ul, ll))
```
"""
mutable struct Nested <: AbstractNested
    ul::AbstractParameters
    ll::AbstractParameters
end


include("lower_level.jl")
include("multi_objective.jl")
include("semi_vectorial.jl")
include("single_objective.jl")

function Nested(;ul::Metaheuristics.AbstractAlgorithm, ll::Metaheuristics.AbstractAlgorithm)
    parameters = Nested(ul.parameters, ll.parameters)
    # upper level configuration
    options_ul = ul.options
    information_ul = ul.information
    # lower level configuration
    options_ll = ll.options
    information_ll = ll.information

    Algorithm(
        parameters;
        options = BLOptions(options_ul, options_ll),
        information = BLInformation(information_ul, information_ll)
    )
end


function initialize!(
        status,
        parameters::AbstractNested,
        problem,
        information,
        options,
        args...;
        kargs...
    )

    if options.ul.f_calls_limit == 0
        D = Metaheuristics.getdim(problem.ul.search_space)
        options.ul.f_calls_limit = 500*D
        if options.ul.iterations == 0
            options.ul.iterations = options.ul.f_calls_limit ÷ D
        end
    end

    if options.ll.f_calls_limit == 0
        D = Metaheuristics.getdim(problem.ll.search_space)
        options.ll.f_calls_limit = 500*D
        if options.ll.iterations == 0
            options.ll.iterations = options.ll.f_calls_limit ÷ D
        end
    end

    status = gen_initial_state(status,problem,parameters,information,options)
    truncate_population!(status, parameters, problem, information, options)
    status
end


function update_state!(
        status,
        parameters::AbstractNested,
        problem,
        information,
        options,
        args...;
        kargs...
    )

    X = reproduction(status, parameters,problem,information,options,args...;kargs...)
    
    # for each x in X, perform, optimize the LL
    for i in 1:size(X,1)
        x = X[i,:]
        ll_sols = lower_level_optimizer(status,parameters,problem,information,options,x)
        solutions = []
        for ll_sol in ll_sols
            sol = create_solution(x, ll_sol, problem) # sol = (x,y, Fxy,fxy,...)
            push!(solutions, sol)
            # save best solution found so far
            if is_better(sol, status.best_sol, parameters)
                status.best_sol = sol
            end
        end
        preferences = upper_level_decision_making(status, parameters, problem, information, options, solutions)
        append!(status.population, solutions[preferences])
    end

    # delete solutions in status.population
    truncate_population!(status, parameters,problem,information,options,args...;kargs...)
end


function reproduction(status, parameters,problem,information,options,args...;kargs...)
    population_ul = get_ul_population(status.population)
    s = Metaheuristics.State(status.best_sol.ul, population_ul)
    if parameters.ul isa Metaheuristics.AbstractNSGA
        Metaheuristics.reproduction(s, parameters.ul, problem.ul, options.ul)
    else
        Metaheuristics.reproduction(s, parameters.ul, problem.ul)
    end
end


function stop_criteria!(status, parameters::AbstractNested, problem, information, options)
    return
end

function final_stage!(status, parameters::AbstractNested, problem, information, options)
    return
end

is_better(A, B, parameters::AbstractNested) = Metaheuristics.is_better(A, B)

