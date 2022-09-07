abstract type AbstractNested <: AbstractParameters end

"""
    Nested(ul, ll)

Nested is a framework that uses nested scheme to solve bilevel problem. You only need
to provide:

- `ul` an upper-level optimizer
- `ll` a lower-level optimizer

with their configuration.
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
        D = size(problem.ul.bounds,2)
        options.ul.f_calls_limit = 500*D
        if options.ul.iterations == 0
            options.ul.iterations = options.ul.f_calls_limit ÷ D
        end
    end

    if options.ll.f_calls_limit == 0
        D = size(problem.ll.bounds,2)
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
    Metaheuristics.reproduction(s, parameters.ul, problem.ul)
end


function stop_criteria!(status, parameters::AbstractNested, problem, information, options)
    return
end

function final_stage!(status, parameters::AbstractNested, problem, information, options)
    return
end

is_better(A, B, parameters::AbstractNested) = Metaheuristics.is_better(A, B)

