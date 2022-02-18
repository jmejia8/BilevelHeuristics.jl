"""
    Heuristic(ul, ll)

Heuristic is a framework that uses nested scheme to solve bilevel problem. You only need
to provide:

- `ul` an upper-level optimizer
- `ll` a lower-level optimizer

with their configuration.
"""
mutable struct Heuristic <: AbstractParameters
    ul::AbstractParameters
    ll::AbstractParameters
end

function Heuristic(;ul::Metaheuristics.AbstractAlgorithm, ll::Metaheuristics.AbstractAlgorithm)
    parameters = Heuristic(ul.parameters, ll.parameters)
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
        parameters::Heuristic,
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
        parameters::Heuristic,
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


function lower_level_optimizer(
        status,
        parameters::Heuristic,
        problem,
        information,
        options,
        x,
        initial_ll_sols = [],
        args...;
        kargs...
    )

    # lower level function parametrized by x
    f_x(y) = Metaheuristics.evaluate(x, y, problem.ll)
    
    # changing seed is important
    options.ll.seed = rand(UInt)
    method = Metaheuristics.Algorithm(parameters.ll;
                                      options=options.ll,
                                      information=information.ll
                                     )

    if !isempty(initial_ll_sols)
        Y = initial_ll_sols
        @assert parameters.ll.N == size(Y,1)
        population_ll = [Metaheuristics.create_child(y, f(Y[i,:])) for i in 1:size(Y,1)]
        method.status = Metaheuristics.State(population_ll[1], population_ll)
    end

    res = Metaheuristics.optimize(f_x, problem.ll.bounds, method)

    return lower_level_decision_making(status, parameters,problem,information,options,res,args...;kargs...)
end

# single-objective case
function lower_level_decision_making(
        status,
        parameters::Heuristic,
        problem,
        information,
        options,
        results_ll::State,
        args...;
        kargs...
    ) 

    return handle_ll_multimodality(results_ll, problem, options) 
end

# multi-objective case
function lower_level_decision_making(
        status,
        parameters::Heuristic,
        problem,
        information,
        options,
        results_ll::State{T},
        args...;
        kargs...
    ) where T <: Metaheuristics.AbstractMultiObjectiveSolution

    return results_ll.population
end


function truncate_population!(
        status,
        parameters::Heuristic,
        problem,
        information,
        options
    )

    mask = sortperm(status.population, lt = (a, b) -> is_better(a,b, parameters))
    N = parameters.ul.N
    status.population = status.population[mask[1:N]]

end

function truncate_population!(
        status::BLState{BLIndividual{U, L}},
        parameters::Heuristic,
        problem,
        information,
        options
    ) where U <: Metaheuristics.AbstractMultiObjectiveSolution where L <: AbstractSolution

    population_ul = get_ul_population(status.population)
    Metaheuristics.environmental_selection!(population_ul, parameters.ul)

    # TODO improve performance this part
    mask = Int[]
    for sol in get_ul_population(status.population)
        i = findfirst( s -> s==sol, population_ul)
        isnothing(i) && continue
        push!(mask, i)
    end

    deleteat!(status.population, mask)    
end

function reproduction(status, parameters,problem,information,options,args...;kargs...)
    population_ul = get_ul_population(status.population)
    s = Metaheuristics.State(status.best_sol.ul, population_ul)
    Metaheuristics.reproduction(status_ul, parameters.ul, problem.ul)
end


function stop_criteria!(status, parameters::Heuristic, problem, information, options)
    return
end

function final_stage!(status, parameters::Heuristic, problem, information, options)
    return
end

is_better(A, B, parameters::Heuristic) = Metaheuristics.is_better(A, B)
