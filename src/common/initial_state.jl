include("upper_level_decision_making.jl")


function gen_initial_state(status, problem::BLProblem,parameters,information,options)
    if typeof(parameters) <: AbstractNested
        N = parameters.ul.N
    else
        N = parameters.N
    end 

    a = problem.ul.search_space.lb'
    b = problem.ul.search_space.ub'
    D = length(a)

    X = a .+ (b - a) .* rand(N, D)

    population_ = []
    for i in 1:N
        x = X[i,:]
        ll_sols = lower_level_optimizer(status, parameters, problem, information, options, x)
        solutions = []
        for ll_sol in ll_sols
            push!(solutions, create_solution(x, ll_sol, problem))
        end
        decisions = upper_level_decision_making(status, parameters, problem, information, options, solutions)
        append!(population_, solutions[decisions])
    end

    population = [s for s in population_]

    best_solution = Metaheuristics.get_best(population)

    return BLState(best_solution, population)
end

