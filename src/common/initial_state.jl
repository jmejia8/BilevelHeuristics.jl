function gen_initial_state(status, problem::BLProblem,parameters,information,options)
    N = parameters.N

    a = view(problem.ul.bounds, 1, :)'
    b = view(problem.ul.bounds, 2, :)'
    D = length(a)

    X = a .+ (b - a) .* rand(N, D)

    population_ = []
    for i in 1:N
        x = X[i,:]
        ll_sols = lower_level_optimizer(status, parameters, problem, information, options, x)
        for ll_sol in ll_sols
            push!(population_, create_solution(x, ll_sol, problem))
        end
    end
    population = [s for s in population_]

    best_solution = Metaheuristics.get_best(population)

    return BLState(best_solution, population)
end

