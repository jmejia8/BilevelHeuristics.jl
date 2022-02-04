module LowerLevelSABO

import LinearAlgebra: norm
import Metaheuristics
import Statistics: var
import ..follower_pos, ..leader_pos, ..gen_optimal

function gen_optimal_sabo(x, problem, local_population)
    f(y) = Metaheuristics.evaluate(x, y, problem.ll) 

    bounds = problem.ll.bounds
    D = size(bounds, 2)

    if length(local_population) < D
        f_calls_limit = 1000D
        K = 7
    else
        K = 3
        f_calls_limit = 100D
    end
    
    η_max = 1.2
    N = K * D

    eca = Metaheuristics.ECA(K = K, N = N, η_max = η_max)

    if !isnothing(local_population)

        population = []
        for i = 1:length(local_population)
            y = follower_pos(local_population[i])
            push!(population, Metaheuristics.generateChild(y, f(y)))
        end

        a = bounds[1, :]
        b = bounds[2, :]
        while length(population) < N
            y = a + (b - a) .* rand(D)
            push!(population, Metaheuristics.generateChild(y, f(y)))
        end


        eca.parameters.N = length(population)
        eca.status = Metaheuristics.State(Metaheuristics.get_best(population), population)

    end

    eca.options.f_calls_limit = f_calls_limit
    res = Metaheuristics.optimize(f, bounds, eca)


    res = Metaheuristics.optimize(f, bounds, eca)

    # res_local = neldermead(f, res)
    # res.best_sol.x = res_local.minimizer
    # res.best_sol.f = res_local.minimum
    return res.best_sol
end

#=
function BFGS_LL(x, y0, parameters::SABO, problem, information, options)
    f(y) = Metaheuristics.evaluate(x, y, problem.ll)

    Metaheuristics.reset_to_violated_bounds!(y0, problem.ll.bounds)

    options_bfgs = Optim.Options(f_calls_limit=1000, outer_iterations=2, f_tol=1e-8)
    method = Optim.Fminbox(Optim.BFGS(linesearch = LineSearches.BackTracking(order=3)))
    # approx
    r = Optim.optimize(f,
                       problem.ll.bounds[1, :],
                       problem.ll.bounds[2, :],
                       y0,
                       method,
                       options_bfgs
                      )

    return Metaheuristics.create_child( Optim.minimizer(r), Optim.minimum(r))
end
=#



function lower_level_optimizer(
        status, # an initialized State (if apply)
        parameters,
        problem,
        information,
        options,
        p,
        accurate = false,
        args...;
        kargs...
    )

    D = size(problem.ll.bounds, 2)

    if !accurate && length(status.population) > D
        n = length(status.population)

        distances = map( sol -> norm( leader_pos(sol) - p), status.population )
        I = sortperm(distances)
        C = status.population[I[1:D]] # the nearest to p
        sol = gen_optimal_sabo(p, problem, C)
    else
        sol = gen_optimal(p, problem, parameters, options)
    end


    ########## Improve Stage ##########        
    # sol = BFGS_LL(p, sol.x, parameters, problem, information, options)

    return [sol]

end

#=
function lower_level_optimizer(
        status, # an initialized State (if apply)
        parameters::SABO,
        problem,
        information,
        options,
        x,
        args...;
        kargs...
    )

    K = parameters.K
    D_ll = size(problem.ll.bounds, 2)

    f_calls = 0
    if length(status.population) > 3K
        n = length(status.population)

        distances = map( sol -> norm( leader_pos(sol) - x), status.population )
        I = sortperm(distances)
        V = status.population[I[1:K]]


        y = center_ll(V, parameters)
        Metaheuristics.replace_with_random_in_bounds!(y, problem.ll.bounds)

        ########## Improve Stage ##########        
        sol = BFGS_LL(x, y, parameters, problem, information, options)
    else
        sol = gen_optimal(x, problem)
    end

    return [sol]

end
=#

end
