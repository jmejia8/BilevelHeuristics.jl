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
        # prior information
        population = []
        for i = 1:length(local_population)
            y = follower_pos(local_population[i])
            push!(population, Metaheuristics.generateChild(y, f(y)))
        end
        # Complete with random
        a = bounds[1, :]
        b = bounds[2, :]
        while length(population) < N
            y = a + (b - a) .* rand(D)
            push!(population, Metaheuristics.generateChild(y, f(y)))
        end
        # tell eca you have generated solutions
        eca.parameters.N = length(population)
        eca.status = Metaheuristics.State(Metaheuristics.get_best(population), population)

    end

    eca.options.f_calls_limit = f_calls_limit
    res = Metaheuristics.optimize(f, bounds, eca)

    return res.best_sol
end


function Metaheuristics.stop_criteria!(
        status,
        parameters::Metaheuristics.ECA,
        problem,
        information,
        options,
        args...;
        kargs...
    )
    status.stop = status.stop ||
                    Metaheuristics.call_limit_stop_check(status, information, options) ||
                    Metaheuristics.iteration_stop_check(status, information, options)  ||
                    Metaheuristics.time_stop_check(status, information, options)       ||
                    Metaheuristics.accuracy_stop_check(status, information, options)   ||
                    Metaheuristics.diff_check(status, information, options; d = 1e-8)

end

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
        distances = map( sol -> norm( leader_pos(sol) - p), status.population )
        I = sortperm(distances)
        C = status.population[I[1:D]] # the nearest to p
        sol = gen_optimal_sabo(p, problem, C)
    else
        sol = gen_optimal(p, problem, parameters, options)
    end

    return [sol]

end


end
