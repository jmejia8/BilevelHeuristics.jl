module LowerLevelSABO

import LinearAlgebra: norm
import Metaheuristics
import Statistics: var
import ..follower_pos, ..leader_pos, ..gen_optimal, ..SECA

function gen_optimal_sabo(x, parameters, problem, local_population=[], accurate=false)
    f(y) = Metaheuristics.evaluate(x, y, problem.ll) 

    bounds = problem.ll.search_space
    D = Metaheuristics.getdim(bounds)
    K = parameters.K

    if accurate || length(local_population) < D
        f_calls_limit = 1000D
        N = 2K * D
    else
        N = K * D
        f_calls_limit = 100D
    end
    
    η_max = 1.2

    eca = SECA(;λ = parameters.λ,K = K, N = N, η_max = η_max)

    if !isempty(local_population)
        # prior information
        population = []
        for i = 1:length(local_population)
            y = follower_pos(local_population[i])
            push!(population, Metaheuristics.generateChild(y, f(y)))
        end
        # Complete with random
        a = bounds.lb
        b = bounds.ub
        while length(population) < N
            y = a + (b - a) .* rand(D)
            push!(population, Metaheuristics.generateChild(y, f(y)))
        end
        # tell eca you have generated solutions
        # eca.parameters.N = length(population)
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

    D = Metaheuristics.getdim(problem.ll.search_space)
    N = length(status.population)

    C = []

    if N > D
        distances = map( sol -> norm( leader_pos(sol) - p), status.population )
        I = sortperm(distances)
        C = status.population[I[1:D]] # the nearest solutions to p
    end

    sol = gen_optimal_sabo(p, parameters, problem, C, accurate)

    return [sol]

end


end
