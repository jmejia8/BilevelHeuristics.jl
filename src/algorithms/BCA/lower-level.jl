function lower_level_optimizer(
        status,
        parameters::BCA,
        problem,
        information,
        options,
        x,
        args...;
        kargs...
    )


    ff(y) = Metaheuristics.evaluate(x, y, problem.ll)

    D =  Metaheuristics.getdim(problem.ll.search_space)

    N = parameters.n
    K = parameters.K
    η_max = parameters.η_max
    opts_ll = options.ll
    opts_ll.seed = rand(UInt32)
    
    # lower level parameters
    method = Metaheuristics.ECA(;N, η_max, K,
                                resize_population = parameters.resize_population,
                                options = opts_ll)
    # perform optimization
    res = Metaheuristics.optimize(ff, problem.ll.search_space, method)

    ############# updated to handle multi-modal problems at lower level #################
    return handle_ll_multimodality(res, problem, options) 

end

function handle_ll_multimodality(res, problem, options) 

    fmin = Metaheuristics.minimum(res)

    fs = Metaheuristics.fvals(res.population)
    mask = findall( v -> abs(v - fmin) < 1e-12, fs)

    if isnothing(mask) || length(mask) == 1
        # unique LL optimum detected
        return [res.best_sol]
    end

    ll_sols = [res.best_sol]

    # for normalization
    Diag = norm( problem.ll.search_space.Δ )

    # min distance between solution with same fitness
    d_tol = 0.01Diag

    # save all solutions far from the best solutions wit same fitness
    candidate_ll_sols = res.population[mask]

    for candidate in candidate_ll_sols
        save_sol = true

        for sol in ll_sols
            @fastmath distance_to_best = norm( sol.x - candidate.x )
            if distance_to_best < d_tol
                save_sol = false
                break
            end
        end
        save_sol && push!(ll_sols, candidate)
    end

    unique!(ll_sols)
    
    n = length(ll_sols)
    options.ul.debug && n > 1 && @info "Lower level seems multimodal ($n optimums)."
    
    return ll_sols
end


