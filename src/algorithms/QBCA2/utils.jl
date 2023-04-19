function is_better_qbca2(A, B, parameters::QBCA2)
    ΔF = abs(leader_f(A) - leader_f(B))
    Δf = abs(follower_f(A) - follower_f(B))

    Δx = leader_pos(A) - leader_pos(B)
    Δy = follower_pos(A) - follower_pos(B)

    δ1 = parameters.δ1
    δ2 = parameters.δ2
    ε1 = parameters.ε1
    ε2 = parameters.ε2

    if Δf < ε2 && ΔF < ε1 && norm(Δx) < δ1 && norm(Δy) > δ2
        return false
    end

    return leader_f(A) < leader_f(B) 
end

function center_ul(U, parameters::QBCA2)
    fitness = map(u -> leader_f(u), U)
    mass = Metaheuristics.fitnessToMass(fitness)

    d = length(leader_pos(U[1]))

    c = zeros(Float64, d)

    for i = 1:length(mass)
        c += mass[i] .* leader_pos(U[i])
    end

    return c / sum(mass), argmin(mass)
end


function reevaluate!(
        status,
        parameters::QBCA2,
        problem,
        information,
        options,
        args...;
        kargs...)

    if status.iteration > 1 && status.iteration % parameters.t_reevaluate != 0
        return
    end
    
    # First, reevaluate elite solution
    options.ul.debug && @info "Re-evaluating best solution..."

    x, y_best = minimizer(status)
    ll_sol = gen_optimal(x, problem, parameters, options)

    sol = create_solution(x, ll_sol, problem)
    status.best_sol.ul = sol.ul
    status.best_sol.ll = sol.ll

    y = ll_sol.x
    if norm(y - y_best) < 1e-2
        return
    end
    # if the elite solution was updated, the entire population is reevaluated as well
    options.ul.debug && @info "Re-evaluating entire population ..."

    for sol in status.population
        ll_sol = gen_optimal(leader_pos(sol), problem, parameters, options)
        sol_new = create_solution(leader_pos(sol), ll_sol, problem)
        sol.ul = sol_new.ul
        sol.ll = sol_new.ll
        if leader_f(status.best_sol) > leader_f(sol)
            status.best_sol = sol
            options.ul.debug && @info "Best solution found in reevaluation."
        end
    end

end

function surrogate_search!(
        status,
        parameters::QBCA2,
        problem,
        information,
        options,
        args...;
        kargs...)

    N = length(status.population)
    a = problem.ul.search_space.lb
    b = problem.ul.search_space.ub

    X = zeros(N, length(a))# map(sol -> leader_pos(sol)', status.population)
    for i = 1:N
        X[i,:] = leader_pos(status.population[i])
    end
    

    y = map(sol -> leader_f(sol), status.population)
    
    X = (X .- a') ./ (b - a)'
    method = BiApprox.KernelInterpolation(y, X, λ = parameters.λ)
    BiApprox.train!(method)
    F̂ = BiApprox.approximate(method)

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    x_initial = (leader_pos(status.best_sol) - a) ./ (b - a)
    optimizer = Optim.Fminbox(Optim.BFGS())
    res = Optim.optimize(F̂, a, b, x_initial, optimizer, Optim.Options(outer_iterations = 1))
    p = a .+ (b - a) .* res.minimizer

    
    if norm(p - leader_pos(status.best_sol)) < 1e-2
        # nothing to do
        options.ul.debug && @info "Surrogate-based improvement not necessary."
        return
    end
    

    ll_sol = gen_optimal(p, problem, parameters, options)
    sol = create_solution(p, ll_sol, problem)


    if leader_f(sol) < leader_f(status.best_sol)
        fv1 = leader_f(status.best_sol)

        status.best_sol.ul = sol.ul
        status.best_sol.ll = sol.ll

        fv2 = leader_f(sol) 
        options.ul.debug && @info "Surrogate-based improvement success. $fv1 --> $fv2"
    else
        options.ul.debug && @warn "Surrogate-based improvement failed."
    end

end
