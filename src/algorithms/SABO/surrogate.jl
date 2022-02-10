function surrogate_search!(
        status,
        parameters::SABO,
        problem,
        information,
        options,
        args...;
        kargs...)

    N = length(status.population)
    a = problem.ul.bounds[1,:]
    b = problem.ul.bounds[2,:]

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
    res = Optim.optimize(F̂, zeros(length(a)), ones(length(a)), x_initial, optimizer, Optim.Options(outer_iterations = 1))
    p = a .+ (b - a) .* res.minimizer

    
    if norm(p - leader_pos(status.best_sol)) < 1e-2
        # nothing to do
        options.ul.debug && @info "Surrogate-based improvement not necessary."
        return
    end
    

    ll_sols = LowerLevelSABO.lower_level_optimizer(status, parameters, problem, information, options, p, true)

    sol = create_solution(p, ll_sols[1], problem)

    if leader_f(sol) < leader_f(status.best_sol)
        fv1 = leader_f(status.best_sol)

        status.best_sol = sol

        fv2 = leader_f(sol) 
        options.ul.debug && @info "Surrogate-based improvement success. $fv1 --> $fv2"
    else
        options.ul.debug && @warn "Surrogate-based improvement failed."
    end

end


function center_ul(U, parameters::SABO)
    fitness = map(u -> leader_f(u), U)
    mass = Metaheuristics.fitnessToMass(fitness)

    d = length(leader_pos(U[1]))

    c = zeros(Float64, d)

    for i = 1:length(mass)
        c += mass[i] .* leader_pos(U[i])
    end

    return c / sum(mass), argmin(mass)
end
