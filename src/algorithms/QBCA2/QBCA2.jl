mutable struct QBCA2 <: Metaheuristics.AbstractParameters
    N::Int
    K::Int
    η_max::Float64
    δ1::Float64
    δ2::Float64
    ε1::Float64
    ε2::Float64
    t_reevaluate::Int
    autodiff::Symbol # :finite or :forward
    use_surrogate_model::Bool
    λ::Float64
end

include("lower-level.jl")

function QBCA2(;N = 0,
        K = 3,
        η_max=2.0,
        δ1 = 1e-2,
        δ2 = 1e-2,
        ε1 = 1e-2,
        ε2 = 1e-2,
        t_reevaluate=10,
        use_surrogate_model = false,
        λ = 1e-12,
        autodiff = :finite,
        options_ul = Metaheuristics.Options(),
        options_ll = Metaheuristics.Options(),
        information_ul = Metaheuristics.Information(),
        information_ll = Metaheuristics.Information()
    )

    parameters = QBCA2(N, K, η_max, δ1, δ2, ε1, ε2, t_reevaluate, autodiff, use_surrogate_model, λ)


    return Algorithm(parameters;
                     options = BLOptions(options_ul, options_ll),
                     information = BLInformation(information_ul, information_ll)
                    )
end


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


function initialize!(
        status, # an initialized State (if apply)
        parameters::QBCA2,
        problem,
        information,
        options,
        args...;
        kargs...
    )
    a = view(problem.ul.bounds, 1, :)'
    b = view(problem.ul.bounds, 2, :)'
    D = length(a)
    D_ll = size(problem.ll.bounds, 2)


    #### initialize budget and parameters
    if parameters.N == 0
        parameters.N = 2parameters.K * D
    end

    if options.ul.f_calls_limit == 0
        options.ul.f_calls_limit = 1000*D
        if options.ul.iterations == 0
            options.ul.iterations = options.ul.f_calls_limit ÷ parameters.K
        end
    end

    if options.ll.f_calls_limit == 0
        options.ll.f_calls_limit = 1000*D_ll
        if options.ll.iterations == 0
            options.ll.iterations = options.ll.f_calls_limit ÷ parameters.K
        end
    end
    #################

    N = parameters.N
    K = parameters.K
    resize_population = false

    X = a .+ (b - a) .* rand(N, D)

    population_ = []

    for i in 1:parameters.N
        x = X[i,:]
        ll_sols = lower_level_optimizer(status, parameters, problem, information, options, x)
        for ll_sol in ll_sols
            ll_sol_improved = BFGS_LL(x, ll_sol.x, parameters, problem, information, options)
            push!(population_, create_solution(x, ll_sol_improved, problem))
        end
    end

    population = [s for s in population_]

    truncate_population!(status, parameters, problem, information, options, (a, b) -> is_better_qbca2(a,b, parameters))
    
    best = Metaheuristics.get_best(population)
    return BLState(best, population) # replace this
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

function update_state!(
        status,
        parameters::QBCA2,
        problem,
        information,
        options,
        args...;
        kargs...
    )

    if parameters.use_surrogate_model
        surrogate_search!(status, parameters, problem, information, options)
    end
    

    a = problem.ul.bounds[1,:]
    b = problem.ul.bounds[2,:]
    D = length(a)

    X = zeros(parameters.N, D)

    for i in 1:parameters.N
        X[i,:] = leader_pos(status.population[i])
    end


    I = randperm(parameters.N)
    J = randperm(parameters.N)

    is_better(a, b) = is_better_qbca2(a,b, parameters)
    
    population = status.population
    # success_rate = 0.0

    K = parameters.K
    for i in 1:parameters.N

        U = Metaheuristics.getU(population, K, I, i, parameters.N)
        # stepsize
        ηX = parameters.η_max * rand()


        cX, u_worst = center_ul(U, parameters)

        # u: worst element in U
        u = leader_pos(U[u_worst])

        x = leader_pos(population[i])
        p = x .+ ηX .* (cX .- u)
        Metaheuristics.replace_with_random_in_bounds!(p, problem.ul.bounds)
        ll_sols = lower_level_optimizer(status, parameters, problem, information, options, p)


        for ll_sol in ll_sols
            sol = create_solution(p, ll_sol, problem)

            if is_better_qbca2(sol, status.population[i], parameters)
                i_worst = findworst(status.population, is_better)
                status.population[i_worst] = sol


                if is_better_qbca2(sol, status.best_sol, parameters)
                    status.best_sol = sol
                end
            end

        end
        
    end

    reevaluate!(status, parameters, problem, information, options)

end


function stop_criteria!(status, parameters::QBCA2, problem, information, options)
    status.stop = status.stop || fitness_variance_stop_check(status, information, options)
    return 
end

function final_stage!(
        status,
        parameters::QBCA2,
        problem,
        information,
        options,
        args...;
        kargs...
    )
    return
end
