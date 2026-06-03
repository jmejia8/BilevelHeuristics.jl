"""
    QBCA(;N, K, η_ul_max, η_ll_max, α, β, autodiff)

Quasi-Newton Bilevel Centers Algorithm — combines the center-of-mass variation (upper
level) with a **BFGS quasi-Newton solver** (lower level) and **Tikhonov regularisation**
for improved lower-level accuracy.

## How it works

The upper-level search follows the same center-of-mass strategy as [`BCA`](@ref).  The key
difference is in the lower-level solver:
1. For a given upper-level `x`, the lower level is first explored with a lightweight ECA
   centre-of-mass search to get an approximate `y`.
2. That `y` is then refined by a BFGS quasi-Newton method, minimising a Tikhonov-
   regularised objective `f(x, y) + α·‖y‖² + β·‖y - y₀‖²`, which stabilises the
   lower-level solution and prevents overfitting to a single `x`.
3. Both the upper and lower levels compute their own centers of mass, and the
   lower-level step also uses a separate step size `η_ll_max`.

This hybrid approach reduces the number of lower-level function evaluations compared to
BCA, especially on problems where the lower level is smooth and unimodal.

## Parameters
- `N` — upper-level population size (auto‑computed if 0).
- `K` — number of solutions to generate centers (default `3`).
- `η_ul_max` — upper-level step size (default `2.0`).
- `η_ll_max` — lower-level step size (default `1/η_ul_max`).
- `α`, `β` — Tikhonov regularisation weights (default `0.05` each).
- `autodiff` — differentiation mode for BFGS: `:finite` (finite differences) or
  `:forward` (forward-mode AD) (default `:finite`).

## Reference
> Mejía-de-Dios, J. A., & Mezura-Montes, E. (2019, June). A metaheuristic for bilevel
> optimization using Tikhonov regularization and the quasi-Newton method. In *2019 IEEE
> Congress on Evolutionary Computation (CEC)* (pp. 3134–3141). IEEE.
"""
mutable struct QBCA <: Metaheuristics.AbstractParameters
    N::Int
    K::Int
    η_ul_max::Float64
    η_ll_max::Float64
    α::Float64
    β::Float64
    s_min::Float64
    autodiff::Symbol # :finite or :forward
end

include("lower-level.jl")
include("utils.jl")

function QBCA(;N = 0,
        K = 3,
        η_ul_max=2.0,
        η_ll_max=1.0/η_ul_max,
        α = 1/20,
        β = 1/20,
        autodiff = :finite,
        options_ul = Metaheuristics.Options(),
        options_ll = Metaheuristics.Options(),
        information_ul = Metaheuristics.Information(),
        information_ll = Metaheuristics.Information()
    )

    s_min = 0.0
    parameters = QBCA(N, K, η_ul_max, η_ll_max, α, β, s_min, autodiff)

    return Algorithm(parameters;
                     options = BLOptions(options_ul, options_ll),
                     information = BLInformation(information_ul, information_ll)
                    )
end


function initialize!(
        status, # an initialized State (if apply)
        parameters::QBCA,
        problem,
        information,
        options,
        args...;
        kargs...
    )
    a = problem.ul.search_space.lb' # lower bounds (UL)
    b = problem.ul.search_space.ub' # upper bounds (UL)
    D = length(a)
    D_ll = Metaheuristics.getdim(problem.ll.search_space)


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

    # only for initialization
    bca = BCA(N, 3*D, 3, parameters.η_ul_max, resize_population,N)

    for i in 1:parameters.N
        x = X[i,:]
        ll_sols = lower_level_optimizer(status, bca, problem, information, options, x)
        for ll_sol in ll_sols
            ll_sol_improved = BFGS_LL(x, ll_sol.x, parameters, problem, information, options)
            push!(population_, create_solution(x, ll_sol_improved, problem))
        end
    end

    population = [s for s in population_]

    truncate_population!(status, parameters, problem, information, options, (a, b) -> is_better_qbca(a,b, parameters))
    
    best = Metaheuristics.get_best(population)
    return BLState(best, population) # replace this
end

function update_state!(
        status,
        parameters::QBCA,
        problem,
        information,
        options,
        args...;
        kargs...
    )


    a = problem.ul.search_space.lb
    b = problem.ul.search_space.ub
    D = length(a)
    α = parameters.α
    β = parameters.β

    X = zeros(parameters.N, D)

    for i in 1:parameters.N
        X[i,:] = leader_pos(status.population[i])
    end


    I = randperm(parameters.N)
    J = randperm(parameters.N)

    is_better(a, b) = is_better_qbca(a,b, parameters)
    
    population = status.population
    # success_rate = 0.0

    K = parameters.K
    for i in 1:parameters.N
        # center of mass
        U = Metaheuristics.getU(population, K, I, i, parameters.N)
        V = Metaheuristics.getU(population, K, J, i, parameters.N)
        cX, u_worst = center_ul(U, parameters)
        cY, v_worst = center_ll(V, parameters)

        # stepsize
        ηX = parameters.η_ul_max * rand()
        ηY = parameters.η_ll_max * rand()

        # u: worst element in U
        u = leader_pos(U[u_worst])
        v = follower_pos(V[v_worst])
        x = leader_pos(population[i])

        # solution candidate
        p = x .+ ηX .* (cX .- u)
        Metaheuristics.replace_with_random_in_bounds!(p, problem.ul.search_space)

        y_nearest, d = nearest(status.population, p)
        if d >= 1e-16
            vv = cY - v
            yc = y_nearest + (ηY / norm(vv)) * vv
            Metaheuristics.replace_with_random_in_bounds!(yc, problem.ll.search_space)
            ll_sols = lower_level_optimizer(status, parameters, problem, information, options, p, yc)
        else
            fxy = Metaheuristics.evaluate(p, y_nearest, problem.ll)
            ll_sols = [Metaheuristics.create_child(p, y_nearest, fxy)]
        end

        for ll_sol in ll_sols
            sol = create_solution(p, ll_sol, problem)
            # success_rate += 1.0/parameters.N

            if is_better_qbca(sol, status.population[i], parameters)
                i_worst = findworst(status.population, is_better)
                status.population[i_worst] = sol


                if is_better_qbca(sol, status.best_sol, parameters)
                    status.best_sol = sol
                end
            end
        end 
    end

end


function stop_criteria!(status, parameters::QBCA, problem, information, options)
    status.stop = status.stop || fitness_variance_stop_check(status, information, options)

    if !isnan(information.ul.f_optimum)
        # nothing to do.  ul_diff_check should not be applied
        # when optimum is known
        return
    else
        status.stop = status.stop || ul_diff_check(status, information, options)
    end
    
    return 
end

function final_stage!(
        status,
        parameters::QBCA,
        problem,
        information,
        options,
        args...;
        kargs...
    )
    return
end
