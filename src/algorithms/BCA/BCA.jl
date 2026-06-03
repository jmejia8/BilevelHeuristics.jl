"""
    BCA(;N, n, K, η_max, resize_population)

Bilevel Centers Algorithm — a physics-inspired metaheuristic for single-objective bilevel
optimisation.  It uses a nested scheme where both the upper and lower levels employ a
**center-of-mass** variation operator.

## How it works

At each iteration, for every individual in the upper-level population:
1. A random subset `U` of size `K` is selected.
2. The center of mass `c` of `U` is computed, weighted by the combined fitness
   `Q = F + f` (i.e., both level objectives).
3. A new candidate `p = x_i + η · (c - u_worst)` is generated, where `u_worst` is the
   worst element in `U` and `η` is a random step size.
4. The lower-level problem is solved for `p` (using the same center-of-mass strategy),
   producing one or more optimal lower-level responses.
5. The new pair `(p, y_opt)` replaces a worse member of the population.

The population size may be dynamically reduced during the run (`resize_population`),
shifting from exploration to exploitation.

## Parameters
- `N` — upper-level population size (auto‑computed from `K × D_ul` if 0).
- `n` — lower-level population size (auto‑computed if 0).
- `K` — number of solutions used to compute the center of mass (default `7`).
  Larger `K` → faster convergence (exploitation); smaller `K` → more exploration.
- `η_max` — maximum step size for the variation operator (default `2.0`).
- `resize_population` — if `true`, the population shrinks linearly over the run
  (default `true`).

## Reference
> Mejía-de-Dios, J. A., & Mezura-Montes, E. (2018, November). A physics-inspired algorithm
> for bilevel optimization. In *2018 IEEE International Autumn Meeting on Power, Electronics
> and Computing (ROPEC)* (pp. 1–6). IEEE.
"""
mutable struct BCA <: Metaheuristics.AbstractParameters
    N::Int
    n::Int
    K::Int
    η_max::Float64
    resize_population::Bool
    N_init::Int
end

include("lower-level.jl")
include("utils.jl")

function BCA(;N = 0, n=0, K = 7, η_max=2.0, resize_population = true,
        options_ul = Metaheuristics.Options(),
        options_ll = Metaheuristics.Options(),
        information_ul = Metaheuristics.Information(),
        information_ll = Metaheuristics.Information()
    )

    parameters = BCA(N, n,  K, η_max, resize_population,N)

    Algorithm(
        parameters;
        options = BLOptions(options_ul, options_ll),
        information = BLInformation(information_ul, information_ll)
    )
end


function initialize!(
        status, # an initialized State (if apply)
        parameters::BCA,
        problem,
        information,
        options,
        args...;
        kargs...
    )

    D = Metaheuristics.getdim(problem.ul.search_space)
    D_ll =  Metaheuristics.getdim(problem.ll.search_space)

    parameters.K = max(parameters.K, 2)
    K = parameters.K

    #### initialize budget and parameters
    if parameters.N == 0
        parameters.N = clamp(K * D, K, 1000)
    end

    if parameters.n == 0
        parameters.n = clamp(K * D_ll, K, 1000)
    end

    if options.ul.f_calls_limit == 0
        options.ul.f_calls_limit = 500*D
        if options.ul.iterations == 0
            options.ul.iterations = options.ul.f_calls_limit ÷ K
        end
    end

    if options.ll.f_calls_limit == 0
        options.ll.f_calls_limit = 500*D_ll
        if options.ll.iterations == 0
            options.ll.iterations = options.ll.f_calls_limit ÷ K
        end
    end

    # used for when `resize_population = true`
    parameters.N_init = parameters.N

    status = gen_initial_state(status,problem,parameters,information,options)
    truncate_population!(status, parameters, problem, information, options, is_better_bca)
    status
end


function update_state!(
        status,
        parameters::BCA,
        problem,
        information,
        options,
        args...;
        kargs...
    )

    D = Metaheuristics.getdim(problem.ul.search_space)
    I = randperm(parameters.N)
    K = parameters.K
    population = status.population

    for i in 1:parameters.N
        # compute center of mass
        U = Metaheuristics.getU(population, K, I, i, parameters.N)
        # stepsize
        η = parameters.η_max * rand()
        c, u_worst = center_ul(U, parameters)
        # u: worst element in U
        u = leader_pos(U[u_worst])

        # new solution
        x = leader_pos(population[i]) .+ η .* (c .- u)
        Metaheuristics.replace_with_random_in_bounds!(x, problem.ul.search_space)
        
        # optimize lower leve
        ll_sols = lower_level_optimizer(status, parameters, problem, information, options, x)

        for ll_sol in ll_sols
            sol = create_solution(x, ll_sol, problem)
            push!(status.population, sol)

            if is_better_bca(sol, status.best_sol)
                status.best_sol = sol
            end
        end
    end

    parameters.resize_population && update_population_size!(status, parameters, options)
    truncate_population!(status, parameters, problem, information, options, is_better_bca)
end

function update_population_size!(status, parameters::BCA, options) 
    N_min = 2parameters.K
    N_max = parameters.N_init
    p = 1 - status.F_calls / options.ul.f_calls_limit
    parameters.N = round(Int, N_min + (N_max - N_min)*p )
end


function stop_criteria!(status, parameters::BCA, problem, information, options)
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
        parameters::BCA,
        problem,
        information,
        options,
        args...;
        kargs...
    )
    return
end
