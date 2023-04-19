"""
    QBCA2(;N, K, η_max, δ1, δ2, ε1, ε2, λ, t_reevaluate)

Quasi-newton BCA uses ECA (upper-level) and BFGS (lower-level).

## Parameters
- `N` Upper level population size
- `K` Num. of solutions to generate centers.
- `η_max` Step size
- `δ1`, `δ2`, `ε1` `ε2` Parameters for conditions to avoid pseudo-feasible solutions.

## Usage

Upper level problem: `F(x,y)` with `x` as the upper-level vector.

```julia-repl
julia> F(x, y) = sum(x.^2) + sum(y.^2)
F (generic function with 1 method)

julia> bounds_ul = [-ones(5) ones(5)];

```

Lower level problem: `f(x, y)` with `y` as the lower-level vector.

```julia-repl
julia> f(x, y) = sum((x - y).^2) + y[1]^2
f (generic function with 1 method)

julia> bounds_ll = [-ones(5) ones(5)];

```

Approximate solution.

```julia-repl
julia> res = optimize(F, f, bounds_ul, bounds_ll, QBCA2())
+=========== RESULT ==========+
  iteration: 59
    minimum: 
          F: 3.95536e-07
          f: 9.2123e-11
  minimizer: 
          x: [-1.3573722472608445e-5, -0.00012074600446520904, -0.00035025471067487137, -0.0002315301345354928, -8.239473503719106e-5]
          y: [-6.786860530140986e-6, -0.00012074599672993571, -0.00035025467380887673, -0.00023153010993486042, -8.239472729799499e-5]
    F calls: 1775
    f calls: 299157
    Message: Stopped due UL small fitness variance. 
 total time: 5.6968 s
+============================+

julia> x, y = minimizer(res);

julia> x
5-element Vector{Float64}:
 -1.3573722472608445e-5
 -0.00012074600446520904
 -0.00035025471067487137
 -0.0002315301345354928
 -8.239473503719106e-5

julia> y
5-element Vector{Float64}:
 -6.786860530140986e-6
 -0.00012074599672993571
 -0.00035025467380887673
 -0.00023153010993486042
 -8.239472729799499e-5

julia> Fmin, fmin = minimum(res)
(3.9553637806596925e-7, 9.212297088378278e-11)
```

## Citation
> Mejía-de-Dios, J. A., Mezura-Montes, E., & Toledo-Hernández, P. (2022).
> Pseudo-feasible solutions in evolutionary bilevel optimization: Test problems and
> performance assessment. Applied Mathematics and Computation, 412, 126577.
"""
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
include("utils.jl")

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




function initialize!(
        status, # an initialized State (if apply)
        parameters::QBCA2,
        problem,
        information,
        options,
        args...;
        kargs...
    )
    a = problem.ul.search_space.lb'
    b = problem.ul.search_space.ub'
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
    

    a = problem.ul.search_space.lb
    b = problem.ul.search_space.ub
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
        Metaheuristics.replace_with_random_in_bounds!(p, problem.ul.search_space)
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
        parameters::QBCA2,
        problem,
        information,
        options,
        args...;
        kargs...
    )
    return
end
