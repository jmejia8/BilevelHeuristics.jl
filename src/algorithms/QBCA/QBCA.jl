"""
    QBCA(;N, K, η_ul_max, η_ll_max, α, β, autodiff)

Quasi-newton BCA uses ECA (upper-level) and BFGS (lower-level).

## Parameters
- `N` Upper level population size
- `K` Num. of solutions to generate centers.
- `η_ul_max` UL step size.
- `η_ll_max` LL step size.
- `α, β` Parameters for the Tikhnonov regularization.
- `autodiff=:finite` Used to approximate LL derivates.

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
julia> res = optimize(F, f, bounds_ul, bounds_ll, QBCA())
+=========== RESULT ==========+
  iteration: 71
    minimum: 
          F: 1.20277e-06
          f: 1.8618e-08
  minimizer: 
          x: [-0.00019296602928680934, -0.00031720504506331244, 0.00047217689470620765, 0.00014459596611862214, 0.00048345619641040644]
          y: [-9.647494056567316e-5, -0.0003171519406858993, 0.00047209784939209284, 0.00014457176048263256, 0.0004833752613377002]
    F calls: 2130
    f calls: 366743
    Message: Stopped due UL small fitness variance. 
 total time: 7.7909 s
+============================+

julia> x, y = minimizer(res);

julia> x
5-element Vector{Float64}:
 -0.00019296602928680934
 -0.00031720504506331244
  0.00047217689470620765
  0.00014459596611862214
  0.00048345619641040644

julia> y
5-element Vector{Float64}:
 -9.647494056567316e-5
 -0.0003171519406858993
  0.00047209784939209284
  0.00014457176048263256
  0.0004833752613377002

julia> Fmin, fmin = minimum(res)
(1.2027656204730873e-6, 1.8617960564375732e-8)
```

## Citation
> Mejía-de-Dios, J. A., & Mezura-Montes, E. (2019, June). A metaheuristic for bilevel
> optimization using tykhonov regularization and the quasi-newton method. In 2019 IEEE
> Congress on Evolutionary Computation (CEC) (pp. 3134-3141). IEEE.
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
