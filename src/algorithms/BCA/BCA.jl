"""
    BCA(;N, n, K, η_max, resize_population)

Bilevel Centers Algorithm uses two nested ECA.

## Parameters
- `N` Upper level population size
- `n` Lower level population size.
- `K` Num. of solutions to generate centers.
- `η_max` Step size.

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
julia> res = optimize(F, f, bounds_ul, bounds_ll, BCA())
+=========== RESULT ==========+
  iteration: 108
    minimum: 
          F: 2.7438e-09
          f: 3.94874e-11
  minimizer: 
          x: [-8.80414308649828e-6, 2.1574853199308744e-5, -1.5550602817418899e-6, 1.9314104453973864e-5, 2.1709393089480435e-5]
          y: [-4.907639660543081e-6, 2.173986368018122e-5, -1.8133242873785074e-6, 1.9658451600356374e-5, 2.1624363965042988e-5]
    F calls: 2503
    f calls: 6272518
    Message: Stopped due UL function evaluations limitations. 
 total time: 14.8592 s
+============================+

julia> x, y = minimizer(res);

julia> x
5-element Vector{Float64}:
 -8.80414308649828e-6
  2.1574853199308744e-5
 -1.5550602817418899e-6
  1.9314104453973864e-5
  2.1709393089480435e-5

julia> y
5-element Vector{Float64}:
 -4.907639660543081e-6
  2.173986368018122e-5
 -1.8133242873785074e-6
  1.9658451600356374e-5
  2.1624363965042988e-5

julia> Fmin, fmin = minimum(res)
(2.7438003987697017e-9, 3.9487399650845625e-11)
```

## Citation
> Mejía-de-Dios, J. A., & Mezura-Montes, E. (2018, November). A physics-inspired algorithm
> for bilevel optimization. In 2018 IEEE International Autumn Meeting on Power, Electronics
> and Computing (ROPEC) (pp. 1-6). IEEE.
"""
mutable struct BCA <: Metaheuristics.AbstractParameters
    N::Int
    n::Int
    K::Int
    η_max::Float64
    resize_population::Bool
end

include("lower-level.jl")

function BCA(;N = 0, n=0, K = 7, η_max=2.0, resize_population = true,
        options_ul = Metaheuristics.Options(),
        options_ll = Metaheuristics.Options(),
        information_ul = Metaheuristics.Information(),
        information_ll = Metaheuristics.Information()
    )

    parameters = BCA(N, n,  K, η_max, resize_population)

    Algorithm(
        parameters;
        options = BLOptions(options_ul, options_ll),
        information = BLInformation(information_ul, information_ll)
    )
end


function is_better_bca(A::BLIndividual, B::BLIndividual)
    QxyA = leader_f(A) + follower_f(A)
    QxyB = leader_f(B) + follower_f(B)

    return QxyA < QxyB

end


function truncate_population!(status, parameters, problem, information, options, is_better)
    if parameters.N == length(status.population)
        return
    end

    N = parameters.N
    sort!(status.population, lt = is_better)
    deleteat!(status.population, N + 1:length(status.population))

    return
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

    a = view(problem.ul.bounds, 1, :)'
    b = view(problem.ul.bounds, 2, :)'
    D = length(a)
    D_ll = size(problem.ll.bounds, 2)

    #### initialize budget and parameters
    if parameters.N == 0
        parameters.N = parameters.K * D
    end

    if parameters.n == 0
        parameters.n = parameters.K * D_ll
    end

    if options.ul.f_calls_limit == 0
        options.ul.f_calls_limit = 500*D
        if options.ul.iterations == 0
            options.ul.iterations = options.ul.f_calls_limit ÷ parameters.K
        end
    end

    if options.ll.f_calls_limit == 0
        options.ll.f_calls_limit = 500*D_ll
        if options.ll.iterations == 0
            options.ll.iterations = options.ll.f_calls_limit ÷ parameters.K
        end
    end
    #################

    N = parameters.N

    X = a .+ (b - a) .* rand(N, D)

    population_ = []
    for i in 1:parameters.N
        x = X[i,:]
        ll_sols = lower_level_optimizer(status, parameters, problem, information, options, x)
        for ll_sol in ll_sols
            push!(population_, create_solution(x, ll_sol, problem))
        end
    end
    population = [s for s in population_]


    truncate_population!(status, parameters, problem, information, options, is_better_bca)

    best = Metaheuristics.get_best(population)
    return BLState(best, population) # replace this
end


function center_ul(U, parameters::BCA)
    fitness = map(u -> leader_f(u) + follower_f(u), U)
    mass = Metaheuristics.fitnessToMass(fitness)

    d = length(leader_pos(U[1]))

    c = zeros(Float64, d)

    for i = 1:length(mass)
        c += mass[i] .* leader_pos(U[i])
    end

    return c / sum(mass), argmin(mass)
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


    a = problem.ul.bounds[1,:]
    b = problem.ul.bounds[2,:]
    D = length(a)

    I = randperm(parameters.N)

    population = status.population

    p = status.F_calls / options.ul.f_calls_limit
    K = parameters.K
    for i in 1:parameters.N

        U = Metaheuristics.getU(population, K, I, i, parameters.N)
        # stepsize
        η = parameters.η_max * rand()


        c, u_worst = center_ul(U, parameters)

        # u: worst element in U
        u = leader_pos(U[u_worst])

        x = leader_pos(population[i]) .+ η .* (c .- u)

        Metaheuristics.replace_with_random_in_bounds!(x, problem.ul.bounds)

        ll_sols = lower_level_optimizer(status, parameters, problem, information, options, x)

        for ll_sol in ll_sols
            sol = create_solution(x, ll_sol, problem)
            push!(status.population, sol)

            if is_better_bca(sol, status.best_sol)
                status.best_sol = sol
            end
        end


    end

    parameters.N = round(Int, parameters.K*(D - (D-2)*p))
    truncate_population!(status, parameters, problem, information, options, is_better_bca)

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
