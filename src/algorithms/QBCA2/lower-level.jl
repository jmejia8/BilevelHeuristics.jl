include("surrogate-eca.jl")

struct BLSimplexer{T} <: Optim.Simplexer
    solutions::T
end
BLSimplexer(;solutions=Vector{Float64}[]) = BLSimplexer(solutions)

function Optim.simplexer(A::BLSimplexer, initial_x::AbstractArray{T, N}) where {T, N}
    n = length(initial_x)
    initial_simplex = Array{T, N}[sol for sol in A.solutions]

    initial_simplex
end

function neldermead(f, status::Metaheuristics.State)
    sort!( status.population, lt = (a, b) -> a.f < b.f )

    D = status.best_sol.x
    sols = [ status.population[i].x for i in 1:length(D)+1 ]
    return Optim.optimize(f, status.best_sol.x, Optim.NelderMead(initial_simplex = BLSimplexer(sols)))
    
end

function gen_optimal(x, problem, parameters, options)
    f(y) = Metaheuristics.evaluate(x, y, problem.ll) 

    bounds = problem.ll.search_space

    D = Metaheuristics.getdim(bounds)

    options.ll.seed = rand(UInt) # this is important

    f_calls_limit = options.ll.f_calls_limit
    if parameters.use_surrogate_model
        eca = SECA(K=7, N = 7*D, λ=parameters.λ,options=options.ll)
    else
        eca = Metaheuristics.ECA(K=7, N = 7*D, options=options.ll)
    end
    

    eca.options.f_calls_limit = f_calls_limit
    res = Metaheuristics.optimize(f, bounds, eca)

    res_local = neldermead(f, res)
    res.best_sol.x = res_local.minimizer
    res.best_sol.f = res_local.minimum
    return res.best_sol
end

function BFGS_LL(x, y0, parameters, problem, information, options)
    f(y) = Metaheuristics.evaluate(x, y, problem.ll)

    Metaheuristics.reset_to_violated_bounds!(y0, problem.ll.search_space)

    options_bfgs = Optim.Options(f_calls_limit=1000, outer_iterations=2, f_tol=1e-8)
    method = Optim.Fminbox(Optim.BFGS(linesearch = LineSearches.BackTracking(order=3)))
    # approx
    r = Optim.optimize(f,
                       problem.ll.search_space.lb,
                       problem.ll.search_space.ub,
                       y0,
                       method,
                       options_bfgs
                      )

    return Metaheuristics.create_child( Optim.minimizer(r), Optim.minimum(r))
end


function center_ll(U, parameters)
    fitness = map(u -> follower_f(u), U)
    mass = Metaheuristics.fitnessToMass(fitness)

    d = length(follower_pos(U[1]))

    c = zeros(Float64, d)

    for i = 1:length(mass)
        c += mass[i] .* follower_pos(U[i])
    end

    return c / sum(mass)
end


function lower_level_optimizer(
        status, # an initialized State (if apply)
        parameters::QBCA2,
        problem,
        information,
        options,
        x,
        args...;
        kargs...
    )

    K = 3
    D_ll = Metaheuristics.getdim(problem.ll.search_space)

    y = nothing
    f_calls = 0
    if length(status.population) > K
        n = length(status.population)

        distances = map( sol -> norm( leader_pos(sol) - x), status.population )
        I = sortperm(distances)
        V = status.population[I[1:K]]


        c = center_ll(V, parameters)
        y = Metaheuristics.replace_with_random_in_bounds!(c, problem.ll.search_space)

        ########## Improve Stage ##########        
        sol = BFGS_LL(x, y, parameters, problem, information, options)
    else
        sol = gen_optimal(x, problem, parameters, options)
    end

    return [sol]

end

