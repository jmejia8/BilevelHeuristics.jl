
struct SECA <: Metaheuristics.AbstractParameters
    eca_params::Metaheuristics.ECA
    λ::Float64
end

function Metaheuristics.initialize!(
                status, # an initiliazed State (if apply)
                parameters::SECA,
                problem,
                information,
                options,
                args...;
                kargs...
        )

    # initialize the stuff here
    return Metaheuristics.initialize!(status, parameters.eca_params, problem, information, options,args...;kargs...)
end

function Metaheuristics.stop_criteria!( status,
        parameters::SECA,
        problem,
        information,
        options,
        args...;
        kargs...)

    if status.stop
        return
    end
    

    fs = Metaheuristics.fvals(status.population)
    a = minimum(fs)
    b = maximum(fs)
    status.stop = abs(a - b) < 1e-12

    Metaheuristics.stop_criteria!(status, parameters.eca_params, problem, information, options,args...;kargs...)

    nothing
end

function use_surrogate!( status,
        parameters::SECA,
        problem,
        information,
        options,
        args...;
        kargs...)
    
    
    

    a = problem.bounds[1,:]
    b = problem.bounds[2,:]

    Y = Metaheuristics.fvals(status.population)
    Y /= maximum(abs.(Y))
    N = size(Y,1)

    D = length(a)
    X = zeros(N, D)

    for i = 1:N
        X[i, : ] = status.population[i].x
    end
    

    X = (X .- a') ./ (b - a)'
    method = BiApprox.KernelInterpolation(Y, X, λ = parameters.λ)
    BiApprox.train!(method)
    ff = BiApprox.approximate(method)

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    x_initial = (status.best_sol.x - a) ./ (b - a)

    optimizer = Optim.Fminbox(Optim.BFGS())
    opts = Optim.Options(outer_iterations = 1, f_calls_limit=D^2)

    res = Optim.optimize(ff, zeros(length(a)), ones(length(a)), x_initial, optimizer, opts)
    x_new = a .+ (b - a) .* res.minimizer
    f_new = Metaheuristics.evaluate(x_new, problem)
    if f_new < status.best_sol.f
        # @info(status.best_sol.f, " ---> ", f_new)
        status.best_sol.f = f_new
        status.best_sol.x = x_new
    end

end


function Metaheuristics.update_state!( status,
        parameters::SECA,
        problem,
        information,
        options,
        args...;
        kargs...)


    use_surrogate!(status, parameters, problem, information, options,args...;kargs...)
    
    Metaheuristics.update_state!(status, parameters.eca_params, problem, information, options,args...;kargs...)

    nothing
end


function Metaheuristics.final_stage!( status,
        parameters::SECA,
        problem,
        information,
        options,
        args...;
        kargs...)


    Metaheuristics.final_stage!(status, parameters.eca_params, problem, information, options,args...;kargs...)
    nothing
end

function SECA(;λ = 0.0, kargs...)
    eca = Metaheuristics.ECA(;kargs...)
    parameters = SECA(eca.parameters, λ)


    Metaheuristics.Algorithm(
        parameters,
        information = eca.information,
        options = eca.options
    )
end

