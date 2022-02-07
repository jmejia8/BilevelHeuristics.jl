"""
    optimize(F, f, bounds_ul, bounds_ll, method = BCA(); logger = (status) -> nothing)

Approximate an optimal solution for the bilevel optimization problem `x ∈ argmin F(x, y)` with
`x ∈ bounds_ul` subject to `y ∈ argmin{f(x,y) : y ∈ bounds_ll}`.

## Parameters
- `F` upper-level objective function.
- `f` lower-level objective function.
- `bounds_ul, bounds_ll` upper and lower level boundaries (2×n matrices), respectively.
- `logger` is a functions called at the end of each iteration.

## Example

```jldoctest
julia> F(x, y) = sum(x.^2) + sum(y.^2)
F (generic function with 1 method)

julia> f(x, y) = sum((x - y).^2) + y[1]^2
f (generic function with 1 method)

julia> bounds_ul = bounds_ll = [-ones(5)'; ones(5)']
2×5 Matrix{Float64}:
 -1.0  -1.0  -1.0  -1.0  -1.0
  1.0   1.0   1.0   1.0   1.0

julia> res = optimize(F, f, bounds_ul, bounds_ll)
+=========== RESULT ==========+
  iteration: 108
    minimum: 
          F: 7.68483e-08
          f: 3.96871e-09
  minimizer: 
          x: [1.0283390421119262e-5, -0.00017833559080058394, -1.612275010196171e-5, 0.00012064585960330227, 4.38964383738248e-5]
          y: [1.154609166391327e-5, -0.0001300400306798623, 1.1811981430188257e-6, 8.868498295184257e-5, 5.732849695863675e-5]
    F calls: 2503
    f calls: 5044647
    Message: Stopped due UL function evaluations limitations. 
 total time: 21.4550 s
+============================+
```
"""
function Metaheuristics.optimize(
        F::Function, # objective function UL
        f::Function, # objective function LL 
        bounds_ul::AbstractMatrix,
        bounds_ll::AbstractMatrix,
        method::Metaheuristics.AbstractAlgorithm = BCA();
        logger::Function = (status) -> nothing,
    )



    #####################################
    # common methods
    #####################################
    information = method.information
    options = method.options
    parameters = method.parameters
    ###################################

    problem_ul = Metaheuristics.Problem(F, Array(bounds_ul))
    problem_ll = Metaheuristics.Problem(f, Array(bounds_ll))
    problem = BLProblem(problem_ul, problem_ll)
    seed!(options.ul.seed)

    ###################################

    start_time = time()

    status = method.status
    options.ul.debug && @info("Initializing population...")
    status = initialize!(status,parameters, problem, information, options)
    method.status = status
    status.F_calls = problem.ul.f_calls
    status.f_calls = problem.ll.f_calls
    status.start_time = start_time
    status.final_time = time()

    if options.ul.debug
        msg = "Current Status of " * string(typeof(parameters))
        @info msg
        display(status)
    end

    status.iteration = 1


    convergence = BLState{typeof(status.best_sol)}[]



    ###################################
    # store convergence
    ###################################
    if options.ul.store_convergence
        Metaheuristics.update_convergence!(convergence, status)
    end

    options.ul.debug && @info("Starting main loop...")

    logger(status)

    while !status.stop
        status.iteration += 1

        update_state!(status, parameters, problem, information, options)
        status.final_time = time()

        # store the number of fuction evaluations
        status.F_calls = problem.ul.f_calls
        status.f_calls = problem.ll.f_calls


        if options.ul.store_convergence
            Metaheuristics.update_convergence!(convergence, status)
        end

        status.overall_time = time() - status.start_time
        logger(status)

        # common stop criteria
        status.stop = status.stop ||
        call_limit_stop_check(status, information, options) ||
        accuracy_stop_check(status, information, options) ||
        iteration_stop_check(status,  information, options)

        # user defined stop criteria
        status.stop || stop_criteria!(status, parameters, problem, information, options)


        if options.ul.debug
            msg = "Current Status of " * string(typeof(parameters))
            @info msg
            display(status)
        end

    end

    status.overall_time = time() - status.start_time

    final_stage!(
                 status,
                 parameters,
                 problem,
                 information,
                 options
                )

    status.convergence = convergence

    return status

end
