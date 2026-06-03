function show_status_oneline(status, parameters, options)
    # sorry for the hard code :-)
    !options.ul.verbose && (return)
    d = Any[
         "Iteration" => status.iteration,
         "UL Evals" => status.F_calls,
         "LL Evals" => status.f_calls,
        ]
    # show header

    t = status.iteration
    Fmin, fmin = minimum(status)
    if fmin isa Number
        push!(d, "UL Min" => Fmin)
        push!(d, "LL Min" => fmin)
    else
        _p = get_ul_population(status.population)
        n = length(Metaheuristics.get_non_dominated_solutions(_p))
        s = sprint(print, "$n/$(length(status.population))")
        push!(d, "NDS" => s)
        # 
    end

    n = count(Metaheuristics.is_feasible.(status.population))
    push!(d, "Feasibles" => sprint(print, n, " / ", length(status.population)))
    push!(d, "Time" => @sprintf("%.4f s", status.overall_time))


    if status.iteration <= 1 || status.iteration % 1000 == 0
        nm = [@sprintf(" % 10s ", string(v)) for v in first.(d)]
        lines = [fill('-', length(n) ) |> join for n in nm]
        println("+", join(lines, "+"), "+")
        println("|", join(nm, "|"), "|")
        println("+", join(lines, "+"), "+")
    end
    print("|")

    for v in last.(d)
        if v isa Integer
            txt = @sprintf("% 10d", v)
        elseif v isa AbstractString
            txt = @sprintf("% 10s", v)
        elseif v isa AbstractFloat
            txt = @sprintf("%1.4g", v)
        else 
            print(v, " | ")
            continue
        end
        @printf(" % 10s |", txt)
    end
    println("")
end

"""
    optimize(F, f, bounds_ul, bounds_ll, method = BCA(); logger = (status) -> nothing)

Solve a bilevel optimization problem using a heuristic or metaheuristic method.

The upper-level problem is `min_x F(x, y)` subject to `x ∈ bounds_ul` and any
upper-level constraints returned by `F`, while `y` must be an optimal solution of the
lower-level problem `min_y f(x, y)` subject to `y ∈ bounds_ll` and its own constraints.

## Parameters
- `F(x, y)` — upper-level objective function. Must return one of:
    - scalar value (unconstrained),
    - `(fval, g, h)` tuple where `g` / `h` are vectors of inequality / equality constraints.
- `f(x, y)` — lower-level objective function (same signature as `F`).
- `bounds_ul` — upper-level bounds, a `2 × D_ul` matrix where row 1 = lower bounds,
  row 2 = upper bounds.
- `bounds_ll` — lower-level bounds, a `2 × D_ll` matrix (same layout).
- `method` — a bilevel algorithm, e.g. [`BCA`](@ref), [`QBCA`](@ref), [`SABO`](@ref),
  [`SMS_MOBO`](@ref), etc. (default: `BCA()`).
- `logger` — a function `status -> ...` called at the end of every iteration (useful for
  custom logging or visualisation).

## Returns
A [`BLState`](@ref) object containing the best solution found, the final population,
convergence history, and function evaluation counts. Use [`minimum`](@ref) and
[`minimizer`](@ref) to extract results.

## Example (single-objective, unconstrained)

```julia
F(x, y) = sum(x.^2) + sum(y.^2)
f(x, y) = sum((x - y).^2) + y[1]^2
bounds = [-ones(5)'; ones(5)']

res = optimize(F, f, bounds, bounds, BCA())
x, y = minimizer(res)
Fmin, fmin = minimum(res)
```

## Example (constrained)

```julia
function F(x, y)
    fval = sum(x.^2) + sum(y.^2)
    g = [x[1] + x[2] - 1, x[3] - y[3] - 10]   # inequality constraints
    h = [0.0]                                     # equality constraints
    return fval, g, h
end

function f(x, y)
    fval = sum((x - y).^2) + y[1]^2
    g = [x[2] - y[1]^2 - 5]
    h = [0.0]
    return fval, g, h
end

bounds_ul = [-ones(5) ones(5)]
bounds_ll = [-ones(5) ones(5)]
res = optimize(F, f, bounds_ul, bounds_ll, BCA())
```

## Example (multi-objective)

```julia
function F(x, y)  # two upper-level objectives
    [y[1] - x[1], y[2]], [-1.0 - sum(y)], [0.0]
end

function f(x, y)  # two lower-level objectives
    y, [-x[1]^2 + sum(y .^ 2)], [0.0]
end

bounds_ul = [0.0 1.0]'
bounds_ll = [-1 -1; 1 1.0]
res = optimize(F, f, bounds_ul, bounds_ll, SMS_MOBO())
```
"""
function Metaheuristics.optimize(
        F::Function, # objective function UL
        f::Function, # objective function LL 
        bounds_ul,
        bounds_ll,
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

    problem_ul = Metaheuristics.Problem(F, bounds_ul)
    problem_ll = Metaheuristics.Problem(f, bounds_ll)
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
    elseif options.ul.verbose
        show_status_oneline(status, parameters, options)
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
        iteration_stop_check(status,  information, options) ||
        time_stop_check(status, information, options)

        # user defined stop criteria
        status.stop || stop_criteria!(status, parameters, problem, information, options)


        if options.ul.debug
            msg = "Current Status of " * string(typeof(parameters))
            @info msg
            display(status)
        elseif options.ul.verbose
            show_status_oneline(status, parameters, options)
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


