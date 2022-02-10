function call_limit_stop_check(status, information, options)
    cond = status.F_calls > options.ul.f_calls_limit
    if cond
        status.stop_msg *= "Stopped due UL function evaluations limitations. "
    end
    
    cond
end


function iteration_stop_check(status, information, options)
    cond = status.iteration >= options.ul.iterations

    if cond
        status.stop_msg *= "Stopped due completed iterations. "
    end
    
    cond
end


function accuracy_stop_check(status, information, options)
    F_opt = information.ul.f_optimum
    f_opt = information.ll.f_optimum

    if isnan(F_opt) && isnan(f_opt)
        return false
    end

    Fxy, fxy = minimum(status)

    cond1 = isnan(F_opt) ? true : abs(Fxy - F_opt) < options.ul.f_tol
    cond2 = isnan(f_opt) ? true : abs(fxy - f_opt) < options.ll.f_tol

    cond = cond1 && cond2

    if cond
        status.stop_msg *= "Stopped due accuracy met. "
    end
    
    cond
end


function fitness_variance_stop_check(status, information, options)
    Fvals = Metaheuristics.fvals(map(sol -> sol.ul, status.population))
    cond = var(Fvals) < 1e-12

    if cond
        status.stop_msg *= "Stopped due UL small fitness variance. "
    end
    
    cond
end

"""
    diff_check(status, information, options; d = options.ul.f_tol, p = 0.5)
Check the difference between best and worst objective function values in current
population (where at least %p of solution are feasible). Return `true` when such difference
is `<= d`, otherwise return `false`.
> Ref. Zielinski, K., & Laur, R. (n.d.). Stopping Criteria for Differential Evolution in
> Constrained Single-Objective Optimization. Studies in Computational Intelligence,
> 111–138. doi:10.1007/978-3-540-68830-3_4 (https://doi.org/10.1007/978-3-540-68830-3_4)
"""
function ul_diff_check(status, information, options; d = options.ul.f_tol, p = 0.3)
    mask = Metaheuristics.is_feasible.(status.population)
    p_feasibles = mean(mask)

    # not enough feasible samples?
    p_feasibles < p && (return false)

    population = map(sol -> sol.ul, status.population)
    fmin = minimum(s -> Metaheuristics.fval(s), population[mask])
    fmax = maximum(s -> Metaheuristics.fval(s), population[mask])

    cond = fmax - fmin <= d

    if cond
        status.stop_msg *= "Stopped due to diff check. "
    end

    cond
end
