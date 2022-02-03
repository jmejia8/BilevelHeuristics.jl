function call_limit_stop_check(status, information, options)
    cond = status.F_calls > options.ul.f_calls_limit
    if cond
        status.stop_msg *= "Stopped due UL function evaluations limitations. "
    end
    
    cond
end


function iteration_stop_check(status, information, options)
    cond = status.iteration > options.ul.iterations

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

    # when only provides upper or lower level best known solution
    F_opt = isnan(F_opt) ? Fxy : F_opt
    f_opt = isnan(f_opt) ? fxy : f_opt


    cond = abs(Fxy - F_opt) < options.ul.f_tol && abs(fxy - f_opt) < options.ll.f_tol


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
