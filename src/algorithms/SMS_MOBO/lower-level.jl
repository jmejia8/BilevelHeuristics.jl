function lower_level_optimizer(status,
        parameters::SMS_MOBO,
        problem,
        information,
        options,
        x,
        population = [],
        args...;
        kargs...
    )

    f(y) = Metaheuristics.evaluate(x, y, problem.ll)

    method = Metaheuristics.Algorithm(parameters.ll, information = information.ll, options=options.ll)
    method.options.seed = rand(UInt) # change seed at LL is important


    if !isempty(population)
        population_ll = [Metaheuristics.create_child(y, f(y)) for y in population]

        while parameters.ll.N > length(population_ll)
            D = size(problem.ll.bounds, 2)
            y = problem.ll.bounds[1,:] + (problem.ll.bounds[2,:] - problem.ll.bounds[1,:]) .* rand(D)
            c = Metaheuristics.create_child(y, f(y))
            push!(population_ll, c) 
        end
        
        method.status = Metaheuristics.State(population_ll[1], population_ll)
    end
    

    res = Metaheuristics.optimize(f, problem.ll.bounds, method)
    Metaheuristics.fast_non_dominated_sort!(res.population)

    return res.population
end

