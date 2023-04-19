include("modified-nsga2.jl")

function lower_level_optimizer(status,
        parameters::BLEMO,
        problem,
        information,
        options,
        x,
        population = [],
        elite_set = [],
        args...;
        kargs...
    )

    f(y) = Metaheuristics.evaluate(x, y, problem.ll)

    if !isempty(elite_set)
        elite_set_ll = [s[2] for s in elite_set]
        method = NSGA2_LL(parameters.ll; elite_set=elite_set_ll, options=options.ll)
    else
        method = NSGA2(options=options.ll)
        method.parameters = parameters.ll
    end
    
    method.options.seed = rand(UInt) # change seed at LL is important


    if !isempty(population)
        population_ll = [Metaheuristics.create_child(y, f(y)) for y in population]

        while parameters.ll.N > length(population_ll)
            D = size(problem.ll.search_space[1,:])
            y = problem.ll.search_space[1,:] + (problem.ll.search_space[2,:] - problem.ll.search_space[1,:]) .* rand(D)
            c = Metaheuristics.create_child(y, f(y))
            push!(population_ll, c) 
        end
        
        method.status = Metaheuristics.State(population_ll[1], population_ll)
    end
    

    res = Metaheuristics.optimize(f, problem.ll.search_space, method)

    return res.population
    #return Metaheuristics.get_non_dominated_solutions(res.population)
end

