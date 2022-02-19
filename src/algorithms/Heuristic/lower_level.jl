# single-objective case
function lower_level_decision_making(
        status,
        parameters::Heuristic,
        problem,
        information,
        options,
        results_ll::State,
        args...;
        kargs...
    ) 

    return handle_ll_multimodality(results_ll, problem, options) 
end

function lower_level_optimizer(
        status,
        parameters::Heuristic,
        problem,
        information,
        options,
        x,
        initial_ll_sols = [],
        args...;
        kargs...
    )

    # lower level function parametrized by x
    f_x(y) = Metaheuristics.evaluate(x, y, problem.ll)
    
    # changing seed is important
    options.ll.seed = rand(UInt)
    method = Metaheuristics.Algorithm(parameters.ll;
                                      options=options.ll,
                                      information=information.ll
                                     )

    if !isempty(initial_ll_sols)
        Y = initial_ll_sols
        @assert parameters.ll.N == size(Y,1)
        population_ll = [Metaheuristics.create_child(y, f(Y[i,:])) for i in 1:size(Y,1)]
        method.status = Metaheuristics.State(population_ll[1], population_ll)
    end

    res = Metaheuristics.optimize(f_x, problem.ll.bounds, method)

    return lower_level_decision_making(status, parameters,problem,information,options,res,args...;kargs...)
end
