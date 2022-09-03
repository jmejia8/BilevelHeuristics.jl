mutable struct DBMA_LL <: Metaheuristics.AbstractParameters
    parameters::DE
    environmental_selection::NSGA2
end


function lower_level_optimizer(
        status,
        blparameters::DBMA,
        problem,
        information,
        options,
        x,
        initial_ll_sols = [],
        args...;
        kargs...
    )

    de = blparameters.ll

    # lower level function parametrized by x
    f_x(y) = Metaheuristics.evaluate(x, y, problem.ll)

    nsga2 = NSGA2(N=de.N).parameters
    parms = DBMA_LL(de, nsga2)
    ll_method = Metaheuristics.Algorithm(parms;
                                         options=options.ll,
                                         information=information.ll
                                        )


    options.ll.seed = rand(UInt)
    res = Metaheuristics.optimize(f_x, problem.ll.bounds, ll_method)

    lower_level_decision_making(status, blparameters,problem,information,options,res,args...;kargs...)

end


function update_state!(
        status,
        _parameters::DBMA_LL,
        problem::Metaheuristics.AbstractProblem,
        information::Information,
        options::Options,
        args...;
        kargs...
    )
    parameters = _parameters.parameters
    population = status.population

    new_vectors = Metaheuristics.reproduction(status, parameters, problem)

    # evaluate solutions
    new_solutions = Metaheuristics.create_solutions(new_vectors, problem,ε=options.h_tol)
    append!(status.population, new_solutions)

    Metaheuristics.environmental_selection!(status.population, _parameters.environmental_selection)
    status.best_sol = Metaheuristics.get_best(status.population)
end

initialize!(status,alg::DBMA_LL,args...;kargs...) = initialize!(status,alg.parameters,args...;kargs...)

final_stage!(status,alg::DBMA_LL,args...;kargs...) = final_stage!(status,alg.parameters,args...;kargs...)





