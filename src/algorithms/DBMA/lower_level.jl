mutable struct DBMA_LL <: Metaheuristics.AbstractDifferentialEvolution
    parameters::εDE
end

include("utils.jl")

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

    parms = DBMA_LL(de)
    ll_method = Metaheuristics.Algorithm(parms;
                                         options=options.ll,
                                         information=information.ll
                                        )


    options.ll.seed = rand(UInt)
    res = Metaheuristics.optimize(f_x, problem.ll.bounds, ll_method)

    lower_level_decision_making(status, blparameters,problem,information,options,res,args...;kargs...)

end

