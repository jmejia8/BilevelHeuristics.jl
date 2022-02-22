function semivectorial_ul_preferences(family, parameters)    
    sort!(family, lt = (a, b) -> BilevelHeuristics.is_better(a,b, parameters))
    Fmin = BilevelHeuristics.ulfval(family[1])
    i = 2
    while i <= length(family) && Fmin == BilevelHeuristics.ulfval(family[i])
        i += 1
    end

    # nothing to remove?
    i > length(family) && (return 1:length(family))

    return 1:i-1
end

function upper_level_decision_making(
        status::BLState{BLIndividual{U, Metaheuristics.xFgh_indiv}},
        parameters,
        problem,
        information,
        options,
        solutions, # solutions to be chosen
        args...;
        kargs...
    ) where U <: AbstractSolution # where L <: AbstractMultiObjectiveSolution

    semivectorial_ul_preferences(solutions, parameters)
end
