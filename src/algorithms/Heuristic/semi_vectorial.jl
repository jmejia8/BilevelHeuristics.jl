function upper_level_decision_making(
        status::BLState{BLIndividual{U, L}},
        parameters,
        problem,
        information,
        options,
        solutions,
        args...;
        kargs...
    ) where U <: AbstractSolution where L <: Metaheuristics.AbstractMultiObjectiveSolution

    eachindex(solutions)
end
