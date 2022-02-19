# multi-objective case
function lower_level_decision_making(
        status,
        parameters::Heuristic,
        problem,
        information,
        options,
        results_ll::State{T},
        args...;
        kargs...
    ) where T <: Metaheuristics.AbstractMultiObjectiveSolution

    return results_ll.population
end

function upper_level_decision_making(
        status::BLState{BLIndividual{U, L}},
        parameters,
        problem,
        information,
        options,
        solutions,
        args...;
        kargs...
    ) where U <: Metaheuristics.AbstractMultiObjectiveSolution where L <: AbstractSolution

    eachindex(solutions)
end

function truncate_population!(
        status::BLState{BLIndividual{U, L}},
        parameters::Heuristic,
        problem,
        information,
        options
    ) where U <: Metaheuristics.AbstractMultiObjectiveSolution where L <: AbstractSolution

    population_ul = get_ul_population(status.population)
    Metaheuristics.environmental_selection!(population_ul, parameters.ul)

    # TODO improve performance this part
    mask = Int[]
    for sol in get_ul_population(status.population)
        i = findfirst( s -> s==sol, population_ul)
        isnothing(i) && continue
        push!(mask, i)
    end

    deleteat!(status.population, mask)    
end
