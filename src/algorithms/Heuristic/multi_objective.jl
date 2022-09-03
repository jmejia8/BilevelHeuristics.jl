# multi-objective LL case
function lower_level_decision_making(
        status,
        parameters::AbstractNested,
        problem,
        information,
        options,
        results_ll::State{T},
        args...;
        kargs...
    ) where T <: AbstractMultiObjectiveSolution

    return results_ll.population
end

function upper_level_decision_making(
        status::BLState{BLIndividual{T,T}},
        parameters,
        problem,
        information,
        options,
        solutions,
        args...;
        kargs...
    ) where T <: AbstractMultiObjectiveSolution

    population_ul = get_ul_population(solutions)
    Metaheuristics.get_non_dominated_solutions_perm(population_ul)
end

function upper_level_decision_making(
        status::BLState{BLIndividual{U,Union{Metaheuristics.xf_indiv, Metaheuristics.xfgh_indiv}}},
        parameters::AbstractNested,
        problem,
        information,
        options,
        solutions,
        args...;
        kargs...
    ) where U <: AbstractMultiObjectiveSolution # where L <: AbstractSolution

    population_ul = get_ul_population(solutions)
    Metaheuristics.get_non_dominated_solutions_perm(population_ul)
end


function truncate_population!(
        status::BLState{BLIndividual{U, L}},
        parameters::AbstractNested,
        problem,
        information,
        options
    ) where U <: AbstractMultiObjectiveSolution where L <: AbstractSolution

    length(status.population) <= parameters.ul.N && (return) 

    population_ul = get_ul_population(status.population)
    Metaheuristics.environmental_selection!(population_ul, parameters.ul)

    # TODO improve performance this part
    delete_mask = ones(Bool, length(status.population))
    for sol in get_ul_population(status.population)
        i = findfirst( s -> s==sol, population_ul)
        isnothing(i) && continue
        delete_mask[i] = false
    end

    deleteat!(status.population, delete_mask)    
end
