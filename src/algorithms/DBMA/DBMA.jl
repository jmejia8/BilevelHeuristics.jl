mutable struct DBMA <: AbstractNested
    ul::DE
    ll::DE
end

include("lower_level.jl")

function DBMA(;
        Nu = 50,
        Nl = 50,
        F  = 0.7,
        CR = 0.5,
        options_ul = Metaheuristics.Options(),
        options_ll = Metaheuristics.Options(),
        information_ul = Metaheuristics.Information(),
        information_ll = Metaheuristics.Information()
    )

    de_ul = DE(;N = Nu, F, CR)
    de_ll = DE(;N = Nl, F, CR)

    parameters = DBMA(de_ul.parameters, de_ll.parameters)

    Algorithm(parameters;
              options     = BLOptions(options_ul, options_ll),
              information = BLInformation(information_ul, information_ll)
             )
end



function truncate_population!(
        status::BLState{BLIndividual{U, L}},
        parameters::DBMA,
        problem,
        information,
        options
    ) where U <: AbstractMultiObjectiveSolution where L <: AbstractSolution

    length(status.population) <= parameters.ul.N && (return) 

    population_ul = get_ul_population(status.population)

    # environmental_selection for NSGA2
    nsga2 = NSGA2(N=parameters.ul.N).parameters
    Metaheuristics.environmental_selection!(population_ul, nsga2)

    # TODO improve performance this part
    delete_mask = ones(Bool, length(status.population))
    for sol in get_ul_population(status.population)
        i = findfirst( s -> s==sol, population_ul)
        isnothing(i) && continue
        delete_mask[i] = false
    end

    deleteat!(status.population, delete_mask)    
end

