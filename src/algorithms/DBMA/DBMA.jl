mutable struct DBMA <: AbstractNested
    ul::εDE
    ll::εDE
end

include("lower_level.jl")

function Base.getproperty(obj::εDE, sym::Symbol)
    if hasproperty(obj, sym)
        return getfield(obj, sym)
    end

    getfield(obj.de, sym)
end

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

    de_ul = εDE(;N = Nu, F, CR)
    de_ll = εDE(;N = Nl, F, CR)
    de_ul.parameters.N = Nu

    parameters = DBMA(de_ul.parameters, de_ll.parameters)


    Algorithm(parameters;
              options     = BLOptions(options_ul, options_ll),
              information = BLInformation(information_ul, information_ll)
             )
end


function _ϵ_control!(status, blparameters::DBMA, bloptions)
    population = get_ul_population(status.population)
    parameters = blparameters.ul
    options = bloptions.ul

    ε_0 = parameters.ε_0
    t  = status.iteration
    Tc = parameters.Tc
    cp = parameters.cp

    if status.iteration > 1
        parameters.ε = Metaheuristics.ε_level_control_function(ε_0, t, Tc, cp)
        return
    end

    elite_sols = sortperm(population, lt = is_better)

    θ = round(Int, 0.2length(elite_sols))
    s = rand(population[elite_sols[1:θ]])
    
    parameters.ε_0 = Metaheuristics.sum_violations(s)
    parameters.Tc = round(Int, 0.2*options.iterations)
    parameters.N = parameters.de.N
    p = parameters
    parameters.ε = Metaheuristics.ε_level_control_function(p.ε_0, t, p.Tc, p.cp)

end


function truncate_population!(
        status::BLState{BLIndividual{U, L}},
        parameters::DBMA,
        problem,
        information,
        options
    ) where U <: AbstractMultiObjectiveSolution where L <: AbstractSolution 


    length(status.population) <= parameters.ul.N && (return) 
    _ϵ_control!(status, parameters, options)

    population_ul = get_ul_population(status.population)

    dde = DBMA_LL(parameters.ul)
    Metaheuristics.environmental_selection!(population_ul, dde)
 

    # TODO improve performance this part
    delete_mask = ones(Bool, length(status.population))
    for (j, sol) in enumerate(get_ul_population(status.population))
        i = findfirst( s -> s==sol, population_ul)
        isnothing(i) && continue
        delete_mask[j] = false
    end

    deleteat!(status.population, delete_mask)    
end


function upper_level_decision_making(
        status::BLState{BLIndividual{T,T}},
        parameters::DBMA,
        problem,
        information,
        options,
        solutions,
        args...;
        kargs...
    ) where T <: AbstractMultiObjectiveSolution

    eachindex(get_ul_population(solutions))
end

