mutable struct NSGA2_LL <: Metaheuristics.AbstractParameters
    ll::NSGA2
    elite_set
end

function NSGA2_LL(
        nsga2 = NSGA2();
        elite_set = [],
        information = Metaheuristics.Information(),
        options = Metaheuristics.Options(),
    )

    parameters = NSGA2_LL(nsga2, elite_set)
    Metaheuristics.Algorithm(
              parameters,
              information = information,
              options = options,
             )
end


function Metaheuristics.initialize!( status, parameters::NSGA2_LL, args...; kargs...)
    Metaheuristics.initialize!(status, parameters.ll, args...; kargs...)
end

function Metaheuristics.update_state!(
    status,
    parameters_::NSGA2_LL,
    problem,
    information,
    options,
    args...;
    kargs...
    )

    elite_set = parameters_.elite_set

    if isempty(elite_set)
        error("Elite set is empty.")
    end


    parameters = parameters_.ll
    

    I = 1:parameters.N
    Q = typeof(status.population[1])[]
    for i = 1:2:parameters.N

        pa = Metaheuristics.tournament_selection(status.population, rand(I))
        pb = rand(elite_set)

        # crossover
        c1, c2 = Metaheuristics.SBX_crossover(Metaheuristics.get_position(pa), 
                                              Metaheuristics.get_position(pb),
                                              problem.bounds,parameters.η_cr,
                                              parameters.p_cr)
       
        # mutation
        Metaheuristics.polynomial_mutation!(c1,problem.bounds,parameters.η_m, parameters.p_m)
        Metaheuristics.polynomial_mutation!(c2,problem.bounds,parameters.η_m, parameters.p_m)
       
        # rapair solutions if necesary
        Metaheuristics.reset_to_violated_bounds!(c1, problem.bounds)
        Metaheuristics.reset_to_violated_bounds!(c2, problem.bounds)

        # evaluate offspring
        offspring1 = Metaheuristics.create_solution(c1, problem)

        offspring2 = Metaheuristics.create_solution(c2, problem) 
       
        # save offsprings
        push!(Q, offspring1)
        push!(Q, offspring2)
    end

    status.population = vcat(status.population, Q)

    # non-dominated sort, crowding distance, elitist removing
    Metaheuristics.truncate_population!(status.population, parameters.N)
end


function Metaheuristics.final_stage!(
    status,
    parameters_::NSGA2_LL,
    problem,
    information,
    options,
    args...;
    kargs...
    )
end
