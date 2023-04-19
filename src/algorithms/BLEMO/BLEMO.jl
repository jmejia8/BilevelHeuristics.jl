abstract type AbstractBLEMO end # for inheritance purposes

mutable struct Subpopulation{T}
    x::Vector{Float64}
    subpopulation::Vector{T}
end

mutable struct BLEMO <: AbstractBLEMO
    ul::NSGA2
    ll::NSGA2
    archive # archive
    subpopulations::Vector{Subpopulation}
end


mutable struct BLEMOIndividual{T} <: Metaheuristics.AbstractSolution
    ul::T
    ll::T

    subpopulation
end

#=
mutable struct BLEMOIndividual{T} <: Metaheuristics.AbstractSolution
    ul::Vector{T} # upper level solution (x, F, G, etc)

    # lower level solutions (y, f, g, etc), i.e., array of lower level solutions
    subpopulation::Vector{T}
end
=#


function create_BLEMO_solutions(x, subpopulation, subpopulation_id, problem)
    solutions = BLEMOIndividual[]

    for sol_ll in subpopulation.subpopulation
        y = Metaheuristics.get_position(sol_ll)
        sol_ul = Metaheuristics.create_child(x, problem.ul.f(x, y))
        problem.ul.f_calls += 1

        push!(solutions, BLEMOIndividual(sol_ul, sol_ll, subpopulation))
    end 

    solutions
end

function BLEMO(;
        ul = NSGA2(),
        ll = NSGA2(),
        options_ul = Metaheuristics.Options(),
        options_ll = Metaheuristics.Options(),
        information_ul = Metaheuristics.Information(),
        information_ll = Metaheuristics.Information()
    )

    parameters = BLEMO(ul.parameters, ll.parameters, [], [])

    return Algorithm(parameters;
                     options = BLOptions(options_ul, options_ll),
                     information = BLInformation(information_ul, information_ll)
                    )
end

get_ul_population(pop::Vector{BLEMOIndividual}) = [sol.ul for sol in pop]
get_ll_population(pop::Vector{BLEMOIndividual{Metaheuristics.xFgh_indiv}}) = [sol.ll for sol in pop]

include("lower-level.jl")


function initialize!(
        status, # an initialized State (if apply)
        parameters::AbstractBLEMO,
        problem,
        information,
        options,
        args...;
        kargs...
    )

    a = problem.ul.search_space.lb # lower bounds (UL)
    b = problem.ul.search_space.ub # upper bounds (UL)
    D = Metaheuristics.getdim(problem.ul.search_space)
    D_ll = Metaheuristics.getdim(problem.ll.search_space)

    N_ul = parameters.ul.N
    N_ll = parameters.ll.N

    N_subpop = floor(Int, N_ul / N_ll )


    population = BLEMOIndividual{Metaheuristics.xFgh_indiv}[]
    parameters.subpopulations = []

    for i in 1:N_subpop
        rn = rand(D)
        x = a + (b - a) .* rn
        

        sols_ll = lower_level_optimizer(status, parameters, problem, information, options, x)
        subpopulation = Subpopulation(x, sols_ll)
        push!(parameters.subpopulations, subpopulation)
        subpopulation_id = length(parameters.subpopulations)

        population = vcat(population, create_BLEMO_solutions(x, subpopulation, subpopulation_id, problem))
        #push!(population, create_BLEMO_solution(x, subpopulation, problem))
    end

    return BLState(population[1], population) # replace this

end


function blemo_genetic_operators(population, parameters, problem; selection=:tournament)
    i = rand(1:length(population))
    j = rand(1:length(population))

    while j == i
        j = rand(1:length(population))
    end
    
    if selection == :tournament
        pa = Metaheuristics.tournament_selection(population, i)
        pb = Metaheuristics.tournament_selection(population, j)
    else
        pa = population[i]
        pb = population[j]
    end
    

    # crossover
    c1, c2 = Metaheuristics.SBX_crossover(Metaheuristics.get_position(pa),
                                          Metaheuristics.get_position(pb),
                                          problem.search_space,
                                          parameters.η_cr,
                                          parameters.p_cr)

    # choosing one at random
    c = rand([c1, c2])

    # mutation
    Metaheuristics.polynomial_mutation!(c ,problem.search_space,parameters.η_m, parameters.p_m)

    # repair offprings if necessary
    Metaheuristics.reset_to_violated_bounds!(c , problem.search_space)

    return c
end

function update_rank_crowding!(population)
    # compute non-dominated sorting and crowding distance
    Metaheuristics.fast_non_dominated_sort!(population)
    ranks = [s.rank for s in population]

    # bad ranks due NaNs
    if ranks[1] < 1
        @warn "Some ranks are disrupted. Some NaNs can be promoting this issue."
        for i in findall(r -> r < 1, ranks)
            population[i].rank = ranks[end]+1
        end
        sort!(population, by = s -> s.rank)
        ranks = [s.rank for s in population]
        @info "Mitigation done."
    end
 
    fronts = [Int[] for i in eachindex(unique(ranks))]

    for (i, r) in enumerate(ranks)
        push!(fronts[r], i)
    end

    for front in fronts
        Metaheuristics.update_crowding_distance!(population[front])
    end

end


function update_state!(
        status,
        parameters::AbstractBLEMO,
        problem,
        information,
        options,
        args...;
        kargs...
    )

    # Get elite set
    all_ul_solutions = [[sol.ul, sol.ll] for sol in status.population]
    best_rank = minimum(s -> s[1].rank, all_ul_solutions) # = 1 ?
    mask = findall(s -> s[1].rank == best_rank, all_ul_solutions)
    elite_set = all_ul_solutions[mask]

    # population at lower level
    population_ll = vcat([s for subpop in parameters.subpopulations for s in subpop.subpopulation]...)


    N_ul = parameters.ul.N
    N_ll = parameters.ll.N

    N_subpop = floor(Int, N_ul / N_ll )

    # offprings for the bilevel problem
    #Q = typeof(status.population[1])[]
    #Q_ul = Vector[]
    Q = BLEMOIndividual[]


    population_ul = [s[1]  for s in all_ul_solutions]
    Q_ul = []
    # let's perform the matings at upper level
    for i = 1:N_subpop
        c = blemo_genetic_operators(population_ul, parameters.ul, problem.ul)
        push!(Q_ul, c)
    end

    new_subpopulations = Subpopulation[]
    for x in Q_ul
        Q_ll = Vector[] # sub population

        # let's perform the matings at lower level
        for i = 1:N_ll
            c = blemo_genetic_operators(population_ll, parameters.ll, problem.ll)

            # save children
            push!(Q_ll, c)
        end

        # step 2: optimize using subpopulations
        sols_ll = lower_level_optimizer(status,
                                              parameters,
                                              problem,
                                              information,
                                              options,
                                              x,
                                              Q_ll,
                                              elite_set
                                             )

        subpopulation = Subpopulation(x, sols_ll)
        push!(new_subpopulations, subpopulation)

        new_solutions = create_BLEMO_solutions(x, subpopulation, 0, problem)


        Q = vcat(Q, new_solutions)

    end

    R = vcat(status.population, Q)


    # only save the N best sols
    # according to the rank with different subpop and best crowding dist 
    truncate_population!(R, parameters)
    status.population = R



    # step 5
    for (i, subpopulation) in enumerate(parameters.subpopulations)
        x = subpopulation.x 

        # each y in subpopulation
        subpopulation_vars = [Metaheuristics.get_position(s) for s in subpopulation.subpopulation]


        # perform NSGA-II again
        subpopulation = lower_level_optimizer(status,
                                              parameters, 
                                              problem,
                                              information,
                                              options,
                                              x,
                                              subpopulation_vars)

        parameters.subpopulations[i].subpopulation = subpopulation
    end

    BLEMO_update_archive!(parameters.archive, status.population)

    if options.ul.debug
        @info "Archive:"
        println("Size: ", length(parameters.archive))
    end


end


function update_archive!(archive, population)

    for sol in population
        for i in eachindex(sol.ul)
            if sol.ul[i].rank == sol.subpopulation[i].rank == 1
                push!(archive, [sol.ul[i], sol.subpopulation[i]])
            end
        end
    end

    if isempty(archive)
        return
    end

    mask = Metaheuristics.get_non_dominated_solutions_perm(map(s -> s[1], archive))
    dominated = ones(Bool, length(archive))
    dominated[mask] .= false 
    deleteat!(archive, dominated)
    

end

function BLEMO_update_archive!(archive, population)

    for sol in population
        if sol.ul.rank == sol.ll.rank == 1
            push!(archive, [sol.ul, sol.ll])
        end
    end

    if isempty(archive)
        return
    end

    mask = Metaheuristics.get_non_dominated_solutions_perm(map(s -> s[1], archive))
    dominated = ones(Bool, length(archive))
    dominated[mask] .= false 

    deleteat!(archive, dominated)
    unique!(archive)


    na = length(archive)

    # FIXME
    archive_limit = 5000
    # limit archive
    if na > archive_limit
        I = randperm(length(archive))
        deleteat!(archive, sort(unique(I[1:na-archive_limit ])))
    end

end



function truncate_population!(population, parameters::AbstractBLEMO)
    N = parameters.ul.N
    
    # non-dominated sort, crowding distance, elitist removing
    BLEMO_update_rank_and_crowding_distance!(population)

    ranking_rule(a_, b_, a = a_.ul, b = b_.ul) = a.rank < b.rank || (a.rank == b.rank && a.crowding > b.crowding)

    sort!(population, lt = ranking_rule, alg=PartialQuickSort(N))

    ns =  length(parameters.subpopulations)
    
    # update subpopulations
    empty!(parameters.subpopulations)

    for sol in population
        subpopulation_id = findfirst(sp -> sp === sol.subpopulation, parameters.subpopulations)
        
        if isnothing(subpopulation_id)
            push!(parameters.subpopulations, sol.subpopulation)
            subpopulation_id = length(parameters.subpopulations)
        end

        # sol.subpopulation_id = subpopulation_id

        length(parameters.subpopulations) == ns && break
        
    end

    deleteat!(population, N+1:length(population))

end

function BLEMO_update_rank_and_crowding_distance!(population)
    population_ul = get_ul_population(population)
    update_rank_crowding!(population_ul)
end


function stop_criteria!(status, parameters::AbstractBLEMO, problem, information, options)
    return
end


function final_stage!(
        status,
        parameters::AbstractBLEMO,
        problem,
        information,
        options,
        args...;
        kargs...
    )

    #=
    options.ul.debug && @info "Combining final population and archive..."
    
    mask = findall(s -> s.ul.rank == 1 && s.ll.rank == 1, status.population)

    pop = [[s.ul, s.ll] for s in status.population]
    parameters.archive = vcat(parameters.archive, pop)
    =#

    return
end

