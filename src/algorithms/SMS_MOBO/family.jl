mutable struct Family{T} <: Metaheuristics.AbstractSolution
    ul::Vector{T} # upper level solution (x, F, G, etc)
    # lower level solutions (y, f, g, etc), i.e., array of lower level solutions
    lower_level_sols::Vector{T}

    super_rank::Int
    contribution::Float64
end


function create_family(x, lower_level_sols, problem)
    ul = Metaheuristics.xFgh_indiv[]

    for sol in lower_level_sols
        y = Metaheuristics.get_position(sol)
        sol_ul = Metaheuristics.create_child(x, problem.ul.f(x, y))
        problem.ul.f_calls += 1
        push!(ul, sol_ul)
    end 

    # In this step, it is not possible to compute the super rank and the contribution
    super_rank = 0
    contribution = 0.0

    # only considering non dominated solutions
    #rank_1 = Metaheuristics.get_non_dominated_solutions_perm(ul)

    #return Family(ul[rank_1], lower_level_sols[rank_1], super_rank, contribution)
    return Family(ul, lower_level_sols, super_rank, contribution)
end


function update_super_rank!(population)
    # those are pointers to families
    population_ul = [ s for family in population for s in family.ul ]

    # update rank of ul population
    Metaheuristics.fast_non_dominated_sort!(population_ul)

    for family in population
        family.super_rank = minimum(s -> s.rank, family.ul)
        mask = map(s -> s.rank == family.super_rank, family.ul)

        # remove solution dominated for other ones in same super rank
        family.ul =  family.ul[mask]
        family.lower_level_sols =  family.lower_level_sols[mask]
    end

end

function show_optim_info(io::IO, status::BLState{Family{Metaheuristics.xFgh_indiv}})
    population = get_ul_population(status.population)
    isempty(population) && (return)
    

    @printf(io, "%12s", "population:\n")
    show(io, "text/plain", Array(population))

    # non-dominated
    pf = Metaheuristics.get_non_dominated_solutions(population)
    println(io, "\nnon-dominated solution(s):")
    show(io, "text/plain", pf)
    print(io, "\n")
end

