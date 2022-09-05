function compare(a, b, parameters::DBMA_LL)
    ϕ_a = Metaheuristics.sum_violations(a)
    ϕ_b = Metaheuristics.sum_violations(b)
    ε = parameters.parameters.ε

    # ε level comparison
    if (ϕ_a <= ε && ϕ_b <= ε) || (ϕ_a == ϕ_b)
        return Metaheuristics.compare(fval(a), fval(b))
    end

    ϕ_a < ϕ_b ? 1 : 2
end


function fast_non_dominated_sort!(pop, parameters::DBMA_LL)
    n = length(pop)

    dom_list = [ Int[] for i in 1:n ]
    rank = zeros(Int, n)
    dom_count = zeros(Int, n)

    for i in 1:n
        for j in i+1:n
            comparison = compare(pop[i], pop[j], parameters)
            if comparison == 1 #is_better(pop[i], pop[j])
                push!(dom_list[i], j)
                dom_count[j] += 1
            elseif comparison == 2 #is_better(pop[j], pop[i])
                push!(dom_list[j], i)
                dom_count[i] += 1
            end
        end
        if dom_count[i] == 0
            rank[i] = 1
        end
    end

    k = UInt16(2)
    while any(==(k-one(UInt16)), (rank[p] for p in 1:n)) #ugly workaround for #15276
        for p in 1:n
            if rank[p] == k-one(UInt16)
                for q in dom_list[p]
                    dom_count[q] -= one(UInt16)
                    if dom_count[q] == zero(UInt16)
                        rank[q] = k
                    end
                end
            end
        end
        k += one(UInt16)
    end

    return rank
end



initialize!(status,alg::DBMA_LL,problem::Metaheuristics.AbstractProblem, information::Information, options::Options, args...; kargs...) = initialize!(status,alg.parameters, problem, information, options,args...;kargs...)

final_stage!(status,alg::DBMA_LL,problem::Metaheuristics.AbstractProblem, information::Information, options::Options, args...; kargs...) = final_stage!(status,alg.parameters, problem, information, options,args...;kargs...)


function update_state!(status,
        _parameters::DBMA_LL,
        problem::Metaheuristics.AbstractProblem,
        information::Information,
        options::Options,
        args...; kargs...
    )

    parameters = _parameters.parameters

    # ε_level_control_function
    ε_0 = parameters.ε_0
    t = status.iteration
    Tc = parameters.Tc
    cp = parameters.cp
    parameters.ε = Metaheuristics.ε_level_control_function(ε_0, t, Tc, cp)

    # update_state!(status, parameters.de, args...; kargs...)
    new_vectors = Metaheuristics.reproduction(status, parameters.de, problem)

    # evaluate solutions
    new_solutions = Metaheuristics.create_solutions(new_vectors, problem,ε=options.h_tol)
    append!(status.population, new_solutions)

    # reduce population
    Metaheuristics.environmental_selection!(status.population, _parameters)

end


function Metaheuristics.environmental_selection!(population, parameters::DBMA_LL)

    fast_non_dominated_sort!(population, parameters)

    # Use crowding distance to compare solution and preserve diversity
    N = parameters.parameters.N
    let f::Int = 1
        ind = 0
        indnext = findlast(x -> x.rank == f, population)
        while !isnothing(indnext) && 0 < indnext <= N
            ind = indnext
            f += 1
            indnext = findlast(x -> x.rank == f, population)
        end
        isnothing(indnext) && (indnext = length(population)) 
        Metaheuristics.update_crowding_distance!(view(population, ind+1:indnext))
        sort!(view(population, ind+1:indnext), by = x -> x.crowding, rev = true, alg = PartialQuickSort(N-ind))
    end
    deleteat!(population, N+1:length(population))
end


