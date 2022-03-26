function sms_update_archive!(archive, population)

    for sol in population
        for i in eachindex(sol.ul)
            if sol.ul[i].rank == sol.lower_level_sols[i].rank == 1
                push!(archive, [sol.ul[i], sol.lower_level_sols[i]])
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


    unique!(archive)
    

end
