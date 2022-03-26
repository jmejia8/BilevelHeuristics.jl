function remap_ΔH!(ΔS)    
    mask = isfinite.(ΔS)
    if !isempty(ΔS[mask])
        ΔS[.!mask] .= 2maximum(ΔS[mask]) 
    end
end

function update_contribution!(population::AbstractArray{Family{T}}, last_front, n_samples) where T <: AbstractSolution
    for sol in population[1:last_front[1]]
        # elite sols. have the largest contribution
        sol.contribution = 1/eps() # Inf
    end


    for sol in population[last_front[1]: end]
        sol.contribution = 0
    end
 
    # computing contribution for elite individual in families
    R = [ [s for s in sol.ul ] for sol in population[last_front] ]
    population_ul = [s for B in R for s in B]

    if length(population_ul) <= 2
        # unable computing hypervolume only on extrema
        # therefore ΔS[last_front[1]:end] = 0
        return
    end

    # compute contribution
    Metaheuristics.update_contribution!(population_ul, 1:length(population_ul), n_samples)

    for (i, sol) in enumerate(population[last_front])
        ΔS_ul = Metaheuristics.get_contribution.(R[i])
        # replace Inf values for maximum ΔS finite value
        remap_ΔH!(ΔS_ul)

        # combinated contribution
        sol.contribution = sum(ΔS_ul)
    end

end

