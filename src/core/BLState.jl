mutable struct BLState{T}
    best_sol::T
    population::Vector{T}
    F_calls::Int
    f_calls::Int

    iteration::Int
    convergence::Array{BLState}
    start_time::Float64
    final_time::Float64
    overall_time::Float64
    stop::Bool
    stop_msg::String
end


function BLState(best_sol, population)
    BLState(best_sol, population, 0,0,1, BLState[], 0.0,0.0,0.0,false, "")
end


minimum(status::BLState) = (leader_f(status.best_sol), follower_f(status.best_sol))
minimizer(status::BLState) = (leader_pos(status.best_sol), follower_pos(status.best_sol))


function Base.show(io::IO, status::BLState)
    println(io, "+=========== RESULT ==========+")
    @printf(io,"%12s %.0f\n", "iteration:", status.iteration)

    population = get_ul_population(status.population)
    # population = [ sol.ul for sol in status.population]

    if typeof(Array(population)) <: Array{Metaheuristics.xFgh_indiv}
        @printf(io, "%12s", "population:")
        show(io, "text/plain", Array(population))

        # non-dominated
        pf = Metaheuristics.get_non_dominated_solutions(population)
        println(io, "\nnon-dominated solution(s):")
        show(io, "text/plain", pf)
        print(io, "\n")
    elseif !isnothing(status.best_sol)
        Fxy, fxy = minimum(status)
        @printf(io,"%12s \n", "minimum:")
        @printf(io,"%12s %g\n", "F:", Fxy)
        @printf(io,"%12s %g\n", "f:", fxy)

        x, y = minimizer(status)
        @printf(io,"%12s \n", "minimizer:")
        @printf(io,"%12s ", "x:")
        show(io, x)
        @printf(io,"\n%12s ", "y:")
        show(io, y)
        println(io, "")
    end



    @printf(io,"%12s %.0f\n", "F calls:", status.F_calls)
    @printf(io,"%12s %.0f\n", "f calls:", status.f_calls)
    @printf(io,"%12s ", "Message:")
    println(io, status.stop_msg)
    #=
    if !isempty(status.population) &&  typeof(status.population[1]) <: Union{Metaheuristics.xfgh_indiv, Metaheuristics.xFgh_indiv}
        n = sum(map(s -> s.sum_violations ≈ 0, status.population))
        @printf(io,"%12s %d / %d in final population\n", "feasibles:", n, length(status.population))
    end
    =#
    @printf(io,"%12s %.4f s\n", "total time:", status.final_time - status.start_time)
    println(io, "+============================+")
end
