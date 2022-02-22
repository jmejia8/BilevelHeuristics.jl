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

# MOBO
function show_optim_info(
        io::IO,
        status::BLState{BLIndividual{T,T}}
    ) where T <: AbstractMultiObjectiveSolution

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

# MOBO with single-objective lower level
function show_optim_info(
        io::IO,
        status::BLState{BLIndividual{U,L}}
    ) where U <: AbstractMultiObjectiveSolution where L <: AbstractSolution

    population = get_ul_population(status.population)
    isempty(population) && (return)
    

    # @printf(io, "%12s", "population:\n")
    # show(io, "text/plain", Array(population))

    # non-dominated
    pf_mask = Metaheuristics.get_non_dominated_solutions_perm(population)
    pf = population[pf_mask]
    println(io, "Non-dominated solution(s):")
    show(io, "text/plain", pf)
    print(io, "\n")
    
    population_ll = get_ll_population(status.population[pf_mask])
    mask = sortperm(fvals(pf)[:,1])
    fvals_ll = fvals(population_ll)[mask]

    plt = scatterplot(eachindex(fvals_ll),
                      fvals_ll,
                      ylabel="F ",
                      xlabel="Num. of solution",
                      title="Lower Level",
                      border=:dotted)
    show(io, plt)
    println(io, "")

end

# SVBO
function show_optim_info(
        io::IO,
        status::BLState{BLIndividual{U,L}}
    ) where U <: AbstractSolution where L <: AbstractMultiObjectiveSolution

    if isnothing(status.best_sol)
        return
    end
    Fxy, fxy = minimum(status)
    @printf(io,"%12s \n", "minimum:")
    @printf(io,"%12s %g\n", "F:", Fxy)
    @printf(io,"%12s ", "f:")
    println(io, fxy) 

    x, y = minimizer(status)
    @printf(io,"%12s \n", "minimizer:")
    @printf(io,"%12s ", "x:")
    show(io, x)
    @printf(io,"\n%12s ", "y:")
    show(io, y)
    println(io, "")

end

# SOBO
function show_optim_info(
        io::IO,
        status::BLState{BLIndividual{U,L}}
    ) where U <: AbstractSolution where L <: AbstractSolution

    if isnothing(status.best_sol)
        return
    end
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

function Base.show(io::IO, status::BLState)
    println(io, "+=========== RESULT ==========+")
    @printf(io,"%12s %.0f\n", "iteration:", status.iteration)

    show_optim_info(io, status)



    @printf(io,"%12s %.0f\n", "F calls:", status.F_calls)
    @printf(io,"%12s %.0f\n", "f calls:", status.f_calls)
    @printf(io,"%12s ", "Message:")
    println(io, status.stop_msg)
    @printf(io,"%12s %.4f s\n", "total time:", status.final_time - status.start_time)
    println(io, "+============================+")
end

function Base.show(io::IO, ::MIME"text/html", status::BLState)
    println(io, "<pre>")
    show(io, "text/plain", status)
    println(io, "</pre>")
end
