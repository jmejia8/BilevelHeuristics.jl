mutable struct BLIndividual{U, L} <: AbstractSolution
    ul::U
    ll::L
end

function Base.show(io::IO, indiv::BLIndividual)
    println(io, "Upper-level:")
    Base.show(io, indiv.ul)
    println(io, "\nLower-level:")
    Base.show(io, indiv.ll) 
end

function create_solution(x, sol_ll, problem)
    y = Metaheuristics.get_position(sol_ll)
    sol_ul = Metaheuristics.create_child(x, problem.ul.f(x, y))
    problem.ul.f_calls += 1
    return BLIndividual(sol_ul, sol_ll)
end

"""
    get_ul_population(population)

Return the upper level solutions.
"""
get_ul_population(pop::Vector) = [sol.ul for sol in pop]
"""
    get_ll_population(population)

Return the lower level solutions.
"""
get_ll_population(pop::Vector) = [sol.ll for sol in pop]

"""
    ulvector(A)

Get upper-level decision vector.
"""
ulvector(A::AbstractSolution) = Metaheuristics.get_position(A.ul)

"""
    llvector(A)

Get lower-level decision vector.
"""
llvector(A::AbstractSolution)  = Metaheuristics.get_position(A.ll)

"""
    ulfval(A)

Get upper-level function value.
"""
ulfval(A::AbstractSolution) = Metaheuristics.fval(A.ul)

"""
    llfval(A)

Get lower-level function value.
"""
llfval(A::AbstractSolution) = Metaheuristics.fval(A.ll)

leader_pos(A::AbstractSolution  ) = ulvector(A)
follower_pos(A::AbstractSolution) = llvector(A)

leader_f(A::AbstractSolution  ) = ulfval(A)
follower_f(A::AbstractSolution) = llfval(A)

"""
    ulpositions(population)

Get upper-level decision vectors from population.
"""
ulpositions(pop::Vector) = Metaheuristics.positions(get_ul_population(pop))

"""
    llpositions(population)

Get lower-level decision vectors from population.
"""
llpositions(pop::Vector) = Metaheuristics.positions(get_ll_population(pop))

"""
    ulfvals(pop)

Get upper-level function values from population.
"""
ulfvals(pop::AbstractVector) = Metaheuristics.fvals(get_ul_population(pop))

"""
    llfvals(pop)

Get lower-level function values from population.
"""
llfvals(pop::AbstractVector) = Metaheuristics.fvals(get_ll_population(pop))


"""
    ulgvals(pop)

Get upper-level inequality constraints.
"""
ulgvals(pop::AbstractVector) = Metaheuristics.gvals(get_ul_population(pop))

"""
    llgvals(pop)

Get lower-level inequality constraints.
"""
llgvals(pop::AbstractVector) = Metaheuristics.gvals(get_ll_population(pop))

"""
    ulhvals(pop)

Get upper-level equality constraints.
"""
ulhvals(pop::AbstractVector) = Metaheuristics.hvals(get_ul_population(pop))

"""
    llhvals(pop)

Get lower-level equality constraints.
"""
llhvals(pop::AbstractVector) = Metaheuristics.hvals(get_ll_population(pop))

"""
    is_pseudo_feasible(A, B, δ1, δ2, ε1, ε2)

Check whether `A` is a pseudo-feasible solution respect to `B`.
"""
function is_pseudo_feasible(A::BLIndividual, B::BLIndividual, δ1, δ2, ε1, ε2)
    ΔF = abs(leader_f(A) - leader_f(B))
    Δf = abs(follower_f(A) - follower_f(B))

    Δx = leader_pos(A) - leader_pos(B)
    Δy = follower_pos(A) - follower_pos(B)

    if Δf < ε2 && ΔF < ε1 && norm(Δx) < δ1 && norm(Δy) > δ2
        return true
    end

    false
end

is_better_naive(A::BLIndividual, B::BLIndividual) = leader_f(A) < leader_f(B)

function Metaheuristics.is_better(A::BLIndividual, B::BLIndividual)
    A_vio = Metaheuristics.sum_violations(A)
    B_vio = Metaheuristics.sum_violations(B)

    if A_vio < B_vio
        return true
    elseif B_vio < A_vio
        return false
    end

    Metaheuristics.is_better(A.ul, B.ul)
end


function Metaheuristics.sum_violations(A::BLIndividual{T,T}) where T <: Metaheuristics.AbstractUnconstrainedSolution
    0
end

function Metaheuristics.sum_violations(A::BLIndividual) 
    vio_ul = A.ul isa Metaheuristics.AbstractUnconstrainedSolution ? 0 : Metaheuristics.sum_violations(A.ul)
    vio_ll = A.ll isa Metaheuristics.AbstractUnconstrainedSolution ? 0 : Metaheuristics.sum_violations(A.ll)
    # FIXME: Consider LL to compute this?
    vio_ul + vio_ll
end

function Metaheuristics.is_feasible(A::BLIndividual)
    Metaheuristics.is_feasible(A.ul) && Metaheuristics.is_feasible(A.ll)
end

