"""
    BLIndividual{U, L} <: AbstractSolution

A bilevel solution coupling an upper-level individual (`U`) with its associated
lower-level individual (`L`).  Both `U` and `L` are solution types from
`Metaheuristics.jl` (e.g. `xFgh_indiv` for constrained problems).

## Fields
- `ul` — upper-level solution (decision vector `x`, objective `F`, constraints `G`, `H`).
- `ll` — lower-level solution (decision vector `y`, objective `f`, constraints `g`, `h`).

Use convenience accessors ([`ulvector`](@ref), [`llvector`](@ref), [`ulfval`](@ref),
[`llfval`](@ref), etc.) to extract components.
"""
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

Return the upper-level solutions from a population of [`BLIndividual`](@ref)s.
"""
get_ul_population(pop::Vector) = [sol.ul for sol in pop]
"""
    get_ll_population(population)

Return the lower-level solutions from a population of [`BLIndividual`](@ref)s.
"""
get_ll_population(pop::Vector) = [sol.ll for sol in pop]

"""
    ulvector(A)

Extract the upper-level decision vector `x` from a [`BLIndividual`](@ref).
"""
ulvector(A::AbstractSolution) = Metaheuristics.get_position(A.ul)

"""
    llvector(A)

Extract the lower-level decision vector `y` from a [`BLIndividual`](@ref).
"""
llvector(A::AbstractSolution)  = Metaheuristics.get_position(A.ll)

"""
    ulfval(A)

Extract the upper-level objective value `F(x, y)` from a [`BLIndividual`](@ref).
"""
ulfval(A::AbstractSolution) = Metaheuristics.fval(A.ul)

"""
    llfval(A)

Extract the lower-level objective value `f(x, y)` from a [`BLIndividual`](@ref).
"""
llfval(A::AbstractSolution) = Metaheuristics.fval(A.ll)

"""
    leader_pos(A)
    follower_pos(A)

Aliases for [`ulvector`](@ref) and [`llvector`](@ref), respectively.
"""
leader_pos(A::AbstractSolution  ) = ulvector(A)
follower_pos(A::AbstractSolution) = llvector(A)

"""
    leader_f(A)
    follower_f(A)

Aliases for [`ulfval`](@ref) and [`llfval`](@ref), respectively.
"""
leader_f(A::AbstractSolution  ) = ulfval(A)
follower_f(A::AbstractSolution) = llfval(A)

"""
    ulpositions(population)

Extract the upper-level decision vectors from every solution in `population`.
Returns an `N × D_ul` matrix.
"""
ulpositions(pop::Vector) = Metaheuristics.positions(get_ul_population(pop))

"""
    llpositions(population)

Extract the lower-level decision vectors from every solution in `population`.
Returns an `N × D_ll` matrix.
"""
llpositions(pop::Vector) = Metaheuristics.positions(get_ll_population(pop))

"""
    ulfvals(pop)

Extract all upper-level objective values from `population`.
Returns a vector of length `N` (single-objective) or an `N × M` matrix (multi-objective).
"""
ulfvals(pop::AbstractVector) = Metaheuristics.fvals(get_ul_population(pop))

"""
    llfvals(pop)

Extract all lower-level objective values from `population`.
Returns a vector of length `N` (single-objective) or an `N × M` matrix (multi-objective).
"""
llfvals(pop::AbstractVector) = Metaheuristics.fvals(get_ll_population(pop))


"""
    ulgvals(pop)

Extract upper-level inequality constraint values `G(x, y)` for each solution in `population`.
Returns an `N × n_ineq` matrix.
"""
ulgvals(pop::AbstractVector) = Metaheuristics.gvals(get_ul_population(pop))

"""
    llgvals(pop)

Extract lower-level inequality constraint values `g(x, y)` for each solution in `population`.
Returns an `N × n_ineq` matrix.
"""
llgvals(pop::AbstractVector) = Metaheuristics.gvals(get_ll_population(pop))

"""
    ulhvals(pop)

Extract upper-level equality constraint values `H(x, y)` for each solution in `population`.
Returns an `N × n_eq` matrix.
"""
ulhvals(pop::AbstractVector) = Metaheuristics.hvals(get_ul_population(pop))

"""
    llhvals(pop)

Extract lower-level equality constraint values `h(x, y)` for each solution in `population`.
Returns an `N × n_eq` matrix.
"""
llhvals(pop::AbstractVector) = Metaheuristics.hvals(get_ll_population(pop))

"""
    is_pseudo_feasible(A, B, δ1, δ2, ε1, ε2)

Check whether `A` is a *pseudo-feasible* solution with respect to `B`.

A solution `(x_A, y_A)` is pseudo-feasible relative to `(x_B, y_B)` when:
- The upper-level objectives are close: `|F(A) - F(B)| < ε1`.
- The lower-level objectives are close: `|f(A) - f(B)| < ε2`.
- The upper-level positions are close: `|x_A - x_B| < δ1`.
- The lower-level positions are *distant*: `|y_A - y_B| > δ2`.

This indicates that the two solutions have similar objective values at both levels but
different lower-level responses — a sign of multi-modality at the lower level. Such
solutions can mislead the search, and algorithms like [`QBCA2`](@ref) explicitly avoid
them.
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

