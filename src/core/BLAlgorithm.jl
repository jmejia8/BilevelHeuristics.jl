"""
    BLAlgorithm{T} <: Metaheuristics.AbstractAlgorithm

A bilevel algorithm wrapping algorithm-specific parameters (`T`), the current
[`BLState`](@ref), [`BLInformation`](@ref), and [`BLOptions`](@ref).

Users typically construct instances via the algorithm constructors ([`BCA`](@ref),
[`QBCA`](@ref), etc.) rather than directly.
"""
mutable struct BLAlgorithm{T} <: Metaheuristics.AbstractAlgorithm
    parameters::T
    status::BLState
    information::BLInformation
    options::BLOptions
end

function Algorithm(
        parameters;
        initial_state::BLState = BLState(nothing, []),
        information::BLInformation = BLInformation(),
        options::BLOptions = BLOptions(),
    )

    BLAlgorithm(parameters, initial_state, information, options)

end

Base.show(io::IO, blalgorithm::BLAlgorithm) = show(io, blalgorithm.parameters)

