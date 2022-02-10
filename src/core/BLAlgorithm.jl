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

