mutable struct BLProblem <: Metaheuristics.AbstractProblem
    ul::Metaheuristics.Problem
    ll::Metaheuristics.Problem
end

struct BLInformation
    ul::Metaheuristics.Information
    ll::Metaheuristics.Information
end

function Base.show(io::IO, blinfo::BLInformation)
    println(io, "Upper-level information:")
    Base.show(io, blinfo.ul)

    println(io, "\nLower-level information:")
    Base.show(io, blinfo.ll) 
end

struct BLOptions
    ul::Metaheuristics.Options
    ll::Metaheuristics.Options
end


function Base.show(io::IO, bloptions::BLOptions)
    println(io, "Upper-level options:")
    Base.show(io, bloptions.ul)

    println(io, "\nLower-level options:")
    Base.show(io, bloptions.ll) 
end

include("BLIndividual.jl")
include("BLState.jl")
include("BLAlgorithm.jl")

