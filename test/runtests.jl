using Metaheuristics
using BilevelHeuristics
using Test
import Random: seed!
seed!(1)

include("algorithms.jl")
include("problems.jl")

@testset "Algorithms" begin
    test_blemo()
    test_BCA()
end


@testset "Test Problems" begin
    test_problems()
    test_ds_problems()
end

