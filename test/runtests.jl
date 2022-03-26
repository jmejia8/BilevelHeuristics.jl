using Metaheuristics
using BilevelHeuristics
using Test
import Random: seed!
seed!(1)

include("heuristic.jl")
include("algorithms.jl")
include("problems.jl")

@testset "Algorithms" begin
    test_blemo()
    test_BCA()
    test_sms_mobo()
end

@testset "Heuristic" begin
    test_heuristic_MOBO2()
    test_heuristic_MOBO()
    test_heuristic_SVBO()
    test_heuristic_SOBO()
end

@testset "Test Problems" begin
    test_problems()
    test_ds_problems()
end
