using Metaheuristics
using BilevelHeuristics
using Test
import Random: seed!
seed!(1)

include("DBMA.jl")
include("Nested.jl")
include("algorithms.jl")
include("problems.jl")

@testset "DBMA" begin
    test_dbma_MOBO()
end



@testset "Algorithms" begin
    test_blemo()
    test_BCA()
    test_sms_mobo()
end

@testset "Nested" begin
    test_nested_MOBO2()
    test_nested_MOBO()
    test_nested_SVBO()
    test_nested_SOBO()
end

@testset "Test Problems" begin
    test_problems()
    test_ds_problems()
end


