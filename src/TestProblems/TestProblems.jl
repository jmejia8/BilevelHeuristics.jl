module TestProblems

export MMF, MTP, DS, TP

include("MMF/MMF.jl")
include("MTP/MTP.jl")
include("DS/DS.jl")
include("TP/TP.jl")
include("RealWorld/RealWorld.jl")

function get_problem(problem::AbstractString)
    if problem[1:2] == "TP"
        return TP.get_problem(parse(Int, problem[3:end]))
    elseif problem[1:2] == "DS"
        return DS.get_problem(parse(Int, problem[3:end]))
    elseif problem == "GoldMining"
        return RealWorld.GoldMining()
    else
        error("Undefined problem $problem")
    end
    
end


get_problem(problem::Symbol) = get_problem(String(problem))


end
