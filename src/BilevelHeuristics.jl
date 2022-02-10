module BilevelHeuristics

export BCA, QBCA, QBCA2, SABO, optimize, minimum, minimizer
export TestProblems, BLEMO

# import dependencies
include("external.jl")

# core
include("core/core.jl")

include("common/common.jl")
include("optimize.jl")
include("TestProblems/TestProblems.jl")
include("BiApprox/BiApprox.jl")


# algorithms
include("algorithms/BCA/BCA.jl")
include("algorithms/QBCA/QBCA.jl")
include("algorithms/QBCA2/QBCA2.jl")
include("algorithms/SABO/SABO.jl")
include("algorithms/BLEMO/BLEMO.jl")


end # module
