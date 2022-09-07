module BilevelHeuristics

export BCA, QBCA, QBCA2, SABO, optimize, minimum, minimizer
export TestProblems, BLEMO, SMS_MOBO

export get_ll_population, get_ul_population,ulvector,llvector,ulfval,llfval,ulpositions
export ulpositions,llpositions,is_pseudo_feasible
export ulfvals, ulgvals, ulhvals, llfvals, llgvals, llhvals
export BLState, BLOptions, BLInformation
export Nested
export DBMA


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
include("algorithms/SMS_MOBO/SMS_MOBO.jl")

# framework
include("algorithms/Nested/Nested.jl")
include("algorithms/DBMA/DBMA.jl")

# deprecations
@deprecate Heuristic Nested
@deprecate Heuristic(;kargs...) Nested(;kargs...)


end # module
