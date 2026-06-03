mutable struct SMS_MOBO{T} <: AbstractBLEMO where T <: Metaheuristics.AbstractParameters
    ul::SMS_EMOA
    ll::T
    ul_offsprings::Int
    archive # archive
end

include("family.jl")
include("compute-contribution.jl")
include("archiving.jl")


function get_ul_population(pop::Vector{Family{Metaheuristics.xFgh_indiv}})
    return [ s for sol in pop for s in sol.ul ]
end

get_ll_population(pop::Vector{Family{Metaheuristics.xFgh_indiv}}) = vcat([sol.lower_level_sols for sol in pop]...)

"""
    SMS_MOBO(;
        ul = Metaheuristics.SMS_EMOA(;N = 100),  # upper level optimizer
        ll = Metaheuristics.NSGA2(;N = 50),      # lower level optimizer
        ul_offsprings = 10,
        options_ul = Options(iterations = 100, f_calls_limit = Inf),
        options_ll = Options(iterations = 50, f_calls_limit = Inf),
    )

**S**-metric-selection **M**ulti-**O**bjective **B**ilevel **O**ptimization — described
in Mejía-de-Dios & Mezura-Montes (2022).

SMS_MOBO extends the S-metric selection idea (SMS-EMOA) to the bilevel setting.  Both the
upper and lower levels solve multi-objective problems:
- The **upper level** uses `SMS_EMOA` (hypervolume-based indicator) to drive the search
  toward a well-distributed Pareto front.
- The **lower level** uses a multi-objective evolutionary algorithm (`NSGA2`, `SPEA2`, or
  `SMS_EMOA`) to find the optimal response set for each upper-level decision.
- A **family** concept groups an upper-level `x` with its associated lower-level optimal
  `y` solutions, and a **super-rank** criterion combines dominance information from both
  levels for environmental selection.

The non-dominated upper-level solutions are stored in `method.parameters.archive` and are
also accessible via `res.population` after optimisation.

## Parameters
- `ul` — upper-level optimizer (`SMS_EMOA` is the only valid choice).
- `ll` — lower-level optimizer (`NSGA2`, `SPEA2`, or `SMS_EMOA`).
- `ul_offsprings` — number of new upper-level candidate solutions generated per
  generation (default `10`).

## Example

```julia
using BilevelHeuristics

function F(x, y)   # two upper-level objectives
    [y[1] - x[1], y[2]], [-1.0 - sum(y)], [0.0]
end

function f(x, y)   # two lower-level objectives
    y, [-x[1]^2 + sum(y .^ 2)], [0.0]
end

bounds_ul = [0.0 1.0]'
bounds_ll = [-1 -1; 1 1.0]

res = optimize(F, f, bounds_ul, bounds_ll, SMS_MOBO())
# Access the Pareto archive:
archive = SMS_MOBO().parameters.archive
```
"""
function SMS_MOBO(;
        ul = Metaheuristics.SMS_EMOA(;N = 100),
        ll = Metaheuristics.NSGA2(;N = 50),
        ul_offsprings = 10,
        options_ul = Metaheuristics.Options(iterations = 100, f_calls_limit=Inf),
        options_ll = Metaheuristics.Options(iterations = 50, f_calls_limit=Inf),
        information_ul = Metaheuristics.Information(),
        information_ll = Metaheuristics.Information()
    )

    parameters = SMS_MOBO(ul.parameters, ll.parameters, ul_offsprings, [])

    return Algorithm(parameters;
                     options = BLOptions(options_ul, options_ll),
                     information = BLInformation(information_ul, information_ll)
                    )
end

include("lower-level.jl")


function initialize!(
        status, # an initialized State (if apply)
        parameters::SMS_MOBO,
        problem,
        information,
        options,
        args...;
        kargs...
    )

    a = problem.ul.search_space.lb # lower bounds (UL)
    b = problem.ul.search_space.ub # upper bounds (UL)
    D = length(a)
    D_ll = Metaheuristics.getdim(problem.ll.search_space)

    N_ul = parameters.ul.N
    N_ll = parameters.ll.N

    population = Family{Metaheuristics.xFgh_indiv}[]

    options.ul.debug && @info "Generating $N_ul UL solutions..."
    for i in 1:N_ul
        options.ul.debug && @info "\t creating sol $i"
        x = a + (b - a) .* rand(D)

        y_members = lower_level_optimizer(status, parameters, problem, information, options, x)

        push!(population, create_family(x, y_members, problem))
    end

    update_super_rank!(population)

    options.ul.debug && @show length(population)

    return BLState(population[1], population)
end

function update_state!(
        status,
        parameters::SMS_MOBO,
        problem,
        information,
        options,
        args...;
        kargs...
    )
    N_ul = parameters.ul.N
    N_ll = parameters.ll.N


    # all different UL decision vector
    population_ul = [family.ul[1] for family in status.population]

    options.ul.debug && @info "Reproduction"
    # get UL sols from archive with different x
    for i = 1:parameters.ul_offsprings

        # let's perform the matings at upper level
        x = blemo_genetic_operators(population_ul, parameters.ul, problem.ul, selection = :tournament)

        # y_members for the new `x`
        Q_ll = Vector[] 

        R_ll = [s for family in status.population for s in family.lower_level_sols]

        # let's perform the matings at lower level
        for i = 1:N_ll
            c = blemo_genetic_operators(R_ll, parameters.ll, problem.ll, selection = :random)
            # save children
            push!(Q_ll, c)
        end


        # step 2: optimize using LL sols in families
        lower_level_sols = lower_level_optimizer(status,
                                              parameters,
                                              problem,
                                              information,
                                              options,
                                              x, Q_ll)

        new_family = create_family(x, lower_level_sols, problem)

        # save new_family
        push!(status.population, new_family)

    end
         

    options.ul.debug && @info "Environmental selection"
    # remove worst element in population
    reduce_population!(status.population, parameters)

    options.ul.debug && @info "Archiving"
    # FIXME: performance improvement
    sms_update_archive!(parameters.archive, status.population)

    if options.ul.debug
        nsr = length(unique([fam.super_rank for fam in status.population]))
        NN = length(status.population)
        @info "|Archive| = $(length(parameters.archive))."
        @info "There are $nsr different fronts in population."
        @info "There are $NN families in population."


        println("Archive:") 
        A_ul = [s[1] for s in parameters.archive]
        display(A_ul)
    end
end

# We want remove a x from population
function reduce_population!(population, parameters::SMS_MOBO)
    update_super_rank!(population)
    sort!(population, by = s -> s.super_rank)

    rnk = 1
    ind = 0
    indnext = findlast(family -> family.super_rank == rnk, population)
    while !isnothing(indnext) && 0 < indnext <= parameters.ul.N
        ind = indnext
        rnk += 1
        indnext = findlast(x -> x.super_rank == rnk, population)
    end
    isnothing(indnext) && (indnext = length(population)) 

    # remove worst solutions (after last front)
    indnext < length(population) && deleteat!(population, indnext+1:length(population))

    while length(population) > parameters.ul.N
        last_front = ind+1:length(population)
        update_contribution!(population, last_front, parameters.ul.n_samples)
        worst = argmin(map(s -> s.contribution, population))
        # remove worst sub-front
        deleteat!(population, worst)
    end
end

