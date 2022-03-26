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
        ul = Metaheuristics.SMS_EMOA(;N = 100), # upper level optimizer
        ll = Metaheuristics.NSGA2(;N = 50), # lower level optimizer
        ul_offsprings = 10,
        options_ul = Options(iterations = 100, f_calls_limit=Inf),
        options_ll = Options(iterations = 50, f_calls_limit=Inf)
    )

### Parameters

- `ul` is the upper level optimizer: `SMS_EMOA` is the unique valid optimizer.
- `ll` can be `NSGA2`, `SPEA2`, or `SMS_EMOA`.
- `ul_offsprings` is the number of new solutions generated for each generation

Optimal solutions can obtained from `res.population` when `res = optimize(...)`.
or via `method.parameters.archive`.

```julia-repl
julia> using BilevelHeuristics

julia> function F(x, y) # upper level
           [y[1] - x[1], y[2]], [-1.0 - sum(y)], [0.0]
       end
F (generic function with 1 method)

julia> function f(x, y) # lower level
           y, [-x[1]^2 + sum(y .^ 2)], [0.0]
       end
f (generic function with 1 method)

julia> bounds_ul = Array([0.0 1]') # upper level bounds for x
2×1 Matrix{Float64}:
 0.0
 1.0

julia> bounds_ll = [-1 -1; 1 1.0] # lower level bounds for y
2×2 Matrix{Float64}:
 -1.0  -1.0
  1.0   1.0

julia> method = SMS_MOBO()
SMS_MOBO{NSGA2}(SMS_EMOA(N=100, η_cr=20.0, p_cr=0.9, η_m=20.0, p_m=-1.0, n_samples=10000), NSGA2(N=50, η_cr=20.0, p_cr=0.9, η_m=20.0, p_m=-1.0), 10, Any[])

julia> res = optimize(F, f, bounds_ul, bounds_ll, method)
+=========== RESULT ==========+
  iteration: 100
population:
         ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀F space⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
         ┌────────────────────────────────────────┐ 
       1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   f₂    │⠦⢤⣤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│ 
         │⠀⠀⠈⠙⠒⠄⢄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠈⠙⠲⢄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢆⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      -1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         └────────────────────────────────────────┘ 
         ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀0⠀ 
         ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀f₁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
non-dominated solution(s):
         ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀F space⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
         ┌────────────────────────────────────────┐ 
       1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   f₂    │⠦⢤⣤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│ 
         │⠀⠀⠈⠙⠒⠄⢄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠈⠙⠲⢄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢆⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      -1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
         └────────────────────────────────────────┘ 
         ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀0⠀ 
         ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀f₁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
    F calls: 54500
    f calls: 2725000
    Message: Stopped due completed iterations. 
 total time: 22.1997 s
+============================+

julia> final_archive = method.parameters.archive;

julia> final_archive[1][1] # upper level (solution #1)
(f = [-1.7312900738859534, -0.1378039391413857], g = [-0.003743764044991882], h = [0.0], x = [0.872837777072331])

julia> final_archive[1][2] # lower level (solution #1)
(f = [-0.8584522968136225, -0.1378039391413857], g = [-0.005915513537101735], h = [0.0], x = [-0.8584522968136225, -0.1378039391413857])
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

    a = view(problem.ul.bounds, 1, :) # lower bounds (UL)
    b = view(problem.ul.bounds, 2, :) # upper bounds (UL)
    D = length(a)
    D_ll = size(problem.ll.bounds, 2)

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

