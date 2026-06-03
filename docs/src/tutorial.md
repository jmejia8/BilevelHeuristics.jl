# Tutorial: Multi-objective Bilevel Optimization

This tutorial demonstrates how to use the [`SMS_MOBO`](@ref) algorithm for
multi-objective bilevel optimization.

## 1. Define the bilevel optimization problem

We consider a multi-objective bilevel problem with two objectives at each level.

**Upper level (leader):**
```math
\begin{aligned}
\min_{x,y} \; & F(x,y) = [y_1 - x_1,\; y_2] \\
\text{s.t.} \; & G(x,y) = -1 - y_1 - y_2 \leq 0 \\
& x \in [0, 1]
\end{aligned}
```

**Lower level (follower):**
```math
\begin{aligned}
\min_{y} \; & f(x,y) = [y_1,\; y_2] \\
\text{s.t.} \; & g(x,y) = -x_1^2 + y_1^2 + y_2^2 \leq 0 \\
& y \in [-1, 1]^2
\end{aligned}
```

```@repl sms_mobo
using BilevelHeuristics

function F(x, y)
    [y[1] - x[1], y[2]], [-1.0 - sum(y)], [0.0]
end

function f(x, y)
    y, [-x[1]^2 + sum(y .^ 2)], [0.0]
end

bounds_ul = Array([0.0 1]')
bounds_ll = [-1 -1; 1 1.0]
```

## 2. Configure and run SMS_MOBO

The algorithm uses `SMS_EMOA` at the upper level and `NSGA2` at the lower level by default.

```@repl sms_mobo
method = SMS_MOBO(;
    ul_offsprings = 3,
    options_ul = Options(iterations = 10, f_calls_limit = Inf, verbose = false),
    options_ll = Options(iterations = 15, f_calls_limit = Inf),
)
```

Trace convergence with a custom logger that records non-dominated front sizes:

```@repl sms_mobo
logger_hist = []
function my_logger(status) # this function is called once after each iteration
    iter = status.iteration
    # let's compute the UL population Pareto front
    ul_pop = get_ul_population(status.population)
    nd_mask = Metaheuristics.get_non_dominated_solutions_perm(ul_pop)
    nd_fvals = Metaheuristics.fvals(ul_pop[nd_mask])
    feasible_mask = Metaheuristics.is_feasible.(ul_pop[nd_mask])
    nd_feasible = nd_fvals[feasible_mask, :]
    # save some info at current interation
    push!(logger_hist, (
        iter = iter,
        n_nd = size(nd_fvals, 1),
        n_feasible = size(nd_feasible, 1),
    ))
end

res = optimize(F, f, bounds_ul, bounds_ll, method; logger = my_logger)
```

## 3. Analyze convergence

Print the first and last entries of the convergence history:

```@repl sms_mobo
n = length(logger_hist)
for i in vcat(1:min(3, n), max(n-2, 4):n)
    e = logger_hist[i]
    println("iter=$(e.iter): n_nd=$(e.n_nd)  n_feasible=$(e.n_feasible)")
end
```

## 4. Extract the feasible Pareto front and compute hypervolume

```@repl sms_mobo
ul_pop = get_ul_population(res.population);
feasible_mask = Metaheuristics.is_feasible.(ul_pop);
feasible_pop = ul_pop[feasible_mask];
nd_feasible_mask = Metaheuristics.get_non_dominated_solutions_perm(feasible_pop);
pf_feasible = Metaheuristics.fvals(feasible_pop[nd_feasible_mask])

println("Feasible PF size: ", size(pf_feasible, 1))
for i in 1:min(3, size(pf_feasible, 1)) # First 3 solutions:
    println("  ", pf_feasible[i, :])
end
```

Compute the hypervolume indicator with respect to a reference point:

```@repl sms_mobo
ref_point = [0.0, 2.0]
hv = Metaheuristics.PerformanceIndicators.hypervolume(pf_feasible, ref_point)
println("HV (ref = $ref_point) = $hv")
```

## 5. Save data as CSV

Export the full archive `[X Y F f G g H h]` using `DelimitedFiles.writedlm`:

```@repl sms_mobo
using DelimitedFiles

final_archive = method.parameters.archive

X = Metaheuristics.positions([sol[1] for sol in final_archive]);
Y = Metaheuristics.positions([sol[2] for sol in final_archive]);
F_vals = Metaheuristics.fvals([sol[1] for sol in final_archive]);
f_vals = Metaheuristics.fvals([sol[2] for sol in final_archive]);
G = Metaheuristics.gvals([sol[1] for sol in final_archive]);
g = Metaheuristics.gvals([sol[2] for sol in final_archive]);
H = Metaheuristics.hvals([sol[1] for sol in final_archive]);
h = Metaheuristics.hvals([sol[2] for sol in final_archive]);

data = hcat(X, Y, F_vals, f_vals, G, g, H, h)
header = vcat(
    ["x$i" for i in 1:size(X, 2)],
    ["y$i" for i in 1:size(Y, 2)],
    ["F$i" for i in 1:size(F_vals, 2)],
    ["f$i" for i in 1:size(f_vals, 2)],
    ["G$i" for i in 1:size(G, 2)],
    ["g$i" for i in 1:size(g, 2)],
    ["H$i" for i in 1:size(H, 2)],
    ["h$i" for i in 1:size(h, 2)],
);

open("sms_mobo_results.csv", "w") do io
    writedlm(io, permutedims(header), ',')
    writedlm(io, data, ',')
end

println("Saved $(size(data,1)) rows to sms_mobo_results.csv")
println("Columns: ", join(header, ", "))
```

Reload the saved data:

```@repl sms_mobo
reloaded = readdlm("sms_mobo_results.csv", ',', skipstart = 1)
println("Reloaded $(size(reloaded,1)) rows × $(size(reloaded,2)) columns")
```
