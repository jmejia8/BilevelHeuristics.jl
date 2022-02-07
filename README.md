# BilevelHeuristics.jl


This package implements different heuristic and metaheuristic algorithms for 
Bilevel Optimization (BO).

[![Build Status](https://app.travis-ci.com/jmejia8/BilevelHeuristics.jl.svg?branch=main)](https://app.travis-ci.com/jmejia8/BilevelHeuristics.jl)

**BilevelHeuristics.jl is still in early development so please send feedback or open issues.**

## Installation


Open the Julia (Julia 1.6 or later) REPL and press `]` to open the Pkg prompt. To add this package, use the add command:


Type `]`  
```julia-repl
pkg> add https://github.com/jmejia8/BilevelHeuristics.jl.git
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("https://github.com/jmejia8/BilevelHeuristics.jl.git")
```

## Algorithms

Current implemented Bilevel Metaheuristics include:

- [x] [BCA](https://doi.org/10.1109/ROPEC.2018.8661368): Bilevel Centers Algorithm
- [x] [QBCA](https://doi.org/10.1109/CEC.2019.8790097): BCA with a lower-level Quasi-Newton optimization method.
- [x] [QBCA2](https://doi.org/10.1016/j.amc.2021.126577): Improved QBCA (implements conditions to avoid pseudo-feasible solutions)
- [x] [SABO](https://doi.org/10.1145/3377930.3390236): Surrogate-assisted Bilevel Optimization.
- [x]  BLEMO: Bilevel Evolutionary Multi-objective Optimization
- [ ]  SMS-MOBO: S-metic-selection-based Multi-objective Bilevel Optimization.

## Example

The following example illustrates the usage of `BilevelHeuristics.jl`. Here, `BCA` is used,
but this example works for the other optimizers.


Defining objective functions corresponding to the BO problem.

**Upper level (leader problem):**

```julia
using BilevelHeuristics

F(x, y) = sum(x.^2) + sum(y.^2)
bounds_ul = [-ones(5) ones(5)] 
```

**Lower level (follower problem):**

```julia
f(x, y) = sum((x - y).^2) + y[1]^2
bounds_ll = [-ones(5) ones(5)];
```
**Approximate solution:**

```julia
res = optimize(F, f, bounds_ul, bounds_ll, BCA())
```

**Output:**
```
+=========== RESULT ==========+
  iteration: 108
    minimum: 
          F: 4.03387e-10
          f: 2.94824e-10
  minimizer: 
          x: [-1.1460768817533927e-5, 7.231706879604178e-6, 3.818596951258517e-6, 2.294324313691869e-6, 1.8770952450067828e-6]
          y: [1.998748659975197e-6, 9.479307908087866e-6, 6.180041276047425e-6, -7.642051857319683e-6, 2.434166021682429e-6]
    F calls: 2503
    f calls: 5062617
    Message: Stopped due UL function evaluations limitations. 
 total time: 26.8142 s
+============================+
```

```julia
x, y = minimizer(res) # upper (x) and lower (y) level decision vectors
Fmin, fmin = minimum(res) # upper and lower objective values.
```
