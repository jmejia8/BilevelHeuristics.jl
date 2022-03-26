var documenterSearchIndex = {"docs":
[{"location":"api/#API-References","page":"API References","title":"API References","text":"","category":"section"},{"location":"api/","page":"API References","title":"API References","text":"optimize","category":"page"},{"location":"api/#Metaheuristics.optimize","page":"API References","title":"Metaheuristics.optimize","text":"  optimize(\n        f::Function, # objective function\n        bounds::AbstractMatrix,\n        method::AbstractAlgorithm = ECA();\n        logger::Function = (status) -> nothing,\n  )\n\nMinimize a n-dimensional function f with domain bounds (2×n matrix) using method = ECA() by default.\n\nExample\n\nMinimize f(x) = Σx² where x ∈ [-10, 10]³.\n\nSolution:\n\njulia> f(x) = sum(x.^2)\nf (generic function with 1 method)\n\njulia> bounds = [  -10.0 -10 -10; # lower bounds\n                    10.0  10 10 ] # upper bounds\n2×3 Array{Float64,2}:\n -10.0  -10.0  -10.0\n  10.0   10.0   10.0\n\njulia> result = optimize(f, bounds)\n+=========== RESULT ==========+\n  iteration: 1429\n    minimum: 2.5354499999999998e-222\n  minimizer: [-1.5135301653303966e-111, 3.8688354844737692e-112, 3.082095708730726e-112]\n    f calls: 29989\n total time: 0.1543 s\n+============================+\n\n\n\n\n\noptimize(F, f, bounds_ul, bounds_ll, method = BCA(); logger = (status) -> nothing)\n\nApproximate an optimal solution for the bilevel optimization problem x ∈ argmin F(x, y) with x ∈ bounds_ul subject to y ∈ argmin{f(x,y) : y ∈ bounds_ll}.\n\nParameters\n\nF upper-level objective function.\nf lower-level objective function.\nbounds_ul, bounds_ll upper and lower level boundaries (2×n matrices), respectively.\nlogger is a functions called at the end of each iteration.\n\nExample\n\njulia> F(x, y) = sum(x.^2) + sum(y.^2)\nF (generic function with 1 method)\n\njulia> f(x, y) = sum((x - y).^2) + y[1]^2\nf (generic function with 1 method)\n\njulia> bounds_ul = bounds_ll = [-ones(5)'; ones(5)']\n2×5 Matrix{Float64}:\n -1.0  -1.0  -1.0  -1.0  -1.0\n  1.0   1.0   1.0   1.0   1.0\n\njulia> res = optimize(F, f, bounds_ul, bounds_ll)\n+=========== RESULT ==========+\n  iteration: 108\n    minimum: \n          F: 7.68483e-08\n          f: 3.96871e-09\n  minimizer: \n          x: [1.0283390421119262e-5, -0.00017833559080058394, -1.612275010196171e-5, 0.00012064585960330227, 4.38964383738248e-5]\n          y: [1.154609166391327e-5, -0.0001300400306798623, 1.1811981430188257e-6, 8.868498295184257e-5, 5.732849695863675e-5]\n    F calls: 2503\n    f calls: 5044647\n    Message: Stopped due UL function evaluations limitations. \n total time: 21.4550 s\n+============================+\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"BLInformation","category":"page"},{"location":"api/#BilevelHeuristics.BLInformation","page":"API References","title":"BilevelHeuristics.BLInformation","text":"BLInformation(ul, ll)\n\nBLInformation stores information Information about problems at each level (upper and lower level).\n\n\n\n\n\n","category":"type"},{"location":"api/","page":"API References","title":"API References","text":"BLOptions","category":"page"},{"location":"api/#BilevelHeuristics.BLOptions","page":"API References","title":"BilevelHeuristics.BLOptions","text":"BLOptions(ul, ll)\n\nBLOptions stores common settings Options for metaheuristics at each level (upper and lower level).\n\n\n\n\n\n","category":"type"},{"location":"api/","page":"API References","title":"API References","text":"get_ll_population","category":"page"},{"location":"api/#BilevelHeuristics.get_ll_population","page":"API References","title":"BilevelHeuristics.get_ll_population","text":"get_ll_population(population)\n\nReturn the lower level solutions.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":" get_ul_population","category":"page"},{"location":"api/#BilevelHeuristics.get_ul_population","page":"API References","title":"BilevelHeuristics.get_ul_population","text":"get_ul_population(population)\n\nReturn the upper level solutions.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"ulvector","category":"page"},{"location":"api/#BilevelHeuristics.ulvector","page":"API References","title":"BilevelHeuristics.ulvector","text":"ulvector(A)\n\nGet upper-level decision vector.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"llvector","category":"page"},{"location":"api/#BilevelHeuristics.llvector","page":"API References","title":"BilevelHeuristics.llvector","text":"llvector(A)\n\nGet lower-level decision vector.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"ulfval","category":"page"},{"location":"api/#BilevelHeuristics.ulfval","page":"API References","title":"BilevelHeuristics.ulfval","text":"ulfval(A)\n\nGet upper-level function value.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"llfval","category":"page"},{"location":"api/#BilevelHeuristics.llfval","page":"API References","title":"BilevelHeuristics.llfval","text":"llfval(A)\n\nGet lower-level function value.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"ulfvals","category":"page"},{"location":"api/","page":"API References","title":"API References","text":"llfvals","category":"page"},{"location":"api/","page":"API References","title":"API References","text":"ulgvals","category":"page"},{"location":"api/#BilevelHeuristics.ulgvals","page":"API References","title":"BilevelHeuristics.ulgvals","text":"ulgvals(pop)\n\nGet upper-level inequality constraints.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"llgvals","category":"page"},{"location":"api/#BilevelHeuristics.llgvals","page":"API References","title":"BilevelHeuristics.llgvals","text":"llgvals(pop)\n\nGet lower-level inequality constraints.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"ulhvals","category":"page"},{"location":"api/#BilevelHeuristics.ulhvals","page":"API References","title":"BilevelHeuristics.ulhvals","text":"ulhvals(pop)\n\nGet upper-level equality constraints.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"llhvals","category":"page"},{"location":"api/#BilevelHeuristics.llhvals","page":"API References","title":"BilevelHeuristics.llhvals","text":"llhvals(pop)\n\nGet lower-level equality constraints.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"ulpositions","category":"page"},{"location":"api/#BilevelHeuristics.ulpositions","page":"API References","title":"BilevelHeuristics.ulpositions","text":"ulpositions(population)\n\nGet upper-level decision vectors from population.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"llpositions","category":"page"},{"location":"api/#BilevelHeuristics.llpositions","page":"API References","title":"BilevelHeuristics.llpositions","text":"llpositions(population)\n\nGet lower-level decision vectors from population.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API References","title":"API References","text":"is_pseudo_feasible","category":"page"},{"location":"api/#BilevelHeuristics.is_pseudo_feasible","page":"API References","title":"BilevelHeuristics.is_pseudo_feasible","text":"is_pseudo_feasible(A, B, δ1, δ2, ε1, ε2)\n\nCheck whether A is a pseudo-feasible solution respect to B.\n\n\n\n\n\n","category":"function"},{"location":"examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Here, some examples are presented to illustrate some study cases.","category":"page"},{"location":"examples/#Constrained-Problems","page":"Examples","title":"Constrained Problems","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Here, equality and inequality constraints are defined as:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Constraints at upper-level:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"G_i(xy) leq 0  i = 12ldotsI","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"H_j(xy) = 0  j = 12ldotsJ","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Constraints at lower-level:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"g_k(xy) leq 0  i = 12ldotsK","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"h_l(xy) = 0  j = 12ldotsL","category":"page"},{"location":"examples/#Implementation","page":"Examples","title":"Implementation","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Upper level problem: F(x,y) with x as the upper-level vector.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"julia> function F(x, y)\n  Fxy = sum(x.^2) + sum(y.^2)\n  Gxy = [ x[1] + x[2] - 1,  x[3] - x[3] - 10]\n  Hxy = [0.0]\n  return Fxy, Gxy, Hxy\nend\nF (generic function with 1 method)\n\njulia> bounds_ul = [-ones(5) ones(5)];\n","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Lower level problem: f(x, y) with y as the lower-level vector.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"julia> function f(x, y) \n  fxy = sum((x - y).^2) + y[1]^2\n  gxy = [x[2] - y[1]^2 - 5]\n  hxy = [0.0]\n  fxy, gxy, hxy\nend\nf (generic function with 1 method)\n\njulia> bounds_ll = [-ones(5) ones(5)];\n","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Approximate solution.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"julia> res = optimize(F, f, bounds_ul, bounds_ll, BCA())\n+=========== RESULT ==========+\n  iteration: 108\n    minimum: \n          F: 4.08164e-10\n          f: 3.49457e-10\n  minimizer: \n          x: [1.821445118847534e-6, 9.431162141567291e-6, 5.039441103662204e-6, 1.2440713582037751e-5, -9.430574843388418e-6]\n          y: [5.173759982237097e-6, -1.6496788937326975e-6, -1.591480747137886e-6, 2.1659898236762077e-6, -3.175376796124624e-6]\n    F calls: 2503\n    f calls: 5123090\n    Message: Stopped due UL function evaluations limitations. \n total time: 32.3749 s\n+============================+\n\n\njulia> res.best_sol\nUpper-level:\n(f = 4.0816e-10, g = [-0.9999887473927396, -10.0], h = [0.0], x = [1.821e-06, 9.431e-06, …, -9.431e-06])\nLower-level:\n(f = 3.4946e-10, g = [-4.999990568864626], h = [0.0], x = [5.174e-06, -1.650e-06, …, -3.175e-06])","category":"page"},{"location":"#BilevelHeuristics-Heuristics-and-Metaheuristics-for-Bilevel-Optimization","page":"Index","title":"BilevelHeuristics - Heuristics and Metaheuristics for Bilevel Optimization","text":"","category":"section"},{"location":"","page":"Index","title":"Index","text":"Author: Jesus Mejía (@jmejia8)","category":"page"},{"location":"","page":"Index","title":"Index","text":"Approximate algorithms for bilevel optimization.","category":"page"},{"location":"","page":"Index","title":"Index","text":"(Image: Source) (Image: Build Status)","category":"page"},{"location":"#Introduction","page":"Index","title":"Introduction","text":"","category":"section"},{"location":"","page":"Index","title":"Index","text":"Bilevel Optimization is a very challenging task that require high-performance algorithms to optimize hierarchical problem. This package implements a variety of approximate  algorithms for bilevel optimization.","category":"page"},{"location":"","page":"Index","title":"Index","text":"BilevelHeuristics extends the Metaheuristics.jl API to implement bilevel optimization algorithms.","category":"page"},{"location":"#Installation","page":"Index","title":"Installation","text":"","category":"section"},{"location":"","page":"Index","title":"Index","text":"Open the Julia (Julia 1.6 or later) REPL and press ] to open the Pkg prompt. To add this package, use the add command:","category":"page"},{"location":"","page":"Index","title":"Index","text":"pkg> add BilevelHeuristics","category":"page"},{"location":"","page":"Index","title":"Index","text":"Or, equivalently, via the Pkg API:","category":"page"},{"location":"","page":"Index","title":"Index","text":"julia> import Pkg; Pkg.add(\"BilevelHeuristics\")","category":"page"},{"location":"#Example","page":"Index","title":"Example","text":"","category":"section"},{"location":"","page":"Index","title":"Index","text":"julia> F(x, y) = sum(x.^2) + sum(y.^2)\nF (generic function with 1 method)\n\njulia> f(x, y) = sum((x - y).^2) + y[1]^2\nf (generic function with 1 method)\n\njulia> bounds_ul = bounds_ll = [-ones(5)'; ones(5)']\n2×5 Matrix{Float64}:\n -1.0  -1.0  -1.0  -1.0  -1.0\n  1.0   1.0   1.0   1.0   1.0\n\njulia> res = optimize(F, f, bounds_ul, bounds_ll)\n+=========== RESULT ==========+\n  iteration: 108\n    minimum: \n          F: 7.68483e-08\n          f: 3.96871e-09\n  minimizer: \n          x: [1.0283390421119262e-5, -0.00017833559080058394, -1.612275010196171e-5, 0.00012064585960330227, 4.38964383738248e-5]\n          y: [1.154609166391327e-5, -0.0001300400306798623, 1.1811981430188257e-6, 8.868498295184257e-5, 5.732849695863675e-5]\n    F calls: 2503\n    f calls: 5044647\n    Message: Stopped due UL function evaluations limitations. \n total time: 21.4550 s\n+============================+","category":"page"},{"location":"algorithms/#Algorithms","page":"Algorithms","title":"Algorithms","text":"","category":"section"},{"location":"algorithms/#BCA","page":"Algorithms","title":"BCA","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"BCA","category":"page"},{"location":"algorithms/#BilevelHeuristics.BCA","page":"Algorithms","title":"BilevelHeuristics.BCA","text":"BCA(;N, n, K, η_max, resize_population)\n\nBilevel Centers Algorithm uses two nested ECA.\n\nParameters\n\nN Upper level population size\nn Lower level population size.\nK Num. of solutions to generate centers.\nη_max Step size.\n\nUsage\n\nUpper level problem: F(x,y) with x as the upper-level vector.\n\njulia> F(x, y) = sum(x.^2) + sum(y.^2)\nF (generic function with 1 method)\n\njulia> bounds_ul = [-ones(5) ones(5)];\n\n\nLower level problem: f(x, y) with y as the lower-level vector.\n\njulia> f(x, y) = sum((x - y).^2) + y[1]^2\nf (generic function with 1 method)\n\njulia> bounds_ll = [-ones(5) ones(5)];\n\n\nApproximate solution.\n\njulia> res = optimize(F, f, bounds_ul, bounds_ll, BCA())\n+=========== RESULT ==========+\n  iteration: 108\n    minimum: \n          F: 2.7438e-09\n          f: 3.94874e-11\n  minimizer: \n          x: [-8.80414308649828e-6, 2.1574853199308744e-5, -1.5550602817418899e-6, 1.9314104453973864e-5, 2.1709393089480435e-5]\n          y: [-4.907639660543081e-6, 2.173986368018122e-5, -1.8133242873785074e-6, 1.9658451600356374e-5, 2.1624363965042988e-5]\n    F calls: 2503\n    f calls: 6272518\n    Message: Stopped due UL function evaluations limitations. \n total time: 14.8592 s\n+============================+\n\njulia> x, y = minimizer(res);\n\njulia> x\n5-element Vector{Float64}:\n -8.80414308649828e-6\n  2.1574853199308744e-5\n -1.5550602817418899e-6\n  1.9314104453973864e-5\n  2.1709393089480435e-5\n\njulia> y\n5-element Vector{Float64}:\n -4.907639660543081e-6\n  2.173986368018122e-5\n -1.8133242873785074e-6\n  1.9658451600356374e-5\n  2.1624363965042988e-5\n\njulia> Fmin, fmin = minimum(res)\n(2.7438003987697017e-9, 3.9487399650845625e-11)\n\nCitation\n\nMejía-de-Dios, J. A., & Mezura-Montes, E. (2018, November). A physics-inspired algorithm for bilevel optimization. In 2018 IEEE International Autumn Meeting on Power, Electronics and Computing (ROPEC) (pp. 1-6). IEEE.\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#QBCA","page":"Algorithms","title":"QBCA","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"QBCA","category":"page"},{"location":"algorithms/#BilevelHeuristics.QBCA","page":"Algorithms","title":"BilevelHeuristics.QBCA","text":"QBCA(;N, K, η_ul_max, η_ll_max, α, β, autodiff)\n\nQuasi-newton BCA uses ECA (upper-level) and BFGS (lower-level).\n\nParameters\n\nN Upper level population size\nK Num. of solutions to generate centers.\nη_ul_max UL step size.\nη_ll_max LL step size.\nα, β Parameters for the Tikhnonov regularization.\nautodiff=:finite Used to approximate LL derivates.\n\nUsage\n\nUpper level problem: F(x,y) with x as the upper-level vector.\n\njulia> F(x, y) = sum(x.^2) + sum(y.^2)\nF (generic function with 1 method)\n\njulia> bounds_ul = [-ones(5) ones(5)];\n\n\nLower level problem: f(x, y) with y as the lower-level vector.\n\njulia> f(x, y) = sum((x - y).^2) + y[1]^2\nf (generic function with 1 method)\n\njulia> bounds_ll = [-ones(5) ones(5)];\n\n\nApproximate solution.\n\njulia> res = optimize(F, f, bounds_ul, bounds_ll, QBCA())\n+=========== RESULT ==========+\n  iteration: 71\n    minimum: \n          F: 1.20277e-06\n          f: 1.8618e-08\n  minimizer: \n          x: [-0.00019296602928680934, -0.00031720504506331244, 0.00047217689470620765, 0.00014459596611862214, 0.00048345619641040644]\n          y: [-9.647494056567316e-5, -0.0003171519406858993, 0.00047209784939209284, 0.00014457176048263256, 0.0004833752613377002]\n    F calls: 2130\n    f calls: 366743\n    Message: Stopped due UL small fitness variance. \n total time: 7.7909 s\n+============================+\n\njulia> x, y = minimizer(res);\n\njulia> x\n5-element Vector{Float64}:\n -0.00019296602928680934\n -0.00031720504506331244\n  0.00047217689470620765\n  0.00014459596611862214\n  0.00048345619641040644\n\njulia> y\n5-element Vector{Float64}:\n -9.647494056567316e-5\n -0.0003171519406858993\n  0.00047209784939209284\n  0.00014457176048263256\n  0.0004833752613377002\n\njulia> Fmin, fmin = minimum(res)\n(1.2027656204730873e-6, 1.8617960564375732e-8)\n\nCitation\n\nMejía-de-Dios, J. A., & Mezura-Montes, E. (2019, June). A metaheuristic for bilevel optimization using tykhonov regularization and the quasi-newton method. In 2019 IEEE Congress on Evolutionary Computation (CEC) (pp. 3134-3141). IEEE.\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#QBCA2","page":"Algorithms","title":"QBCA2","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"QBCA2","category":"page"},{"location":"algorithms/#BilevelHeuristics.QBCA2","page":"Algorithms","title":"BilevelHeuristics.QBCA2","text":"QBCA2(;N, K, η_max, δ1, δ2, ε1, ε2, λ, t_reevaluate)\n\nQuasi-newton BCA uses ECA (upper-level) and BFGS (lower-level).\n\nParameters\n\nN Upper level population size\nK Num. of solutions to generate centers.\nη_max Step size\nδ1, δ2, ε1 ε2 Parameters for conditions to avoid pseudo-feasible solutions.\n\nUsage\n\nUpper level problem: F(x,y) with x as the upper-level vector.\n\njulia> F(x, y) = sum(x.^2) + sum(y.^2)\nF (generic function with 1 method)\n\njulia> bounds_ul = [-ones(5) ones(5)];\n\n\nLower level problem: f(x, y) with y as the lower-level vector.\n\njulia> f(x, y) = sum((x - y).^2) + y[1]^2\nf (generic function with 1 method)\n\njulia> bounds_ll = [-ones(5) ones(5)];\n\n\nApproximate solution.\n\njulia> res = optimize(F, f, bounds_ul, bounds_ll, QBCA2())\n+=========== RESULT ==========+\n  iteration: 59\n    minimum: \n          F: 3.95536e-07\n          f: 9.2123e-11\n  minimizer: \n          x: [-1.3573722472608445e-5, -0.00012074600446520904, -0.00035025471067487137, -0.0002315301345354928, -8.239473503719106e-5]\n          y: [-6.786860530140986e-6, -0.00012074599672993571, -0.00035025467380887673, -0.00023153010993486042, -8.239472729799499e-5]\n    F calls: 1775\n    f calls: 299157\n    Message: Stopped due UL small fitness variance. \n total time: 5.6968 s\n+============================+\n\njulia> x, y = minimizer(res);\n\njulia> x\n5-element Vector{Float64}:\n -1.3573722472608445e-5\n -0.00012074600446520904\n -0.00035025471067487137\n -0.0002315301345354928\n -8.239473503719106e-5\n\njulia> y\n5-element Vector{Float64}:\n -6.786860530140986e-6\n -0.00012074599672993571\n -0.00035025467380887673\n -0.00023153010993486042\n -8.239472729799499e-5\n\njulia> Fmin, fmin = minimum(res)\n(3.9553637806596925e-7, 9.212297088378278e-11)\n\nCitation\n\nMejía-de-Dios, J. A., Mezura-Montes, E., & Toledo-Hernández, P. (2022). Pseudo-feasible solutions in evolutionary bilevel optimization: Test problems and performance assessment. Applied Mathematics and Computation, 412, 126577.\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#SABO","page":"Algorithms","title":"SABO","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"SABO","category":"page"},{"location":"algorithms/#BilevelHeuristics.SABO","page":"Algorithms","title":"BilevelHeuristics.SABO","text":"SABO(;N, K, η_max, δ1, δ2, ε1, ε2, λ, t_reevaluate)\n\nSurrogate Algorithm for Bilevel Optimization.\n\nParameters\n\nN Upper level population size\nK Num. of solutions to generate centers.\nη_max Step size\nδ1, δ2, ε1 ε2 Parameters for conditions to avoid pseudo-feasible solutions.\nλ Parameter for the surrogate model.\nt_reevaluate Indicates how many iterations is reevaluated the lower level.\n\nUsage\n\nUpper level problem: F(x,y) with x as the upper-level vector.\n\njulia> F(x, y) = sum(x.^2) + sum(y.^2)\nF (generic function with 1 method)\n\njulia> bounds_ul = [-ones(5) ones(5)];\n\n\nLower level problem: f(x, y) with y as the lower-level vector.\n\njulia> f(x, y) = sum((x - y).^2) + y[1]^2\nf (generic function with 1 method)\n\njulia> bounds_ll = [-ones(5) ones(5)];\n\n\nApproximate solution.\n\njulia> using Metaheuristics\n\njulia> res = optimize(F, f, bounds_ul, bounds_ll, SABO(options_ul=Options(iterations=12)))\n+=========== RESULT ==========+\n  iteration: 12\n    minimum: \n          F: 0.00472028\n          f: 0.000641749\n  minimizer: \n          x: [-0.03582594991950816, 0.018051141584692676, -0.030154879329873152, -0.017337812299467736, 0.004710839249040477]\n          y: [-0.017912974972476316, 0.018051141514328663, -0.030154879385452187, -0.017337812317661405, 0.004710839272021738]\n    F calls: 372\n    f calls: 513936\n    Message: Stopped due completed iterations. \n total time: 19.0654 s\n+============================+\n\njulia> x, y = minimizer(res);\n\njulia> x\n5-element Vector{Float64}:\n -0.03582594991950816\n  0.018051141584692676\n -0.030154879329873152\n -0.017337812299467736\n  0.004710839249040477\n\njulia> y\n5-element Vector{Float64}:\n -0.017912974972476316\n  0.018051141514328663\n -0.030154879385452187\n -0.017337812317661405\n  0.004710839272021738\n\njulia> Fmin, fmin = minimum(res)\n(0.004720277765002139, 0.0006417493438175533)\n\nCitation\n\nMejía-de-Dios, J. A., & Mezura-Montes, E. (2020, June). A surrogate-assisted metaheuristic for bilevel optimization. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference (pp. 629-635).\n\n\n\n\n\n","category":"type"}]
}
