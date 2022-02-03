FF(x, y) = (x[1] - 1.0)^2 + sum(x[2:end].^2) - sum( ( x[1]^2 - sum(y.^2) ).^2 )
ff(x, y) = x[2]^2 + sum( ( x[1]^2 - sum(y.^2) ).^2 )

function test_BCA()
    
    bounds_ul = bounds_ll = [-5ones(3) 5ones(3)]'
    f_calls = F_calls = 0

    F(x, y) = begin
        F_calls += 1
        return FF(x, y)
    end


    f(x, y) = begin
        f_calls += 1
        return ff(x, y)
    end


    D_ul = size(bounds_ul, 2)
    D_ll = size(bounds_ll, 2)

    options_ul = Options(f_tol = 1e-2, iterations = 2, debug = false, seed=1)
    options_ll = Options(f_tol = 1e-3)

    information_ul = Information(f_optimum = 0.0)
    information_ll = Information(f_optimum = 0.0)
    
    methods = [
               # SABO(;options_ul, options_ll, information_ul, information_ll),
               QBCA2(N = 14;options_ul, options_ll, information_ul, information_ll),
               QBCA( N = 14;options_ul, options_ll, information_ul, information_ll),
               BCA(  N = 14;options_ul, options_ll, information_ul, information_ll),
              ]

    for method in methods
        f_calls = F_calls = 0
        r = optimize( F, f, bounds_ul, bounds_ll, method)

        Fxy, fxy = minimum(r)
        # @test isapprox(Fxy, 0.0, atol=options_ul.f_tol)
        # @test isapprox(fxy, 0.0, atol=options_ll.f_tol)
        @test F_calls == r.F_calls
        @test f_calls == r.f_calls
    end

end


function test_blemo()
    fnum = 1
    F, f, bounds_ul, bounds_ll, Ψ, UL_sols = TestProblems.TP.get_problem(fnum)

    options_ul = Options(iterations = 2, debug = false, seed=1)
    options_ll = Options(iterations = 40)

    nsga2_ul = NSGA2(N = 400, p_m = 0.1, η_m = 20, η_cr = 15, p_cr = 0.9)
    nsga2_ll = NSGA2(N = 40,  p_m = 0.1, η_m = 20, η_cr = 15, p_cr = 0.9)

    method = BLEMO(;ul = nsga2_ul, ll = nsga2_ll, options_ul, options_ll)
    r = optimize(F, f, bounds_ul, bounds_ll, method)
    
    A_ul = map(s -> s[1], method.parameters.archive)

    @test length(A_ul) > 0

    
end

